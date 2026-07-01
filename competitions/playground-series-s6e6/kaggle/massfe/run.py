"""s6e6 massfe: MASS feature-engineering GBDTs (LightGBM + CatBoost).

Strategy: lift the base ceiling by throwing a *large* engineered feature set at
two strong GBDTs, with the ORIGINAL SDSS17 dataset appended to each fold's TRAIN
pool (weight 0.1) and balanced-accuracy-aware early stopping.  This is the Kaggle
grandmaster "mass FE" lever, including groupby AGGREGATION features (group
mean/std/count + deviation of the key colors/redshift over coarse bins) that the
existing pipeline never used.

Self-contained Kaggle SCRIPT kernel (CPU).  Does NOT import the pipeline modules
or kernel_common (so it only needs the original-SDSS17 dataset attached).

All feature engineering is pure pandas/numpy (NO cuDF — the P100 dropped Pascal).
CatBoost is GPU-guarded via catboost.utils.get_gpu_device_count(); this kernel
runs CPU (enable_gpu=false) so it transparently falls to CPU.

Stack-alignment contract (so the produced OOF/test combine with every other model):
  * Labels: GALAXY=0, QSO=1, STAR=2 (alphabetical).
  * Folds:  StratifiedKFold(5, shuffle=True, random_state=42).split on the 577347
    competition rows ONLY, in original train-CSV row order.
  * Original SDSS17 rows go ONLY into each fold's TRAIN pool (never validation),
    down-weighted; OOF is computed only on competition train rows.
  * Group-aggregation / quantile-bin features use ONLY input columns (colors,
    redshift, magnitudes, sky) — never the target — so they are target-free and
    keep the OOF honest while still being computed transductively over
    train+test+original.

Outputs to /kaggle/working/:
  oof_massfe_lgb.npy  / test_massfe_lgb.npy   (577347,3)/(247435,3) float32
  oof_massfe_cat.npy  / test_massfe_cat.npy
  submission_massfe_<m>.csv                   argmax -> label (nice-to-have)
  results.txt                                 per-fold + OOF balanced accuracy
"""

from __future__ import annotations

import os

# Keep CPU helper libs from grabbing every thread (Kaggle CPU kernels = ~4 vCPU).
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import gc
import glob
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Constants / contract
# ----------------------------------------------------------------------------
SEED = 42
N_FOLDS = 5
N_CLASSES = 3
TARGET = "class"
ID_COL = "id"

CLASSES = ["GALAXY", "QSO", "STAR"]  # alphabetical -> 0,1,2
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
INV_MAP = {i: c for c, i in LABEL_MAP.items()}

RAW_NUMS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
BANDS = ["u", "g", "r", "i", "z"]  # ordered by increasing wavelength
RAW_CATS = ["spectral_type", "galaxy_population"]

N_TRAIN_EXPECT = 577347
N_TEST_EXPECT = 247435

# Codes for binned-color categoricals (these reconstruct the competition cats).
SPEC_MAP = {"M": 0, "G/K": 1, "A/F": 2, "O/B": 3}
POP_MAP = {"Blue_Cloud": 0, "Red_Sequence": 1}

# Hyper-params.
ORIGINAL_WEIGHT = 0.1          # appended original rows down-weighted in TRAIN pool
LGB_ROUNDS = 3500
CAT_ITERATIONS = 4000
LGB_EARLY = 200
CAT_EARLY = 250
CAT_CLASS_WEIGHTS = [1.0, 3.25, 5.0]  # up-weight rare STAR for balanced accuracy
CLIP = 1e-7
EPS = np.float32(1e-6)

WORK = Path("/kaggle/working")
try:
    WORK.mkdir(parents=True, exist_ok=True)
except OSError:
    # Non-Kaggle env (e.g. offline FE validation): /kaggle is read-only/absent.
    # main() is the only thing that writes under WORK and is not run off-Kaggle.
    pass
RESULTS_PATH = WORK / "results.txt"

np.random.seed(SEED)

T0 = time.perf_counter()
_RESULT_LINES: list[str] = []


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


def emit(msg: str) -> None:
    """Print AND record for results.txt."""
    print(msg, flush=True)
    _RESULT_LINES.append(msg)


def flush_results() -> None:
    RESULTS_PATH.write_text("\n".join(_RESULT_LINES) + "\n")


# ----------------------------------------------------------------------------
# Data discovery
# ----------------------------------------------------------------------------
def find_competition_root() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/playground-series-s6e6"),
        Path("/kaggle/input/playground-series-s6e6"),
    ]
    candidates += [Path(p).parent for p in glob.glob("/kaggle/input/**/train.csv", recursive=True)]
    seen: list[Path] = []
    for c in candidates:
        if c not in seen:
            seen.append(c)
    for root in seen:
        if (
            (root / "train.csv").exists()
            and (root / "test.csv").exists()
            and (root / "sample_submission.csv").exists()
        ):
            return root
    raise FileNotFoundError("Could not locate competition train/test/sample_submission CSVs.")


def find_original_path() -> Path:
    candidates = [
        Path(
            "/kaggle/input/datasets/fedesoriano/"
            "stellar-classification-dataset-sdss17/star_classification.csv"
        ),
    ]
    candidates += [Path(p) for p in glob.glob("/kaggle/input/**/star_classification.csv", recursive=True)]
    seen: list[Path] = []
    for c in candidates:
        if c not in seen:
            seen.append(c)
    for p in seen:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find star_classification.csv (SDSS17 original dataset).")


# ----------------------------------------------------------------------------
# Original-dataset categorical reconstruction (verified exact on comp data)
#   spectral_type     = cut(r - g, [-inf,-1,-0.5,0,inf])  -> M / G/K / A/F / O/B
#   galaxy_population = cut(u - r, [-inf, 2.2, inf])       -> Blue_Cloud/Red_Sequence
# ----------------------------------------------------------------------------
def rebuild_original_cats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spectral_type"] = (
        pd.cut(
            out["r"] - out["g"],
            [-np.inf, -1.0, -0.5, 0.0, np.inf],
            labels=["M", "G/K", "A/F", "O/B"],
        ).astype(str)
    )
    out["galaxy_population"] = (
        pd.cut(
            out["u"] - out["r"],
            [-np.inf, 2.2, np.inf],
            labels=["Blue_Cloud", "Red_Sequence"],
        ).astype(str)
    )
    return out


# ----------------------------------------------------------------------------
# Feature engineering helpers
# ----------------------------------------------------------------------------
def _qbin(v: np.ndarray, q: int) -> np.ndarray:
    """Quantile-bin a 1-D array into <=q integer codes (target-free).

    Edges from q-quantiles of the finite values; codes via searchsorted. Non-finite
    inputs get code -1. Duplicate edges are collapsed (np.unique) so heavily-tied
    distributions just yield fewer effective bins.
    """
    v = np.asarray(v, dtype="float64")
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return np.full(v.shape, -1, dtype="int32")
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, q + 1)))
    # Interior edges only -> codes in [0, len(edges)-2]; clip keeps it bounded.
    codes = np.searchsorted(edges[1:-1], v, side="right").astype("int32")
    codes[~np.isfinite(v)] = -1
    return codes


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full mass-FE matrix on a frame holding the raw columns.

    Applied identically to (competition train + test + original) concatenated, so
    every quantile bin / group aggregation shares the same parameters and the three
    splits have an identical column schema.
    """
    out = pd.DataFrame(index=df.index)
    for c in RAW_NUMS:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    bands_arr = out[BANDS].to_numpy(dtype="float32")  # (n,5) u,g,r,i,z

    # --- (1) all pairwise band diffs + their abs ---
    pairs = [(BANDS[a], BANDS[b]) for a in range(len(BANDS)) for b in range(a + 1, len(BANDS))]
    for a, b in pairs:
        d = (out[a] - out[b]).astype("float32")
        out[f"{a}_{b}"] = d
        out[f"abs_{a}_{b}"] = d.abs().astype("float32")

    # --- (2) second-order colors (color of colors) ---
    out["ug_gr"] = (out["u_g"] - out["g_r"]).astype("float32")
    out["gr_ri"] = (out["g_r"] - out["r_i"]).astype("float32")
    out["ri_iz"] = (out["r_i"] - out["i_z"]).astype("float32")
    out["ug_ri"] = (out["u_g"] - out["r_i"]).astype("float32")
    out["gr_iz"] = (out["g_r"] - out["i_z"]).astype("float32")

    # --- (3) magnitude band statistics ---
    out["mag_mean"] = np.nanmean(bands_arr, axis=1).astype("float32")
    out["mag_std"] = np.nanstd(bands_arr, axis=1).astype("float32")
    out["mag_min"] = np.nanmin(bands_arr, axis=1).astype("float32")
    out["mag_max"] = np.nanmax(bands_arr, axis=1).astype("float32")
    out["mag_range"] = (out["mag_max"] - out["mag_min"]).astype("float32")
    out["mag_median"] = np.nanmedian(bands_arr, axis=1).astype("float32")
    # argmin/argmax = which band is faintest/brightest (NaN-safe).
    b_for_min = np.where(np.isnan(bands_arr), np.inf, bands_arr)
    b_for_max = np.where(np.isnan(bands_arr), -np.inf, bands_arr)
    out["mag_argmin"] = b_for_min.argmin(axis=1).astype("float32")
    out["mag_argmax"] = b_for_max.argmax(axis=1).astype("float32")
    # slope = linear fit across bands (x=[-2..2], sum(x^2)=10); curvature = mean 2nd diff.
    x_idx = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype="float32")
    out["mag_slope"] = ((bands_arr * x_idx).sum(axis=1) / 10.0).astype("float32")
    second_diff = bands_arr[:, :-2] - 2.0 * bands_arr[:, 1:-1] + bands_arr[:, 2:]
    out["mag_curv"] = np.nanmean(second_diff, axis=1).astype("float32")

    # --- (4) flux features: flux_b = 10**(-0.4*b) + flux stats + flux colors ---
    clipped = np.clip(bands_arr, -30.0, 30.0)
    flux = np.power(np.float32(10.0), np.float32(-0.4) * clipped).astype("float32")
    for j, b in enumerate(BANDS):
        out[f"flux_{b}"] = flux[:, j]
    out["flux_mean"] = flux.mean(axis=1).astype("float32")
    out["flux_std"] = flux.std(axis=1).astype("float32")
    out["flux_min"] = flux.min(axis=1).astype("float32")
    out["flux_max"] = flux.max(axis=1).astype("float32")
    out["flux_range"] = (out["flux_max"] - out["flux_min"]).astype("float32")
    out["flux_sum"] = flux.sum(axis=1).astype("float32")
    for j in range(len(BANDS) - 1):
        a, b = BANDS[j], BANDS[j + 1]
        out[f"fluxratio_{a}_{b}"] = (flux[:, j] / (flux[:, j + 1] + EPS)).astype("float32")

    # --- (5) redshift transforms ---
    rz = out["redshift"].astype("float32")
    rz_abs = rz.abs().astype("float32")
    out["redshift_abs"] = rz_abs
    out["redshift_log1p_abs"] = np.log1p(rz_abs).astype("float32")
    out["redshift_sqrt_abs"] = np.sqrt(rz_abs).astype("float32")
    out["redshift_cbrt"] = np.cbrt(rz.to_numpy(dtype="float32")).astype("float32")
    out["redshift_sq"] = (rz * rz).astype("float32")
    out["redshift_is_neg"] = (rz < 0).astype("float32")
    out["redshift_lt_002"] = (rz < 0.02).astype("float32")
    out["redshift_gt_07"] = (rz > 0.7).astype("float32")
    out["redshift_inv"] = (np.float32(1.0) / (rz_abs + EPS)).astype("float32")

    # --- (6) redshift x band, band / redshift, redshift x color ---
    for b in BANDS:
        out[f"rz_x_{b}"] = (rz * out[b]).astype("float32")
        out[f"{b}_over_rz"] = (out[b] / (rz_abs + EPS)).astype("float32")
    for c in ("u_g", "g_r", "r_i", "i_z"):
        out[f"rz_x_{c}"] = (rz * out[c]).astype("float32")

    # --- (7) color ratios (safe division) ---
    out["ug_gr_ratio"] = (out["u_g"] / (out["g_r"].abs() + EPS)).astype("float32")
    out["gr_ri_ratio"] = (out["g_r"] / (out["r_i"].abs() + EPS)).astype("float32")
    out["ri_iz_ratio"] = (out["r_i"] / (out["i_z"].abs() + EPS)).astype("float32")
    out["ur_gi_ratio"] = (out["u_r"] / (out["g_i"].abs() + EPS)).astype("float32")
    out["z_over_g"] = (out["z"] / (out["g"].abs() + EPS)).astype("float32")

    # --- (8) sky position: sin/cos + unit-sphere cartesian ---
    alpha_rad = np.deg2rad(out["alpha"].to_numpy(dtype="float32"))
    delta_rad = np.deg2rad(out["delta"].to_numpy(dtype="float32"))
    out["alpha_sin"] = np.sin(alpha_rad).astype("float32")
    out["alpha_cos"] = np.cos(alpha_rad).astype("float32")
    out["delta_sin"] = np.sin(delta_rad).astype("float32")
    out["delta_cos"] = np.cos(delta_rad).astype("float32")
    out["sky_x"] = (np.cos(delta_rad) * np.cos(alpha_rad)).astype("float32")
    out["sky_y"] = (np.cos(delta_rad) * np.sin(alpha_rad)).astype("float32")
    out["sky_z"] = np.sin(delta_rad).astype("float32")

    # --- (9) quantile-bin codes q16/q32/q64/q128 on key features ---
    qbin_feats = BANDS + ["redshift", "u_g", "g_r", "r_i", "i_z", "u_r", "g_i"]
    for f in qbin_feats:
        v = out[f].to_numpy(dtype="float32")
        for q in (16, 32, 64, 128):
            out[f"{f}_q{q}"] = _qbin(v, q).astype("float32")

    # --- (10) round / floor / mod categorical codes ---
    rf_feats = BANDS + ["redshift", "u_g", "g_r", "r_i", "i_z"]
    for f in rf_feats:
        v = out[f]
        out[f"{f}_round"] = np.round(v).astype("float32")
        out[f"{f}_floor"] = np.floor(v).astype("float32")
    out["redshift_mod10"] = (np.round(out["redshift"] * 1000.0) % 10).astype("float32")
    for b in ("u", "r", "z"):
        out[f"{b}_mod10"] = (np.round(out[b] * 10.0) % 10).astype("float32")

    # --- (11) binned-color categorical codes (reconstruct the competition cats) ---
    spec_code = df["spectral_type"].astype(str).map(SPEC_MAP).fillna(-1).astype("int32")
    pop_code = df["galaxy_population"].astype(str).map(POP_MAP).fillna(-1).astype("int32")
    out["spectral_type_code"] = spec_code.astype("float32")
    out["galaxy_population_code"] = pop_code.astype("float32")

    # --- (12) groupby AGGREGATIONS over coarse bins (the unused grandmaster lever) ---
    # group mean/std/count + deviation-from-group-mean of the key colors/redshift/mag,
    # over a few coarse, target-free bins. Computed transductively over the whole
    # concatenated frame (no target used -> OOF stays honest).
    redshift_bin = pd.Series(_qbin(out["redshift"].to_numpy(dtype="float32"), 16), index=out.index)
    alpha_cell = _qbin(out["alpha"].to_numpy(dtype="float32"), 12)
    delta_cell = _qbin(out["delta"].to_numpy(dtype="float32"), 12)
    sky_cell = pd.Series((alpha_cell.astype("int64") * 12 + delta_cell.astype("int64")), index=out.index)

    group_defs = {
        "g_spec": spec_code.astype("int64"),
        "g_pop": pop_code.astype("int64"),
        "g_rzbin": redshift_bin.astype("int64"),
        "g_sky": sky_cell,
    }
    agg_keys = ["u_g", "g_r", "r_i", "i_z", "redshift", "mag_mean"]
    for gname, grp in group_defs.items():
        grp = pd.Series(np.asarray(grp), index=out.index)
        counts = grp.map(grp.value_counts())
        out[f"{gname}_count"] = counts.astype("float32")
        for key in agg_keys:
            gm = out[key].groupby(grp).transform("mean")
            gs = out[key].groupby(grp).transform("std")
            out[f"{gname}_{key}_mean"] = gm.astype("float32")
            out[f"{gname}_{key}_std"] = gs.fillna(0.0).astype("float32")
            out[f"{gname}_{key}_dev"] = (out[key] - gm).astype("float32")

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# ----------------------------------------------------------------------------
# Metrics / artifact helpers
# ----------------------------------------------------------------------------
def normalize_proba(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, CLIP, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p.astype("float32")


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    recalls = {}
    for i, name in enumerate(CLASSES):
        mask = y_true == i
        recalls[name] = float((y_pred[mask] == i).mean()) if mask.sum() else 0.0
    return recalls


def report_oof(model_name: str, y: np.ndarray, oof: np.ndarray, fold_scores: list[float]) -> float:
    y_pred = oof.argmax(axis=1)
    ba = float(balanced_accuracy_score(y, y_pred))
    rec = per_class_recall(y, y_pred)
    cm = confusion_matrix(y, y_pred, labels=list(range(N_CLASSES)))
    emit("")
    emit(f"===== {model_name} =====")
    emit(f"  per-fold BA: {[round(s, 6) for s in fold_scores]}")
    emit(f"  mean fold BA: {np.mean(fold_scores):.6f}")
    emit(f"  OVERALL OOF balanced accuracy: {ba:.6f}")
    emit(
        "  per-class recall: "
        f"GALAXY={rec['GALAXY']:.4f} QSO={rec['QSO']:.4f} STAR={rec['STAR']:.4f}"
    )
    emit("  confusion (rows=true GALAXY/QSO/STAR, cols=pred):")
    for i, name in enumerate(CLASSES):
        emit(f"    {name:7s} {cm[i].tolist()}")
    flush_results()
    return ba


def save_model_arrays(name: str, oof: np.ndarray, test_pred: np.ndarray, sample: pd.DataFrame) -> None:
    oof = normalize_proba(oof)
    test_pred = normalize_proba(test_pred)
    np.save(WORK / f"oof_{name}.npy", oof.astype("float32"))
    np.save(WORK / f"test_{name}.npy", test_pred.astype("float32"))
    sub = sample.copy()
    sub[TARGET] = [INV_MAP[i] for i in test_pred.argmax(axis=1)]
    sub.to_csv(WORK / f"submission_{name}.csv", index=False)
    log(f"saved oof_{name}.npy {oof.shape}, test_{name}.npy {test_pred.shape}")


# ----------------------------------------------------------------------------
# Per-fold fit-data builder = competition-train rows + ALL original rows.
# ----------------------------------------------------------------------------
def make_fit_data(X_comp, y_comp, X_orig, y_orig, tr_idx):
    X_fit = np.vstack([X_comp[tr_idx], X_orig])
    y_fit = np.concatenate([y_comp[tr_idx], y_orig]).astype("int64")
    sw = compute_sample_weight(class_weight="balanced", y=y_fit).astype("float32")
    sw[len(tr_idx):] *= np.float32(ORIGINAL_WEIGHT)
    return X_fit, y_fit, sw


# ----------------------------------------------------------------------------
# LightGBM (balanced-accuracy early stopping)
# ----------------------------------------------------------------------------
def train_lgb(X_comp, y_comp, X_orig, y_orig, X_test, folds):
    import lightgbm as lgb

    def feval_bal_acc(y_pred_raw, dataset):
        y_true = dataset.get_label().astype(int)
        n = y_true.shape[0]
        arr = np.asarray(y_pred_raw)
        preds = arr if arr.ndim == 2 else arr.reshape(N_CLASSES, n).T
        return "bal_acc", balanced_accuracy_score(y_true, preds.argmax(axis=1)), True

    params = dict(
        objective="multiclass",
        num_class=N_CLASSES,
        metric="None",            # custom bal_acc is the ONLY metric -> drives early stop
        learning_rate=0.025,
        num_leaves=80,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.82,
        subsample_freq=1,
        colsample_bytree=0.6,     # lower colsample for the much wider feature matrix
        reg_alpha=0.05,
        reg_lambda=10.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
        first_metric_only=True,
    )

    oof = np.zeros((len(y_comp), N_CLASSES), dtype="float32")
    test_pred = np.zeros((len(X_test), N_CLASSES), dtype="float32")
    fold_scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        t0 = time.perf_counter()
        X_fit, y_fit, sw = make_fit_data(X_comp, y_comp, X_orig, y_orig, tr_idx)
        X_va, y_va = X_comp[va_idx], y_comp[va_idx]

        dtrain = lgb.Dataset(X_fit, label=y_fit, weight=sw)
        dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=LGB_ROUNDS,
            valid_sets=[dvalid],
            valid_names=["valid"],
            feval=feval_bal_acc,
            callbacks=[
                lgb.early_stopping(LGB_EARLY, first_metric_only=True),
                lgb.log_evaluation(250),
            ],
        )
        va_p = booster.predict(X_va, num_iteration=booster.best_iteration).astype("float32")
        te_p = booster.predict(X_test, num_iteration=booster.best_iteration).astype("float32")
        oof[va_idx] = va_p
        test_pred += te_p / N_FOLDS

        s = balanced_accuracy_score(y_va, va_p.argmax(axis=1))
        fold_scores.append(float(s))
        log(f"LGB fold {fold}: BA={s:.6f} best_iter={booster.best_iteration} "
            f"({time.perf_counter() - t0:.0f}s)")
        del dtrain, dvalid, booster, X_fit, y_fit, sw
        gc.collect()

    return oof, test_pred, fold_scores


# ----------------------------------------------------------------------------
# CatBoost (GPU-guarded; CPU on this kernel)
# ----------------------------------------------------------------------------
def train_cat(X_comp, y_comp, X_orig, y_orig, X_test, folds, feature_names):
    from catboost import CatBoostClassifier, Pool

    base_params = dict(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",   # macro-F1 proxy for balanced accuracy
        iterations=CAT_ITERATIONS,
        depth=8,
        learning_rate=0.045,
        l2_leaf_reg=8.0,
        random_strength=1.2,
        bootstrap_type="Bayesian",
        bagging_temperature=0.2,
        border_count=128,                       # CPU speed (vs 254) at negligible quality cost
        class_weights=CAT_CLASS_WEIGHTS,
        random_seed=SEED,
        early_stopping_rounds=CAT_EARLY,
        allow_writing_files=False,
        verbose=250,
    )

    # GPU works on P100 (CatBoost has its own CUDA); task_type='GPU' only errors at
    # .fit() so probe up front. This kernel is CPU -> n_gpu=0 -> CPU branch.
    try:
        from catboost.utils import get_gpu_device_count
        n_gpu = int(get_gpu_device_count())
    except Exception as e:  # pragma: no cover
        log(f"CAT GPU probe failed ({e!r}); using CPU")
        n_gpu = 0
    log(f"CatBoost GPU devices detected: {n_gpu}")

    def make_model():
        p = dict(base_params)
        if n_gpu > 0:
            p["task_type"] = "GPU"
            p["devices"] = "0"
        else:
            p["task_type"] = "CPU"
            p["thread_count"] = 4
        return CatBoostClassifier(**p)

    oof = np.zeros((len(y_comp), N_CLASSES), dtype="float32")
    test_pred = np.zeros((len(X_test), N_CLASSES), dtype="float32")
    fold_scores: list[float] = []
    test_pool = Pool(X_test, feature_names=feature_names)

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        t0 = time.perf_counter()
        X_fit = np.vstack([X_comp[tr_idx], X_orig])
        y_fit = np.concatenate([y_comp[tr_idx], y_orig]).astype("int64")
        w = np.ones(len(y_fit), dtype="float32")
        w[len(tr_idx):] = np.float32(ORIGINAL_WEIGHT)
        X_va, y_va = X_comp[va_idx], y_comp[va_idx]

        train_pool = Pool(X_fit, y_fit, feature_names=feature_names, weight=w)
        valid_pool = Pool(X_va, y_va, feature_names=feature_names)

        model = make_model()
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        va_p = model.predict_proba(valid_pool).astype("float32")
        te_p = model.predict_proba(test_pool).astype("float32")
        oof[va_idx] = va_p
        test_pred += te_p / N_FOLDS

        s = balanced_accuracy_score(y_va, va_p.argmax(axis=1))
        fold_scores.append(float(s))
        log(f"CAT fold {fold}: BA={s:.6f} best_iter={model.get_best_iteration()} "
            f"({time.perf_counter() - t0:.0f}s)")
        del train_pool, valid_pool, model, X_fit, y_fit, w
        gc.collect()

    del test_pool
    gc.collect()
    return oof, test_pred, fold_scores


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    emit("# s6e6 massfe : MASS feature engineering -> LightGBM + CatBoost")
    emit(f"# seed={SEED} folds={N_FOLDS} original_weight={ORIGINAL_WEIGHT}")

    root = find_competition_root()
    orig_path = find_original_path()
    log(f"competition root: {root}")
    log(f"original dataset: {orig_path}")

    train = pd.read_csv(root / "train.csv")
    test = pd.read_csv(root / "test.csv")
    sample = pd.read_csv(root / "sample_submission.csv")
    orig_raw = pd.read_csv(orig_path)
    log(f"train={train.shape} test={test.shape} sample={sample.shape} orig={orig_raw.shape}")

    assert len(train) == N_TRAIN_EXPECT, f"train rows {len(train)} != {N_TRAIN_EXPECT}"
    assert len(test) == N_TEST_EXPECT, f"test rows {len(test)} != {N_TEST_EXPECT}"

    # Align test to sample_submission row order (the test-output contract).
    if ID_COL in test.columns and ID_COL in sample.columns:
        test = test.set_index(ID_COL).loc[sample[ID_COL]].reset_index()
        log("test reordered to sample_submission id order")

    # Labels in train-CSV order (GALAXY=0, QSO=1, STAR=2).
    y_comp = train[TARGET].astype(str).map(LABEL_MAP).to_numpy()
    assert not np.any(np.isnan(y_comp.astype(float))), "unmapped class label in train"
    y_comp = y_comp.astype("int64")

    # Original dataset: reconstruct cats, filter to valid classes, drop sentinel mags.
    orig = rebuild_original_cats(orig_raw)
    orig[TARGET] = orig[TARGET].astype(str).str.upper()
    orig = orig[orig[TARGET].isin(CLASSES)].reset_index(drop=True)
    n_before = len(orig)
    band_vals = orig[BANDS].apply(pd.to_numeric, errors="coerce")
    sane_mask = ((band_vals > -100.0) & (band_vals < 100.0)).all(axis=1)
    orig = orig[sane_mask].reset_index(drop=True)
    log(f"original sentinel/out-of-range band rows dropped: {n_before - len(orig)}")
    y_orig = orig[TARGET].map(LABEL_MAP).to_numpy().astype("int64")
    log(f"original after filter: {orig.shape}; class counts="
        f"{pd.Series(y_orig).map(INV_MAP).value_counts().to_dict()}")

    # Build features ONCE on the concatenated frame so quantile bins / group aggs
    # share parameters and the three splits share an identical column schema.
    n1, n2, n3 = len(train), len(test), len(orig)
    cols = RAW_NUMS + RAW_CATS
    combined = pd.concat(
        [train[cols], test[cols], orig[cols]], axis=0, ignore_index=True
    )
    log(f"building mass features on combined frame {combined.shape} ...")
    X_all = add_features(combined)
    feature_names = list(X_all.columns)
    log(f"n_features = {len(feature_names)}")
    emit(f"# n_features={len(feature_names)} n_orig_rows={n3}")

    X_comp = X_all.iloc[:n1].fillna(0.0).to_numpy(dtype="float32")
    X_test = X_all.iloc[n1:n1 + n2].fillna(0.0).to_numpy(dtype="float32")
    X_orig = X_all.iloc[n1 + n2:].fillna(0.0).to_numpy(dtype="float32")
    assert X_comp.shape[0] == n1 and X_test.shape[0] == n2 and X_orig.shape[0] == n3
    log(f"feature matrix: comp={X_comp.shape} test={X_test.shape} orig={X_orig.shape}")
    del X_all, combined, orig, orig_raw, train, test
    gc.collect()

    # Folds depend only on y_comp + n_splits + random_state -> align with all artifacts.
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X_comp, y_comp))

    summary: dict[str, float] = {}

    # ---- LightGBM (saved first so its artifacts survive even if CatBoost overruns) ----
    log("training LightGBM ...")
    oof, test_pred, fs = train_lgb(X_comp, y_comp, X_orig, y_orig, X_test, folds)
    summary["massfe_lgb"] = report_oof("massfe_lgb (LightGBM)", y_comp, oof, fs)
    save_model_arrays("massfe_lgb", oof, test_pred, sample)
    del oof, test_pred
    gc.collect()

    # ---- CatBoost ----
    log("training CatBoost ...")
    oof, test_pred, fs = train_cat(X_comp, y_comp, X_orig, y_orig, X_test, folds, feature_names)
    summary["massfe_cat"] = report_oof("massfe_cat (CatBoost)", y_comp, oof, fs)
    save_model_arrays("massfe_cat", oof, test_pred, sample)
    del oof, test_pred
    gc.collect()

    emit("")
    emit("==================== SUMMARY (OOF balanced accuracy) ====================")
    for m in ("massfe_lgb", "massfe_cat"):
        if m in summary:
            emit(f"  {m} OOF BA = {summary[m]:.6f}")
    emit("DONE")
    flush_results()
    log("ALL DONE")


if __name__ == "__main__":
    main()
