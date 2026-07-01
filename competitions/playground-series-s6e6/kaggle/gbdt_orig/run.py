"""s6e6 gbdt_orig: LightGBM + XGBoost + CatBoost with the ORIGINAL SDSS17 dataset
appended to each fold's TRAIN pool, flux features, and balanced-accuracy-aware
early stopping.

Self-contained Kaggle SCRIPT kernel. Does NOT import the user's pipeline modules.

Contract (must match existing artifacts so the new OOF stacks):
  * Labels: GALAXY=0, QSO=1, STAR=2 (alphabetical / LabelEncoder over CLASSES).
  * Folds: StratifiedKFold(5, shuffle=True, random_state=42).split(X_comp, y_comp)
    on the 577347 competition rows ONLY, in original train-CSV row order.
  * Original SDSS17 rows go ONLY into each fold's TRAIN pool (never validation),
    at a small sample weight; OOF is computed only on competition train rows.

Outputs to /kaggle/working/:
  oof_<m>_orig.npy   (577347, 3) float32, columns [GALAXY,QSO,STAR], train-CSV row order
  test_<m>_orig.npy  (247435, 3) float32, columns [GALAXY,QSO,STAR], sample_submission order
  results.txt        per-fold + overall OOF balanced accuracy + per-class recall per model
  submission_<m>.csv argmax->label (nice-to-have)
"""

from __future__ import annotations

import os

# Keep CPU helper libs from grabbing every thread while one model trains.
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
BANDS = ["u", "g", "r", "i", "z"]
RAW_CATS = ["spectral_type", "galaxy_population"]

N_TRAIN_EXPECT = 577347
N_TEST_EXPECT = 247435

# Hyper-params (lgbm-v3 / cat-v3 spirit).
ORIGINAL_WEIGHT = 0.1          # appended original rows down-weighted in TRAIN pool
LGB_ROUNDS = 4000
XGB_ROUNDS = 4000
CAT_ITERATIONS = 5000
LGB_EARLY = 200
XGB_EARLY = 200
CAT_EARLY = 260
CAT_CLASS_WEIGHTS = [1.0, 3.25, 5.0]  # up-weight rare STAR for balanced accuracy
CLIP = 1e-15

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = WORK / "results.txt"

# Codes for binned-color categoricals (these reconstruct the competition cats).
SPEC_MAP = {"M": 0, "G/K": 1, "A/F": 2, "O/B": 3}
POP_MAP = {"Blue_Cloud": 0, "Red_Sequence": 1}

np.random.seed(SEED)

T0 = time.perf_counter()
_RESULT_LINES: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.perf_counter() - T0:8.1f}s] {msg}"
    print(line, flush=True)


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
        if (root / "train.csv").exists() and (root / "test.csv").exists() and (
            root / "sample_submission.csv"
        ).exists():
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
# ----------------------------------------------------------------------------
def rebuild_original_cats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spectral_type"] = (
        pd.cut(
            out["r"] - out["g"],
            [-np.inf, -1.0, -0.5, 0.0, np.inf],
            labels=["M", "G/K", "A/F", "O/B"],
        )
        .astype(str)
    )
    out["galaxy_population"] = (
        pd.cut(
            out["u"] - out["r"],
            [-np.inf, 2.2, np.inf],
            labels=["Blue_Cloud", "Red_Sequence"],
        )
        .astype(str)
    )
    return out


# ----------------------------------------------------------------------------
# Feature engineering — numeric-only matrix shared by lgb / xgb / cat.
# Applied identically to competition train, test, and original rows.
# ----------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in RAW_NUMS:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    # All 10 pairwise color differences.
    color_pairs = [
        ("u", "g"), ("u", "r"), ("u", "i"), ("u", "z"),
        ("g", "r"), ("g", "i"), ("g", "z"),
        ("r", "i"), ("r", "z"),
        ("i", "z"),
    ]
    for a, b in color_pairs:
        out[f"{a}_{b}"] = (out[a] - out[b]).astype("float32")

    # Second-order colors.
    out["ug_gr"] = (out["u_g"] - out["g_r"]).astype("float32")
    out["gr_ri"] = (out["g_r"] - out["r_i"]).astype("float32")
    out["ri_iz"] = (out["r_i"] - out["i_z"]).astype("float32")

    # Magnitude band statistics.
    band = out[BANDS].to_numpy(dtype="float32")
    out["mag_mean"] = np.nanmean(band, axis=1).astype("float32")
    out["mag_std"] = np.nanstd(band, axis=1).astype("float32")
    out["mag_min"] = np.nanmin(band, axis=1).astype("float32")
    out["mag_max"] = np.nanmax(band, axis=1).astype("float32")
    out["mag_range"] = (out["mag_max"] - out["mag_min"]).astype("float32")

    # Redshift transforms.
    rz = out["redshift"].astype("float32")
    rz_abs = np.abs(rz).astype("float32")
    out["redshift_abs"] = rz_abs
    out["redshift_log1p_abs"] = np.log1p(rz_abs).astype("float32")
    out["redshift_sq"] = (rz * rz).astype("float32")
    out["redshift_is_neg"] = (rz < 0).astype("float32")
    out["redshift_lt_002"] = (rz < 0.02).astype("float32")
    out["redshift_gt_07"] = (rz > 0.7).astype("float32")

    # Redshift x each band.
    for b in BANDS:
        out[f"redshift_x_{b}"] = (rz * out[b]).astype("float32")

    # A few color ratios (safe division).
    eps = np.float32(1e-6)
    out["ug_gr_ratio"] = (out["u_g"] / (out["g_r"].abs() + eps)).astype("float32")
    out["gr_ri_ratio"] = (out["g_r"] / (out["r_i"].abs() + eps)).astype("float32")
    out["ri_iz_ratio"] = (out["r_i"] / (out["i_z"].abs() + eps)).astype("float32")
    out["z_over_g"] = (out["z"] / (out["g"].abs() + eps)).astype("float32")

    # Sky sin/cos of alpha & delta.
    alpha_rad = np.deg2rad(out["alpha"].to_numpy(dtype="float32"))
    delta_rad = np.deg2rad(out["delta"].to_numpy(dtype="float32"))
    out["alpha_sin"] = np.sin(alpha_rad).astype("float32")
    out["alpha_cos"] = np.cos(alpha_rad).astype("float32")
    out["delta_sin"] = np.sin(delta_rad).astype("float32")
    out["delta_cos"] = np.cos(delta_rad).astype("float32")

    # Flux features: flux_b = 10**(-0.4*b), with band magnitudes clipped for safety.
    fluxes = []
    for b in BANDS:
        clipped = np.clip(out[b].to_numpy(dtype="float32"), -30.0, 30.0)
        flux = np.power(np.float32(10.0), np.float32(-0.4) * clipped).astype("float32")
        out[f"flux_{b}"] = flux
        fluxes.append(flux)
    flux_mat = np.vstack(fluxes).T
    out["flux_mean"] = np.nanmean(flux_mat, axis=1).astype("float32")
    out["flux_std"] = np.nanstd(flux_mat, axis=1).astype("float32")

    # Encode the binned-color categoricals as small int codes.
    out["spectral_type_code"] = (
        df["spectral_type"].astype(str).map(SPEC_MAP).fillna(-1).astype("int32")
    )
    out["galaxy_population_code"] = (
        df["galaxy_population"].astype(str).map(POP_MAP).fillna(-1).astype("int32")
    )

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# ----------------------------------------------------------------------------
# Metrics helpers
# ----------------------------------------------------------------------------
def normalize_proba(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, CLIP, 1.0 - CLIP)
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
    emit(f"  confusion (rows=true GALAXY/QSO/STAR, cols=pred):")
    for i, name in enumerate(CLASSES):
        emit(f"    {name:7s} {cm[i].tolist()}")
    flush_results()
    return ba


def save_model_arrays(model_name: str, oof: np.ndarray, test_pred: np.ndarray,
                      sample: pd.DataFrame) -> None:
    oof = normalize_proba(oof)
    test_pred = normalize_proba(test_pred)
    np.save(WORK / f"oof_{model_name}_orig.npy", oof.astype("float32"))
    np.save(WORK / f"test_{model_name}_orig.npy", test_pred.astype("float32"))
    sub = sample.copy()
    sub[TARGET] = [INV_MAP[i] for i in test_pred.argmax(axis=1)]
    sub.to_csv(WORK / f"submission_{model_name}.csv", index=False)
    log(f"saved oof_{model_name}_orig.npy {oof.shape}, test_{model_name}_orig.npy {test_pred.shape}")


# ----------------------------------------------------------------------------
# Per-model training. Each returns (oof[n_comp,3], test_pred[n_test,3], fold_scores).
# X_comp / X_orig are numpy float32 matrices already in identical column order.
# folds: list of (tr_idx, va_idx) over competition rows.
# ----------------------------------------------------------------------------
def make_fit_data(X_comp, y_comp, X_orig, y_orig, tr_idx):
    """Build the fold's fit matrix = competition-train rows + ALL original rows."""
    X_fit = np.vstack([X_comp[tr_idx], X_orig])
    y_fit = np.concatenate([y_comp[tr_idx], y_orig]).astype("int64")
    # balanced per-row weights on the FIT set; original rows then * ORIGINAL_WEIGHT.
    sw = compute_sample_weight(class_weight="balanced", y=y_fit).astype("float32")
    sw[len(tr_idx):] *= np.float32(ORIGINAL_WEIGHT)
    return X_fit, y_fit, sw


def train_lgb(X_comp, y_comp, X_orig, y_orig, X_test, folds):
    import lightgbm as lgb

    def feval_bal_acc(y_pred_raw, dataset):
        # lgb multiclass raw preds: modern lgb (>=4) hands back a 2-D (n_rows,
        # n_classes) array; older lgb flattens column-major (n_classes * n_rows).
        # Handle both so the metric is correct regardless of installed version.
        y_true = dataset.get_label().astype(int)
        n = y_true.shape[0]
        arr = np.asarray(y_pred_raw)
        if arr.ndim == 2:
            preds = arr
        else:
            # 1-D flattened, column-major: [class0 over all rows, class1 ..., class2 ...]
            preds = arr.reshape(N_CLASSES, n).T
        y_hat = preds.argmax(axis=1)
        return "bal_acc", balanced_accuracy_score(y_true, y_hat), True

    # metric="None" disables LightGBM's built-in default (multi_logloss) so the
    # custom bal_acc feval is the ONLY / FIRST metric and therefore drives early
    # stopping. Without this, lgb silently adds multi_logloss as the first metric
    # and first_metric_only selects best_iteration by logloss, NOT balanced accuracy.
    params = dict(
        objective="multiclass",
        num_class=N_CLASSES,
        metric="None",
        learning_rate=0.025,
        num_leaves=80,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.82,
        subsample_freq=1,
        colsample_bytree=0.72,
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


def train_xgb(X_comp, y_comp, X_orig, y_orig, X_test, folds):
    import xgboost as xgb

    def feval_bal_acc(y_pred, dtrain):
        # xgb multi:softprob preds arrive as (n_rows, n_classes).
        y_true = dtrain.get_label().astype(int)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, N_CLASSES)
        y_hat = y_pred.argmax(axis=1)
        return "bal_acc", float(balanced_accuracy_score(y_true, y_hat))

    # Only the custom balanced-accuracy metric drives early stopping (maximize=True);
    # the built-in eval metric is disabled so it does not compete.
    params = dict(
        objective="multi:softprob",
        num_class=N_CLASSES,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.05,
        reg_lambda=8.0,
        tree_method="hist",
        seed=SEED,
        nthread=-1,
        verbosity=0,
        disable_default_eval_metric=True,
    )

    oof = np.zeros((len(y_comp), N_CLASSES), dtype="float32")
    test_pred = np.zeros((len(X_test), N_CLASSES), dtype="float32")
    fold_scores: list[float] = []
    dtest = xgb.DMatrix(X_test)

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        t0 = time.perf_counter()
        X_fit, y_fit, sw = make_fit_data(X_comp, y_comp, X_orig, y_orig, tr_idx)
        X_va, y_va = X_comp[va_idx], y_comp[va_idx]

        dtrain = xgb.DMatrix(X_fit, label=y_fit, weight=sw)
        dvalid = xgb.DMatrix(X_va, label=y_va)
        # Custom balanced-accuracy metric; maximize=True so early stopping tracks BA.
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=XGB_ROUNDS,
            evals=[(dvalid, "valid")],
            custom_metric=feval_bal_acc,
            maximize=True,
            early_stopping_rounds=XGB_EARLY,
            verbose_eval=250,
        )
        best_it = getattr(booster, "best_iteration", None)
        if best_it is None:
            best_it = XGB_ROUNDS - 1
        rng = (0, best_it + 1)
        va_p = booster.predict(dvalid, iteration_range=rng).astype("float32")
        te_p = booster.predict(dtest, iteration_range=rng).astype("float32")
        if va_p.ndim == 1:
            va_p = va_p.reshape(-1, N_CLASSES)
            te_p = te_p.reshape(-1, N_CLASSES)
        oof[va_idx] = va_p
        test_pred += te_p / N_FOLDS

        s = balanced_accuracy_score(y_va, va_p.argmax(axis=1))
        fold_scores.append(float(s))
        log(f"XGB fold {fold}: BA={s:.6f} best_iter={best_it} "
            f"({time.perf_counter() - t0:.0f}s)")
        del dtrain, dvalid, booster, X_fit, y_fit, sw
        gc.collect()

    del dtest
    gc.collect()
    return oof, test_pred, fold_scores


def train_cat(X_comp, y_comp, X_orig, y_orig, X_test, folds, feature_names, cat_idx):
    from catboost import CatBoostClassifier, Pool

    # CatBoost on GPU; fall back to CPU if the GPU init fails for any reason.
    base_params = dict(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        iterations=CAT_ITERATIONS,
        depth=8,
        learning_rate=0.042,
        l2_leaf_reg=8.0,
        random_strength=1.2,
        bootstrap_type="Bayesian",
        bagging_temperature=0.2,
        border_count=254,
        class_weights=CAT_CLASS_WEIGHTS,
        random_seed=SEED,
        early_stopping_rounds=CAT_EARLY,
        allow_writing_files=False,
        verbose=250,
    )

    # Probe for a usable GPU via CatBoost's own utility (works on P100/T4 — CatBoost has its
    # own CUDA kernels, unaffected by the torch sm_60 drop). task_type='GPU' only errors at
    # .fit() time, so construct-time try/except can't catch it — detect up front instead.
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
            p["thread_count"] = 4
        return CatBoostClassifier(**p)

    oof = np.zeros((len(y_comp), N_CLASSES), dtype="float32")
    test_pred = np.zeros((len(X_test), N_CLASSES), dtype="float32")
    fold_scores: list[float] = []
    test_pool = Pool(X_test, feature_names=feature_names, cat_features=cat_idx or None)

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        t0 = time.perf_counter()
        # CatBoost uses class_weights for balance; original rows still down-weighted.
        X_fit = np.vstack([X_comp[tr_idx], X_orig])
        y_fit = np.concatenate([y_comp[tr_idx], y_orig]).astype("int64")
        w = np.ones(len(y_fit), dtype="float32")
        w[len(tr_idx):] = np.float32(ORIGINAL_WEIGHT)
        X_va, y_va = X_comp[va_idx], y_comp[va_idx]

        train_pool = Pool(X_fit, y_fit, feature_names=feature_names,
                          cat_features=cat_idx or None, weight=w)
        valid_pool = Pool(X_va, y_va, feature_names=feature_names, cat_features=cat_idx or None)

        model = make_model()
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        va_p = model.predict_proba(valid_pool).astype("float32")
        te_p = model.predict_proba(test_pool).astype("float32")
        oof[va_idx] = va_p
        test_pred += te_p / N_FOLDS

        s = balanced_accuracy_score(y_va, va_p.argmax(axis=1))
        fold_scores.append(float(s))
        bi = model.get_best_iteration()
        log(f"CAT fold {fold}: BA={s:.6f} best_iter={bi} ({time.perf_counter() - t0:.0f}s)")
        del train_pool, valid_pool, model, X_fit, y_fit, w
        gc.collect()

    del test_pool
    gc.collect()
    return oof, test_pred, fold_scores


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    emit("# s6e6 gbdt_orig : LGB + XGB + CAT with ORIGINAL SDSS17 appended to TRAIN")
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

    # IMPORTANT: align test to sample_submission row order.
    if ID_COL in test.columns and ID_COL in sample.columns:
        test = test.set_index(ID_COL).loc[sample[ID_COL]].reset_index()
        log("test reordered to sample_submission id order")

    # Labels in train-CSV order (GALAXY=0, QSO=1, STAR=2).
    y_comp = train[TARGET].astype(str).map(LABEL_MAP).to_numpy()
    assert not np.any(np.isnan(y_comp.astype(float))), "unmapped class label in train"
    y_comp = y_comp.astype("int64")

    # Original dataset: reconstruct cats, filter to valid classes, map labels.
    orig = rebuild_original_cats(orig_raw)
    orig[TARGET] = orig[TARGET].astype(str).str.upper()
    orig = orig[orig[TARGET].isin(CLASSES)].reset_index(drop=True)
    # Drop original rows with sentinel/out-of-range band magnitudes (SDSS17 has at
    # least one row with u=g=z=-9999.0). They survive feature engineering at weight
    # 0.1 but would leak garbage colors/flux; a sane magnitude window removes them.
    n_before = len(orig)
    band_vals = orig[BANDS].apply(pd.to_numeric, errors="coerce")
    sane_mask = ((band_vals > -100.0) & (band_vals < 100.0)).all(axis=1)
    orig = orig[sane_mask].reset_index(drop=True)
    n_dropped = n_before - len(orig)
    log(f"original sentinel/out-of-range band rows dropped: {n_dropped}")
    y_orig = orig[TARGET].map(LABEL_MAP).to_numpy().astype("int64")
    log(f"original after filter: {orig.shape}; class counts="
        f"{pd.Series(y_orig).map(INV_MAP).value_counts().to_dict()}")

    # Features (identical pipeline for all three frames).
    log("building features ...")
    Xc_df = add_features(train)
    Xt_df = add_features(test)
    Xo_df = add_features(orig)
    feature_names = list(Xc_df.columns)
    assert list(Xt_df.columns) == feature_names and list(Xo_df.columns) == feature_names

    # X is a homogeneous float32 numpy array, so CatBoost cannot mark columns categorical
    # (it requires int/str dtype). The two binned-color codes are small ordinals and work
    # fine as plain numeric features → no native cat_features (avoids the float/cat crash).
    cat_idx: list[int] = []

    # NaN handling: GBDTs handle NaN, but fill for clean matrices / CatBoost.
    X_comp = Xc_df.fillna(0.0).to_numpy(dtype="float32")
    X_test = Xt_df.fillna(0.0).to_numpy(dtype="float32")
    X_orig = Xo_df.fillna(0.0).to_numpy(dtype="float32")
    log(f"feature matrix: comp={X_comp.shape} test={X_test.shape} orig={X_orig.shape} "
        f"n_features={len(feature_names)}")
    emit(f"# n_features={len(feature_names)} n_orig_rows={len(X_orig)}")
    del Xc_df, Xt_df, Xo_df, orig, orig_raw, train, test
    gc.collect()

    # Folds: depend only on y_comp + n_splits + random_state => align with all artifacts.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X_comp, y_comp))

    summary: dict[str, float] = {}

    # ---- LGB + XGB already completed & salvaged from the prior run; cat-only rerun. ----

    # ---- CatBoost (GPU) ----
    log("training CatBoost (GPU) ...")
    oof, test_pred, fs = train_cat(
        X_comp, y_comp, X_orig, y_orig, X_test, folds, feature_names, cat_idx
    )
    summary["cat"] = report_oof("cat_orig (CatBoost)", y_comp, oof, fs)
    save_model_arrays("cat", oof, test_pred, sample)
    del oof, test_pred
    gc.collect()

    emit("")
    emit("==================== SUMMARY (OOF balanced accuracy) ====================")
    for m in ("cat",):
        emit(f"  {m}_orig OOF BA = {summary[m]:.6f}")
    emit(f"  (compare to existing oof_<m>_s42; lgb baseline ~0.964)")
    emit("DONE")
    flush_results()
    log("ALL DONE")


if __name__ == "__main__":
    main()
