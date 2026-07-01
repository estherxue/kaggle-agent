"""s6e6 optunabase: heavily Optuna-tuned strong GBDT base (CatBoost-GPU + LightGBM-GPU).

Goal: produce a marginally stronger single-model base than the hand-tuned catv3
(OOF balanced accuracy ~0.96877) by TPE-searching the CatBoost / LightGBM hyper-
parameters that matter most for this competition, then refitting the best config
with the project's full stack-aligned 5-fold + original-SDSS17 augmentation so the
resulting OOF is honest and combines with the existing artifact pool.

Self-contained Kaggle SCRIPT kernel — does NOT import the user's pipeline modules.

What it does
------------
1. Build the all_v2 numeric feature set (color differences, second-order colors,
   magnitude/flux statistics, redshift transforms + redshift x band/color
   interactions, sky sin/cos, binned-color category codes). FE is a verbatim copy
   of kaggle/gbdt_orig/run.py::add_features (the all_v2 family) and is applied
   identically to competition train, test, and original-SDSS17 rows. The matrix is
   homogeneous float32 (no native categoricals), so CatBoost and LightGBM share it.
2. Optuna TPE study (MedianPruner, fold-level pruning) on a STRATIFIED 150k
   subsample, 3-fold, maximizing balanced accuracy, to tune CatBoost-GPU
   (learning_rate, depth, l2_leaf_reg, random_strength, bagging_temperature,
   border_count). Tuning uses NO original augmentation (fast, only the hyper-param
   RANKING matters) and a reduced iteration cap; class_weights=[1,3.25,5] is fixed
   (matches catv3, drives balanced accuracy).
3. Refit the best CatBoost config with the FULL 5-fold contract + original aug at
   weight 0.1 -> oof_optunacat.npy / test_optunacat.npy. SAVED FIRST (primary).
4. Same recipe for LightGBM-GPU (balanced sample weights) -> oof_optunalgb.npy /
   test_optunalgb.npy. Time-budget gated so the whole kernel finishes < 11h; if
   time runs short LightGBM degrades gracefully (defaults, or skipped) but the
   already-saved CatBoost artifacts are never put at risk.

Contract (so the new OOF stacks with the existing pool)
-------------------------------------------------------
  * Labels: GALAXY=0, QSO=1, STAR=2 (alphabetical / class-name order).
  * Folds: StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split over
    the 577347 competition rows ONLY, integer labels in original train-CSV order
    (the split depends only on y+seed, so it auto-aligns with every other artifact).
  * Original SDSS17 rows go ONLY into each fold's TRAIN pool (never validation),
    down-weighted to 0.1; OOF is computed only on competition train rows.
  * OOF rows in train-CSV order; test rows reindexed to sample_submission id order.
  * Saved arrays are float32, row-normalized, shape (577347,3) / (247435,3).

P100 / environment reality (hard-won; see CLAUDE.md)
----------------------------------------------------
  * Kaggle competition GPU is a single P100 (sm_60 Pascal). CatBoost-GPU and
    LightGBM-GPU have their own CUDA/OpenCL kernels and DO run on P100 — only the
    torch/cuDF stacks dropped Pascal, and this kernel uses neither.
  * CatBoost task_type='GPU' only errors at .fit(); detect up front via
    catboost.utils.get_gpu_device_count() and fall back to CPU. devices='0'
    (single P100, NEVER '0:1').
  * LightGBM GPU support depends on the build; probe with a tiny train and fall
    back to device='cpu' if the build lacks GPU.
  * Early stopping must track balanced accuracy, not logloss: LightGBM uses a
    custom feval (the ONLY metric, via metric='None' + first_metric_only) so it
    drives best_iteration; CatBoost uses eval_metric='TotalF1:average=Macro', the
    macro/recall-sensitive proxy catv3 validated on this competition (CatBoost has
    no built-in macro-recall and a Python custom metric is not GPU-supported).
  * Internet enabled only to `pip install optuna` if the image lacks it.

Outputs to /kaggle/working/
---------------------------
  oof_optunacat.npy   (577347,3) float32   test_optunacat.npy   (247435,3) float32
  oof_optunalgb.npy   (577347,3) float32   test_optunalgb.npy   (247435,3) float32
  submission_optunacat.csv / submission_optunalgb.csv / submission.csv (cat argmax)
  results.txt   best params + per-fold + overall OOF balanced accuracy + per-class recall
"""

from __future__ import annotations

import os

# Keep CPU helper libs (BLAS/OpenMP) from oversubscribing the small Kaggle vCPU set.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc
import glob
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

T0 = time.perf_counter()

# Optuna: import, pip-installing only if missing (internet is enabled for this).
try:
    import optuna
except Exception:  # pragma: no cover - depends on the Kaggle image
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "optuna>=3.2"],
        check=False,
    )
    import optuna

import catboost as cb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

optuna.logging.set_verbosity(optuna.logging.WARNING)

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

N_TRAIN_EXPECT = 577347
N_TEST_EXPECT = 247435

# Binned-color category codes (reconstruct the competition categoricals exactly).
SPEC_MAP = {"M": 0, "G/K": 1, "A/F": 2, "O/B": 3}
POP_MAP = {"Blue_Cloud": 0, "Red_Sequence": 1}

ORIGINAL_WEIGHT = 0.1          # appended original rows down-weighted in TRAIN pool
CAT_CLASS_WEIGHTS = [1.0, 3.25, 5.0]  # up-weight rare STAR for balanced accuracy
CLIP = 1e-15
PREDICT_BATCH = 80_000

# Full-refit budgets.
CAT_ITERATIONS = 5000
CAT_EARLY = 260
LGB_ROUNDS = 4000
LGB_EARLY = 200

# Optuna tuning budgets (kept cheap; the full refit uses the budgets above).
TUNE_SUBSAMPLE = 150_000
TUNE_FOLDS = 3
N_TRIALS_CAT = 80
N_TRIALS_LGB = 80
TUNE_ITERS_CAT = 1200
TUNE_EARLY_CAT = 80
TUNE_ROUNDS_LGB = 1500
TUNE_EARLY_LGB = 80

# Wall-clock budget. Kaggle kills at 12h; stay under 11h. Reservations below are
# worst-case (CPU-fallback) so the LightGBM stage degrades before it ever risks
# overrunning. CatBoost artifacts are saved before any LightGBM work begins.
HARD_DEADLINE_S = 11 * 3600
CAT_TUNE_TIMEOUT = 8100        # 2.25h
LGB_TUNE_TIMEOUT = 6300        # 1.75h
RESERVE_AFTER_CAT_TUNE = 6 * 3600 + 1800   # keep >=6.5h for cat refit + all of lgb
LGB_REFIT_RESERVE = 7200       # keep this much for the lgb refit (CPU worst case)
LGB_MIN_REFIT = 2400           # below this remaining, skip lgb entirely

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = WORK / "results.txt"

np.random.seed(SEED)

_RESULT_LINES: list[str] = []


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


def emit(msg: str) -> None:
    """Print AND record for results.txt (flushed immediately so partial runs survive)."""
    print(msg, flush=True)
    _RESULT_LINES.append(str(msg))
    RESULTS_PATH.write_text("\n".join(_RESULT_LINES) + "\n")


def remaining() -> float:
    return HARD_DEADLINE_S - (time.perf_counter() - T0)


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
    # Mirror dataset: cindyxue1122/s6e6-original-sdss17 (recursive glob).
    candidates = [Path(p) for p in glob.glob("/kaggle/input/**/star_classification.csv", recursive=True)]
    seen: list[Path] = []
    for c in candidates:
        if c not in seen:
            seen.append(c)
    for p in seen:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find star_classification.csv (SDSS17 original mirror).")


# ----------------------------------------------------------------------------
# Original-dataset categorical reconstruction (verified exact on competition data)
# ----------------------------------------------------------------------------
def rebuild_original_cats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spectral_type"] = (
        pd.cut(out["r"] - out["g"], [-np.inf, -1.0, -0.5, 0.0, np.inf],
               labels=["M", "G/K", "A/F", "O/B"]).astype(str)
    )
    out["galaxy_population"] = (
        pd.cut(out["u"] - out["r"], [-np.inf, 2.2, np.inf],
               labels=["Blue_Cloud", "Red_Sequence"]).astype(str)
    )
    return out


# ----------------------------------------------------------------------------
# Feature engineering — all_v2 numeric matrix (verbatim copy of gbdt_orig add_features).
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

    # Color ratios (safe division).
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

    # Flux features: flux_b = 10**(-0.4*b), band magnitudes clipped for safety.
    fluxes = []
    for b in BANDS:
        clipped = np.clip(out[b].to_numpy(dtype="float32"), -30.0, 30.0)
        flux = np.power(np.float32(10.0), np.float32(-0.4) * clipped).astype("float32")
        out[f"flux_{b}"] = flux
        fluxes.append(flux)
    flux_mat = np.vstack(fluxes).T
    out["flux_mean"] = np.nanmean(flux_mat, axis=1).astype("float32")
    out["flux_std"] = np.nanstd(flux_mat, axis=1).astype("float32")

    # Binned-color categoricals as small int codes.
    out["spectral_type_code"] = (
        df["spectral_type"].astype(str).map(SPEC_MAP).fillna(-1).astype("int32")
    )
    out["galaxy_population_code"] = (
        df["galaxy_population"].astype(str).map(POP_MAP).fillna(-1).astype("int32")
    )

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# ----------------------------------------------------------------------------
# Metrics / IO helpers
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
    emit("  per-class recall: "
         f"GALAXY={rec['GALAXY']:.4f} QSO={rec['QSO']:.4f} STAR={rec['STAR']:.4f}")
    emit("  confusion (rows=true GALAXY/QSO/STAR, cols=pred):")
    for i, name in enumerate(CLASSES):
        emit(f"    {name:7s} {cm[i].tolist()}")
    return ba


def save_model_arrays(model_name: str, oof: np.ndarray, test_pred: np.ndarray,
                      sample: pd.DataFrame, write_default_sub: bool = False) -> None:
    oof = normalize_proba(oof)
    test_pred = normalize_proba(test_pred)
    np.save(WORK / f"oof_{model_name}.npy", oof.astype("float32"))
    np.save(WORK / f"test_{model_name}.npy", test_pred.astype("float32"))
    sub = sample.copy()
    sub[TARGET] = [INV_MAP[i] for i in test_pred.argmax(axis=1)]
    sub.to_csv(WORK / f"submission_{model_name}.csv", index=False)
    if write_default_sub:
        sub.to_csv(WORK / "submission.csv", index=False)
    log(f"saved oof_{model_name}.npy {oof.shape}, test_{model_name}.npy {test_pred.shape}")


# ----------------------------------------------------------------------------
# LightGBM balanced-accuracy feval (the ONLY metric, drives early stopping).
# ----------------------------------------------------------------------------
def feval_bal_acc(y_pred_raw, dataset):
    y_true = dataset.get_label().astype(int)
    n = y_true.shape[0]
    arr = np.asarray(y_pred_raw)
    if arr.ndim == 2:
        preds = arr
    else:
        # 1-D flattened column-major: [class0 over all rows, class1 ..., class2 ...]
        preds = arr.reshape(N_CLASSES, n).T
    return "bal_acc", balanced_accuracy_score(y_true, preds.argmax(axis=1)), True


# ----------------------------------------------------------------------------
# GPU detection
# ----------------------------------------------------------------------------
def detect_cat_gpu() -> int:
    try:
        from catboost.utils import get_gpu_device_count
        n = int(get_gpu_device_count())
    except Exception as e:  # pragma: no cover
        log(f"CatBoost GPU probe failed ({e!r}); using CPU")
        n = 0
    log(f"CatBoost GPU devices detected: {n}")
    return n


def detect_lgb_device() -> str:
    try:
        xp = np.random.rand(2000, 6).astype("float32")
        yp = np.random.randint(0, N_CLASSES, size=2000)
        d = lgb.Dataset(xp, label=yp)
        lgb.train(
            {"objective": "multiclass", "num_class": N_CLASSES, "device": "gpu",
             "num_leaves": 7, "max_bin": 255, "verbosity": -1},
            d, num_boost_round=3,
        )
        log("LightGBM GPU build available -> device='gpu'")
        return "gpu"
    except Exception as e:
        log(f"LightGBM GPU probe failed ({e!r}); using device='cpu'")
        return "cpu"


# ----------------------------------------------------------------------------
# CatBoost: param assembly + full 5-fold refit (with original augmentation)
# ----------------------------------------------------------------------------
def cat_params(tuned: dict, iterations: int, early: int, verbose, n_gpu: int) -> dict:
    p = dict(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        iterations=iterations,
        depth=int(tuned.get("depth", 8)),
        learning_rate=float(tuned.get("learning_rate", 0.042)),
        l2_leaf_reg=float(tuned.get("l2_leaf_reg", 8.0)),
        random_strength=float(tuned.get("random_strength", 1.2)),
        bootstrap_type="Bayesian",
        bagging_temperature=float(tuned.get("bagging_temperature", 0.2)),
        border_count=int(tuned.get("border_count", 254)),
        class_weights=CAT_CLASS_WEIGHTS,
        random_seed=SEED,
        early_stopping_rounds=early,
        allow_writing_files=False,
        verbose=verbose,
    )
    if n_gpu > 0:
        p["task_type"] = "GPU"
        p["devices"] = "0"
    else:
        p["task_type"] = "CPU"
        p["thread_count"] = 4
    return p


def cat_predict_batched(model, X: np.ndarray) -> np.ndarray:
    parts = []
    for s in range(0, len(X), PREDICT_BATCH):
        parts.append(model.predict_proba(X[s:s + PREDICT_BATCH]).astype("float32"))
    return np.vstack(parts).astype("float32")


def train_cat_full(tuned, X_comp, y_comp, X_orig, y_orig, X_test, folds, n_gpu):
    oof = np.zeros((len(y_comp), N_CLASSES), dtype="float32")
    test_pred = np.zeros((len(X_test), N_CLASSES), dtype="float32")
    fold_scores: list[float] = []
    params = cat_params(tuned, CAT_ITERATIONS, CAT_EARLY, 250, n_gpu)

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        t0 = time.perf_counter()
        X_fit = np.vstack([X_comp[tr_idx], X_orig])
        y_fit = np.concatenate([y_comp[tr_idx], y_orig]).astype("int64")
        w = np.ones(len(y_fit), dtype="float32")
        w[len(tr_idx):] = np.float32(ORIGINAL_WEIGHT)
        X_va, y_va = X_comp[va_idx], y_comp[va_idx]

        train_pool = Pool(X_fit, y_fit, weight=w)
        valid_pool = Pool(X_va, y_va)
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        va_p = model.predict_proba(valid_pool).astype("float32")
        te_p = cat_predict_batched(model, X_test)
        oof[va_idx] = va_p
        test_pred += te_p / N_FOLDS

        s = balanced_accuracy_score(y_va, va_p.argmax(axis=1))
        fold_scores.append(float(s))
        log(f"CAT fold {fold}: BA={s:.6f} best_iter={model.get_best_iteration()} "
            f"({time.perf_counter() - t0:.0f}s)")
        del train_pool, valid_pool, model, X_fit, y_fit, w
        gc.collect()
    return oof, test_pred, fold_scores


# ----------------------------------------------------------------------------
# LightGBM: param assembly + full 5-fold refit (with original augmentation)
# ----------------------------------------------------------------------------
def lgb_params(tuned: dict, device: str) -> dict:
    p = dict(
        objective="multiclass",
        num_class=N_CLASSES,
        metric="None",
        learning_rate=float(tuned.get("learning_rate", 0.025)),
        num_leaves=int(tuned.get("num_leaves", 80)),
        min_child_samples=int(tuned.get("min_child_samples", 80)),
        feature_fraction=float(tuned.get("feature_fraction", 0.72)),
        bagging_fraction=float(tuned.get("bagging_fraction", 0.82)),
        bagging_freq=1,
        reg_alpha=float(tuned.get("reg_alpha", 0.05)),
        reg_lambda=float(tuned.get("reg_lambda", 10.0)),
        max_depth=-1,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
        first_metric_only=True,
        device=device,
    )
    if device == "gpu":
        p["max_bin"] = 255
    return p


def train_lgb_full(tuned, X_comp, y_comp, X_orig, y_orig, X_test, folds, device):
    oof = np.zeros((len(y_comp), N_CLASSES), dtype="float32")
    test_pred = np.zeros((len(X_test), N_CLASSES), dtype="float32")
    fold_scores: list[float] = []
    params = lgb_params(tuned, device)

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        t0 = time.perf_counter()
        X_fit = np.vstack([X_comp[tr_idx], X_orig])
        y_fit = np.concatenate([y_comp[tr_idx], y_orig]).astype("int64")
        sw = compute_sample_weight("balanced", y_fit).astype("float32")
        sw[len(tr_idx):] *= np.float32(ORIGINAL_WEIGHT)
        X_va, y_va = X_comp[va_idx], y_comp[va_idx]

        dtrain = lgb.Dataset(X_fit, label=y_fit, weight=sw)
        dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        booster = lgb.train(
            params, dtrain, num_boost_round=LGB_ROUNDS,
            valid_sets=[dvalid], valid_names=["valid"], feval=feval_bal_acc,
            callbacks=[lgb.early_stopping(LGB_EARLY, first_metric_only=True),
                       lgb.log_evaluation(250)],
        )
        bi = booster.best_iteration
        va_p = booster.predict(X_va, num_iteration=bi).astype("float32")
        te_p = booster.predict(X_test, num_iteration=bi).astype("float32")
        oof[va_idx] = va_p
        test_pred += te_p / N_FOLDS

        s = balanced_accuracy_score(y_va, va_p.argmax(axis=1))
        fold_scores.append(float(s))
        log(f"LGB fold {fold}: BA={s:.6f} best_iter={bi} ({time.perf_counter() - t0:.0f}s)")
        del dtrain, dvalid, booster, X_fit, y_fit, sw
        gc.collect()
    return oof, test_pred, fold_scores


# ----------------------------------------------------------------------------
# Optuna objectives (3-fold on a 150k stratified subsample; no original aug)
# ----------------------------------------------------------------------------
def make_cat_objective(X_sub, y_sub, sub_folds, n_gpu):
    def objective(trial: "optuna.Trial") -> float:
        tuned = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            depth=trial.suggest_int("depth", 6, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            random_strength=trial.suggest_float("random_strength", 0.2, 5.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
            border_count=trial.suggest_categorical("border_count", [128, 170, 200, 254]),
        )
        params = cat_params(tuned, TUNE_ITERS_CAT, TUNE_EARLY_CAT, False, n_gpu)
        scores: list[float] = []
        for i, (tr, va) in enumerate(sub_folds):
            model = CatBoostClassifier(**params)
            # verbose lives in the constructor params (cat_params -> verbose=False);
            # do NOT also pass it to fit() — setting a verbosity-group key in both
            # places can raise a CatBoostError. Matches the proven catv3/gbdt_orig path.
            model.fit(X_sub[tr], y_sub[tr], eval_set=(X_sub[va], y_sub[va]),
                      use_best_model=True)
            p = model.predict_proba(X_sub[va])
            scores.append(float(balanced_accuracy_score(y_sub[va], p.argmax(axis=1))))
            del model
            gc.collect()
            trial.report(float(np.mean(scores)), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))
    return objective


def make_lgb_objective(X_sub, y_sub, sub_folds, device):
    def objective(trial: "optuna.Trial") -> float:
        tuned = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
        )
        params = lgb_params(tuned, device)
        scores: list[float] = []
        for i, (tr, va) in enumerate(sub_folds):
            sw = compute_sample_weight("balanced", y_sub[tr]).astype("float32")
            dtr = lgb.Dataset(X_sub[tr], label=y_sub[tr], weight=sw)
            dva = lgb.Dataset(X_sub[va], label=y_sub[va], reference=dtr)
            booster = lgb.train(
                params, dtr, num_boost_round=TUNE_ROUNDS_LGB,
                valid_sets=[dva], valid_names=["valid"], feval=feval_bal_acc,
                callbacks=[lgb.early_stopping(TUNE_EARLY_LGB, first_metric_only=True),
                           lgb.log_evaluation(0)],
            )
            p = booster.predict(X_sub[va], num_iteration=booster.best_iteration)
            scores.append(float(balanced_accuracy_score(y_sub[va], p.argmax(axis=1))))
            del booster, dtr, dva
            gc.collect()
            trial.report(float(np.mean(scores)), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))
    return objective


def run_study(objective, n_trials: int, timeout: float, tag: str) -> dict:
    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    try:
        study.optimize(objective, n_trials=n_trials, timeout=max(1.0, timeout),
                       gc_after_trial=True)
    except Exception as e:  # pragma: no cover
        log(f"{tag} study.optimize raised {e!r}; using best-so-far / defaults")
    n_done = len([t for t in study.trials if t.state.name == "COMPLETE"])
    if n_done == 0:
        emit(f"  [{tag}] no completed trials -> falling back to default params")
        return {}
    emit(f"  [{tag}] completed trials: {n_done} (pruned/failed: {len(study.trials) - n_done})")
    emit(f"  [{tag}] best tuning CV balanced accuracy: {study.best_value:.6f}")
    emit(f"  [{tag}] best params: {study.best_params}")
    return dict(study.best_params)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    emit("# s6e6 optunabase : Optuna-tuned CatBoost-GPU + LightGBM-GPU base models")
    emit(f"# seed={SEED} folds={N_FOLDS} original_weight={ORIGINAL_WEIGHT} "
         f"tune_subsample={TUNE_SUBSAMPLE} tune_folds={TUNE_FOLDS}")
    print("pandas:", pd.__version__, "| numpy:", np.__version__,
          "| catboost:", cb.__version__, "| lightgbm:", lgb.__version__,
          "| optuna:", optuna.__version__, flush=True)

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

    # Align test to sample_submission row order.
    if ID_COL in test.columns and ID_COL in sample.columns:
        test = test.set_index(ID_COL).loc[sample[ID_COL]].reset_index()
        log("test reordered to sample_submission id order")

    # Labels in train-CSV order (GALAXY=0, QSO=1, STAR=2).
    y_comp = train[TARGET].astype(str).map(LABEL_MAP).to_numpy()
    assert not np.any(np.isnan(y_comp.astype(float))), "unmapped class label in train"
    y_comp = y_comp.astype("int64")

    # Original dataset: reconstruct cats, filter to valid classes + sane bands.
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

    # Features (identical pipeline for all three frames).
    log("building features (all_v2) ...")
    Xc_df = add_features(train)
    Xt_df = add_features(test)
    Xo_df = add_features(orig)
    feature_names = list(Xc_df.columns)
    assert list(Xt_df.columns) == feature_names and list(Xo_df.columns) == feature_names

    X_comp = Xc_df.fillna(0.0).to_numpy(dtype="float32")
    X_test = Xt_df.fillna(0.0).to_numpy(dtype="float32")
    X_orig = Xo_df.fillna(0.0).to_numpy(dtype="float32")
    log(f"feature matrix: comp={X_comp.shape} test={X_test.shape} orig={X_orig.shape} "
        f"n_features={len(feature_names)}")
    emit(f"# n_features={len(feature_names)} n_orig_rows={len(X_orig)}")
    del Xc_df, Xt_df, Xo_df, orig, orig_raw, train, test
    gc.collect()

    # Contract folds (depend only on y_comp + seed => align with all artifacts).
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X_comp, y_comp))

    # Stratified subsample for tuning.
    sub_size = min(TUNE_SUBSAMPLE, len(y_comp) - 1000)
    sub_idx, _ = train_test_split(
        np.arange(len(y_comp)), train_size=sub_size, stratify=y_comp, random_state=SEED
    )
    X_sub = X_comp[sub_idx]
    y_sub = y_comp[sub_idx]
    sub_folds = list(StratifiedKFold(n_splits=TUNE_FOLDS, shuffle=True,
                                     random_state=SEED).split(X_sub, y_sub))
    log(f"tuning subsample: {X_sub.shape} ({TUNE_FOLDS}-fold)")

    cat_n_gpu = detect_cat_gpu()
    lgb_device = detect_lgb_device()

    # ----------------------------------------------------------------
    # CatBoost: tune -> full refit -> SAVE FIRST (primary artifact).
    # ----------------------------------------------------------------
    emit("")
    emit("==================== CatBoost-GPU (optunacat) ====================")
    cat_tune_to = min(CAT_TUNE_TIMEOUT, max(600.0, remaining() - RESERVE_AFTER_CAT_TUNE))
    log(f"CatBoost Optuna tuning (timeout={cat_tune_to:.0f}s, n_trials<= {N_TRIALS_CAT}) ...")
    cat_best = run_study(
        make_cat_objective(X_sub, y_sub, sub_folds, cat_n_gpu),
        N_TRIALS_CAT, cat_tune_to, "optunacat",
    )

    log("CatBoost full 5-fold refit (+ original augmentation) ...")
    oof, test_pred, fs = train_cat_full(
        cat_best, X_comp, y_comp, X_orig, y_orig, X_test, folds, cat_n_gpu
    )
    report_oof("optunacat (CatBoost)", y_comp, oof, fs)
    save_model_arrays("optunacat", oof, test_pred, sample, write_default_sub=True)
    del oof, test_pred
    gc.collect()

    # ----------------------------------------------------------------
    # LightGBM: time-budget gated tune -> full refit -> save.
    # ----------------------------------------------------------------
    emit("")
    emit("==================== LightGBM-GPU (optunalgb) ====================")
    rem = remaining()
    log(f"remaining budget before LightGBM: {rem:.0f}s")
    if rem < LGB_MIN_REFIT:
        emit(f"  [optunalgb] insufficient time ({rem:.0f}s) -> SKIPPED "
             "(CatBoost artifacts already saved).")
    else:
        lgb_tune_budget = rem - LGB_REFIT_RESERVE
        if lgb_tune_budget >= 900:
            lgb_tune_to = min(LGB_TUNE_TIMEOUT, lgb_tune_budget)
            log(f"LightGBM Optuna tuning (timeout={lgb_tune_to:.0f}s, n_trials<= {N_TRIALS_LGB}) ...")
            lgb_best = run_study(
                make_lgb_objective(X_sub, y_sub, sub_folds, lgb_device),
                N_TRIALS_LGB, lgb_tune_to, "optunalgb",
            )
        else:
            emit("  [optunalgb] not enough time to tune -> default params, refit only")
            lgb_best = {}

        log("LightGBM full 5-fold refit (+ original augmentation) ...")
        oof, test_pred, fs = train_lgb_full(
            lgb_best, X_comp, y_comp, X_orig, y_orig, X_test, folds, lgb_device
        )
        report_oof("optunalgb (LightGBM)", y_comp, oof, fs)
        save_model_arrays("optunalgb", oof, test_pred, sample)
        del oof, test_pred
        gc.collect()

    emit("")
    emit("DONE")
    log("ALL DONE")


if __name__ == "__main__":
    main()
