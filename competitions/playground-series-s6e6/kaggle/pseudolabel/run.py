"""s6e6 pseudolabel: transductive pseudo-labeling of the TEST set, appended to the
TRAIN pool, with LightGBM + XGBoost and balanced-accuracy-aware early stopping.

Self-contained Kaggle SCRIPT kernel (CPU). Does NOT import the user's pipeline
modules; the feature engineering is copied from kaggle/gbdt_orig/run.py so the OOF
stacks against the existing artifacts.

What it does
------------
1. Loads competition train/test + sample_submission and the ORIGINAL SDSS17 dataset.
2. Obtains CONFIDENT pseudo-labels for TEST rows in one of two modes:
     * EXTERNAL  -- if a pseudo-label CSV is mounted under /kaggle/input (the main
       agent supplies the current best stack's TEST prediction, see "Pseudo-label
       file format" below), those pseudo-labels are used for ALL folds.
     * SELF-GEN  -- otherwise, a quick LightGBM is trained PER FOLD on that fold's
       competition-train rows (+ original) and predicts TEST; only rows with
       max-prob >= CONF_THRESHOLD become that fold's pseudo set. Per-fold generation
       keeps the validation fold OUT of the pseudo-label generator, so the OOF is
       honest (no val-label -> pseudo -> train leakage).
3. Appends the confident pseudo-TEST rows to each fold's TRAIN pool at low weight
   (PSEUDO_WEIGHT) PLUS the original SDSS17 at low weight (ORIGINAL_WEIGHT).
4. 5-fold StratifiedKFold(5, shuffle=True, random_state=42) on the COMPETITION rows
   ONLY. Pseudo / original rows go ONLY into TRAIN folds, NEVER validation; OOF is
   computed only on competition rows -> aligns with every other model's OOF.
5. Trains LightGBM + XGBoost (multiclass, balanced-accuracy early stopping) and saves
   oof_pseudolgb.npy / test_pseudolgb.npy and oof_pseudoxgb.npy / test_pseudoxgb.npy,
   plus results.txt and submission_*.csv.

Contract (must match existing artifacts so the new OOF stacks):
  * Labels: GALAXY=0, QSO=1, STAR=2.
  * Folds: StratifiedKFold(5, shuffle=True, random_state=42).split(X_comp, y_comp)
    on the 577347 competition rows ONLY, in original train-CSV row order.
  * OOF (577347, 3) float32 in train-CSV order; test (247435, 3) float32 in
    sample_submission order.

LEAKAGE CAVEAT (read this)
--------------------------
Pseudo-labeling is *transductive*: the model trains on labels it assigned to TEST
rows, so its TEST predictions are influenced by the unlabeled TEST distribution.
The OOF is kept honest only in the sense that pseudo / original rows are excluded
from every validation fold (OOF is scored on real competition labels only).
  * SELF-GEN mode additionally generates the pseudo-labels per-fold using ONLY that
    fold's training rows, so a validation row's true label can NOT leak into the main
    model through a pseudo-label -> the OOF is honest enough to stack on.
  * EXTERNAL mode uses pseudo-labels from a stack that already saw ALL competition
    train labels (incl. rows that act as validation here). Those pseudo-labels can
    encode val-fold information, so the EXTERNAL OOF is OPTIMISTIC and should be
    judged by nested-CV, not taken at face value. EXTERNAL mode's real deliverable is
    the TEST prediction (a transductive boost), not the OOF.

Pseudo-label file format (EXTERNAL mode)
----------------------------------------
Drop a CSV anywhere under /kaggle/input (the kernel globs for it). Preferred name:
``pseudo_test.csv`` (aliases accepted: see PSEUDO_CANDIDATES). Two layouts supported:
  * PROBABILITIES: columns {GALAXY, QSO, STAR} (+ optional ``id``). Rows with
    max-prob >= CONF_THRESHOLD are kept; pseudo-label = argmax. If ``id`` is present
    rows are joined to the test set by id, else assumed in sample_submission order.
  * LABELS: columns {id, class} (class in {GALAXY,QSO,STAR}). ALL provided rows are
    treated as already-confident pseudo-labels (the agent pre-filtered) and used at
    PSEUDO_WEIGHT.
NOTE: the supplied metadata only mounts ``cindyxue1122/s6e6-original-sdss17``. To use
EXTERNAL mode either add ``cindyxue1122/s6e6-pipeline-code`` to dataset_sources or
place ``pseudo_test.csv`` inside the sdss17 dataset. With nothing mounted, the kernel
runs the honest SELF-GEN path automatically.
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

N_TRAIN_EXPECT = 577347
N_TEST_EXPECT = 247435

# Pseudo-labeling knobs.
PSEUDO_WEIGHT = 0.3            # confident pseudo-TEST rows down-weighted in TRAIN pool
ORIGINAL_WEIGHT = 0.1         # original SDSS17 rows down-weighted in TRAIN pool
CONF_THRESHOLD = 0.995        # min max-prob for a TEST row to become a pseudo-label
MAX_PSEUDO_PER_FOLD = 220000  # safety cap on #pseudo rows (keeps runtime/memory bounded)

# Main GBDT hyper-params (lgbm-v3 / gbdt_orig spirit).
LGB_ROUNDS = 4000
XGB_ROUNDS = 4000
LGB_EARLY = 200
XGB_EARLY = 200
CLIP = 1e-15

# Quick LightGBM used to SELF-GENERATE pseudo-labels (cheap, fixed #rounds).
QUICK_ROUNDS = 400
QUICK_LR = 0.05
QUICK_LEAVES = 63

# Preferred external pseudo-label file names (globbed across /kaggle/input).
PSEUDO_CANDIDATES = [
    "pseudo_test.csv",
    "pseudo_labels.csv",
    "best_stack_test.csv",
    "stack_test.csv",
    "test_pseudo.csv",
]
# CSV file names that are definitely NOT pseudo-label files (never treat as such).
_RESERVED_CSVS = {
    "train.csv",
    "test.csv",
    "sample_submission.csv",
    "star_classification.csv",
}

# Codes for binned-color categoricals (these reconstruct the competition cats).
SPEC_MAP = {"M": 0, "G/K": 1, "A/F": 2, "O/B": 3}
POP_MAP = {"Blue_Cloud": 0, "Red_Sequence": 1}

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = WORK / "results.txt"

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


def find_pseudo_csv() -> Path | None:
    """Locate an EXTERNAL pseudo-label CSV under /kaggle/input, or return None.

    Strategy:
      1. exact preferred file names (PSEUDO_CANDIDATES), globbed recursively;
      2. otherwise scan every CSV under /kaggle/input (excluding reserved comp/orig
         files) and accept the first whose columns are a superset of
         {GALAXY,QSO,STAR} or equal to a {id,class}-style label table.
    """
    inp = Path("/kaggle/input")
    if not inp.exists():
        return None
    for name in PSEUDO_CANDIDATES:
        hits = sorted(inp.glob(f"**/{name}"))
        if hits:
            return hits[0]
    # Fallback: column-based detection.
    for hit in sorted(inp.glob("**/*.csv")):
        if hit.name in _RESERVED_CSVS:
            continue
        try:
            header = pd.read_csv(hit, nrows=5)
        except Exception:
            continue
        cols = set(map(str, header.columns))
        if {"GALAXY", "QSO", "STAR"}.issubset(cols):
            return hit
        if "class" in cols and ("id" in cols) and len(cols) <= 3:
            return hit
    return None


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
# Feature engineering — numeric-only matrix (copied from gbdt_orig).
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
    emit("  confusion (rows=true GALAXY/QSO/STAR, cols=pred):")
    for i, name in enumerate(CLASSES):
        emit(f"    {name:7s} {cm[i].tolist()}")
    flush_results()
    return ba


def save_model_arrays(model_name: str, oof: np.ndarray, test_pred: np.ndarray,
                      sample: pd.DataFrame) -> None:
    oof = normalize_proba(oof)
    test_pred = normalize_proba(test_pred)
    np.save(WORK / f"oof_{model_name}.npy", oof.astype("float32"))
    np.save(WORK / f"test_{model_name}.npy", test_pred.astype("float32"))
    sub = sample.copy()
    sub[TARGET] = [INV_MAP[i] for i in test_pred.argmax(axis=1)]
    sub.to_csv(WORK / f"submission_{model_name}.csv", index=False)
    log(f"saved oof_{model_name}.npy {oof.shape}, test_{model_name}.npy {test_pred.shape}")


# ----------------------------------------------------------------------------
# Pseudo-label resolution
# ----------------------------------------------------------------------------
def load_external_pseudo(sample: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (idx_into_test, labels) for an EXTERNAL pseudo-label CSV, or None.

    ``idx_into_test`` indexes the TEST matrix (already in sample_submission order);
    ``labels`` are integer pseudo-labels for those rows.
    """
    path = find_pseudo_csv()
    if path is None:
        return None
    df = pd.read_csv(path)
    cols = set(map(str, df.columns))
    log(f"EXTERNAL pseudo-label file found: {path} columns={sorted(cols)} rows={len(df)}")

    sample_ids = sample[ID_COL].to_numpy()
    pos_of_id = {int(v): i for i, v in enumerate(sample_ids)}

    # --- probability layout ------------------------------------------------
    if {"GALAXY", "QSO", "STAR"}.issubset(cols):
        probs_full = np.zeros((len(sample_ids), N_CLASSES), dtype="float64")
        seen = np.zeros(len(sample_ids), dtype=bool)
        prob_mat = df[["GALAXY", "QSO", "STAR"]].to_numpy(dtype="float64")
        if ID_COL in cols:
            ids = df[ID_COL].to_numpy()
            for j, idv in enumerate(ids):
                pos = pos_of_id.get(int(idv))
                if pos is not None:
                    probs_full[pos] = prob_mat[j]
                    seen[pos] = True
        else:
            if len(df) != len(sample_ids):
                raise ValueError(
                    f"prob pseudo file has no 'id' and {len(df)} rows != {len(sample_ids)} test rows"
                )
            probs_full[:] = prob_mat
            seen[:] = True
        maxp = probs_full.max(axis=1)
        keep = seen & (maxp >= CONF_THRESHOLD)
        idx = np.where(keep)[0]
        labels = probs_full[idx].argmax(axis=1).astype("int64")
        log(f"EXTERNAL(prob): {len(idx)} rows >= {CONF_THRESHOLD} of {seen.sum()} provided")
        return idx, labels

    # --- label layout ({id,class}) ----------------------------------------
    if "class" in cols and ID_COL in cols:
        labels_full = np.full(len(sample_ids), -1, dtype="int64")
        ids = df[ID_COL].to_numpy()
        cls = df["class"].astype(str).str.upper().map(LABEL_MAP).to_numpy()
        for j, idv in enumerate(ids):
            pos = pos_of_id.get(int(idv))
            if pos is not None and not np.isnan(cls[j]):
                labels_full[pos] = int(cls[j])
        idx = np.where(labels_full >= 0)[0]
        labels = labels_full[idx]
        log(f"EXTERNAL(label): {len(idx)} confident pseudo-labels provided (used at PSEUDO_WEIGHT)")
        return idx, labels

    log(f"EXTERNAL pseudo file {path} has unrecognized columns -> ignoring, will SELF-GEN")
    return None


def generate_self_pseudo(X_tr, y_tr, X_orig, y_orig, X_test) -> tuple[np.ndarray, np.ndarray]:
    """Train a quick LightGBM on fold-train (+ original) and return confident TEST rows.

    Returns (idx_into_test, labels). Fold-safe: only the fold's training rows are used,
    so the held-out validation labels never influence the pseudo-labels.
    """
    import lightgbm as lgb

    X_fit = np.vstack([X_tr, X_orig])
    y_fit = np.concatenate([y_tr, y_orig]).astype("int64")
    sw = compute_sample_weight(class_weight="balanced", y=y_fit).astype("float32")
    sw[len(y_tr):] *= np.float32(ORIGINAL_WEIGHT)

    params = dict(
        objective="multiclass",
        num_class=N_CLASSES,
        metric="None",
        learning_rate=QUICK_LR,
        num_leaves=QUICK_LEAVES,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=10.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    dtrain = lgb.Dataset(X_fit, label=y_fit, weight=sw)
    booster = lgb.train(params, dtrain, num_boost_round=QUICK_ROUNDS)
    proba = booster.predict(X_test).astype("float32")
    maxp = proba.max(axis=1)
    idx = np.where(maxp >= CONF_THRESHOLD)[0]
    # Cap (keep the most-confident rows) to bound runtime / memory.
    if len(idx) > MAX_PSEUDO_PER_FOLD:
        order = np.argsort(maxp[idx])[::-1][:MAX_PSEUDO_PER_FOLD]
        idx = idx[order]
    labels = proba[idx].argmax(axis=1).astype("int64")
    del dtrain, booster, X_fit, y_fit, sw, proba
    gc.collect()
    return idx, labels


# ----------------------------------------------------------------------------
# Fold fit-data assembly: comp-train + pseudo-TEST + original.
# ----------------------------------------------------------------------------
def make_fit_data(X_comp, y_comp, tr_idx, X_test, ps_idx, ps_y, X_orig, y_orig):
    blocks = [X_comp[tr_idx]]
    ys = [y_comp[tr_idx]]
    n_tr = len(tr_idx)
    n_ps = len(ps_idx)
    if n_ps:
        blocks.append(X_test[ps_idx])
        ys.append(ps_y)
    blocks.append(X_orig)
    ys.append(y_orig)

    X_fit = np.vstack(blocks)
    y_fit = np.concatenate(ys).astype("int64")
    sw = compute_sample_weight(class_weight="balanced", y=y_fit).astype("float32")
    # pseudo block then original block get extra down-weighting.
    if n_ps:
        sw[n_tr:n_tr + n_ps] *= np.float32(PSEUDO_WEIGHT)
    sw[n_tr + n_ps:] *= np.float32(ORIGINAL_WEIGHT)
    return X_fit, y_fit, sw


# ----------------------------------------------------------------------------
# Per-model training.
# ----------------------------------------------------------------------------
def train_lgb(X_comp, y_comp, X_test, X_orig, y_orig, folds, fold_pseudo):
    import lightgbm as lgb

    def feval_bal_acc(y_pred_raw, dataset):
        y_true = dataset.get_label().astype(int)
        n = y_true.shape[0]
        arr = np.asarray(y_pred_raw)
        if arr.ndim == 2:
            preds = arr
        else:
            preds = arr.reshape(N_CLASSES, n).T
        y_hat = preds.argmax(axis=1)
        return "bal_acc", balanced_accuracy_score(y_true, y_hat), True

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
        ps_idx, ps_y = fold_pseudo[fold - 1]
        X_fit, y_fit, sw = make_fit_data(
            X_comp, y_comp, tr_idx, X_test, ps_idx, ps_y, X_orig, y_orig
        )
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
            f"n_pseudo={len(ps_idx)} ({time.perf_counter() - t0:.0f}s)")
        del dtrain, dvalid, booster, X_fit, y_fit, sw
        gc.collect()

    return oof, test_pred, fold_scores


def train_xgb(X_comp, y_comp, X_test, X_orig, y_orig, folds, fold_pseudo):
    import xgboost as xgb

    def feval_bal_acc(y_pred, dtrain):
        y_true = dtrain.get_label().astype(int)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, N_CLASSES)
        y_hat = y_pred.argmax(axis=1)
        return "bal_acc", float(balanced_accuracy_score(y_true, y_hat))

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
        ps_idx, ps_y = fold_pseudo[fold - 1]
        X_fit, y_fit, sw = make_fit_data(
            X_comp, y_comp, tr_idx, X_test, ps_idx, ps_y, X_orig, y_orig
        )
        X_va, y_va = X_comp[va_idx], y_comp[va_idx]

        dtrain = xgb.DMatrix(X_fit, label=y_fit, weight=sw)
        dvalid = xgb.DMatrix(X_va, label=y_va)
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
            f"n_pseudo={len(ps_idx)} ({time.perf_counter() - t0:.0f}s)")
        del dtrain, dvalid, booster, X_fit, y_fit, sw
        gc.collect()

    del dtest
    gc.collect()
    return oof, test_pred, fold_scores


# ----------------------------------------------------------------------------
# Build per-fold pseudo sets (EXTERNAL once, or SELF-GEN per fold).
# ----------------------------------------------------------------------------
def build_fold_pseudo(X_comp, y_comp, X_test, X_orig, y_orig, folds, sample):
    external = load_external_pseudo(sample)
    fold_pseudo: list[tuple[np.ndarray, np.ndarray]] = []
    if external is not None:
        idx, labels = external
        if len(idx) > MAX_PSEUDO_PER_FOLD:
            log(f"EXTERNAL pseudo set {len(idx)} > cap {MAX_PSEUDO_PER_FOLD}; keeping cap")
            idx, labels = idx[:MAX_PSEUDO_PER_FOLD], labels[:MAX_PSEUDO_PER_FOLD]
        emit(f"# PSEUDO MODE = EXTERNAL  (n_pseudo={len(idx)} used for ALL folds)")
        emit("# WARNING: EXTERNAL OOF is OPTIMISTIC (pseudo-labels saw all train); "
             "judge by nested-CV, trust the TEST prediction.")
        for _ in folds:
            fold_pseudo.append((idx, labels))
        return fold_pseudo, "external"

    emit("# PSEUDO MODE = SELF-GEN  (per-fold quick-LGB, fold-safe -> honest OOF)")
    for fold, (tr_idx, _va_idx) in enumerate(folds, start=1):
        t0 = time.perf_counter()
        idx, labels = generate_self_pseudo(
            X_comp[tr_idx], y_comp[tr_idx], X_orig, y_orig, X_test
        )
        log(f"SELF-GEN fold {fold}: {len(idx)} confident pseudo rows "
            f"(>= {CONF_THRESHOLD}) ({time.perf_counter() - t0:.0f}s) "
            f"class counts={pd.Series(labels).map(INV_MAP).value_counts().to_dict()}")
        fold_pseudo.append((idx, labels))
    return fold_pseudo, "self"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    emit("# s6e6 pseudolabel : LGB + XGB with confident pseudo-TEST appended to TRAIN")
    emit(f"# seed={SEED} folds={N_FOLDS} pseudo_weight={PSEUDO_WEIGHT} "
         f"original_weight={ORIGINAL_WEIGHT} conf_threshold={CONF_THRESHOLD}")

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
    log("building features ...")
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

    # Folds: depend only on y_comp + n_splits + random_state => align with all artifacts.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X_comp, y_comp))

    # Resolve pseudo-labels (EXTERNAL once, or SELF-GEN per fold).
    fold_pseudo, mode = build_fold_pseudo(
        X_comp, y_comp, X_test, X_orig, y_orig, folds, sample
    )
    flush_results()

    summary: dict[str, float] = {}

    # ---- LightGBM ----
    log("training LightGBM ...")
    oof, test_pred, fs = train_lgb(X_comp, y_comp, X_test, X_orig, y_orig, folds, fold_pseudo)
    summary["pseudolgb"] = report_oof("pseudolgb (LightGBM + pseudo)", y_comp, oof, fs)
    save_model_arrays("pseudolgb", oof, test_pred, sample)
    del oof, test_pred
    gc.collect()

    # ---- XGBoost ----
    log("training XGBoost ...")
    oof, test_pred, fs = train_xgb(X_comp, y_comp, X_test, X_orig, y_orig, folds, fold_pseudo)
    summary["pseudoxgb"] = report_oof("pseudoxgb (XGBoost + pseudo)", y_comp, oof, fs)
    save_model_arrays("pseudoxgb", oof, test_pred, sample)
    del oof, test_pred
    gc.collect()

    emit("")
    emit("==================== SUMMARY (OOF balanced accuracy) ====================")
    emit(f"  pseudo mode = {mode}")
    for m in ("pseudolgb", "pseudoxgb"):
        emit(f"  {m} OOF BA = {summary[m]:.6f}")
    if mode == "external":
        emit("  NOTE: EXTERNAL OOF is optimistic (transductive leakage); see docstring.")
    else:
        emit("  NOTE: SELF-GEN OOF is fold-safe / honest; stack it normally.")
    emit("DONE")
    flush_results()
    log("ALL DONE")


if __name__ == "__main__":
    main()
