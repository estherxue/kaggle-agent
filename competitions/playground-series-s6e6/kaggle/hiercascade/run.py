"""s6e6 hiercascade: a 2-stage HIERARCHICAL cascade of binary LightGBMs.

Rationale: on this competition the balanced-accuracy bottleneck is GALAXY recall
(~0.959); STAR and QSO are comparatively easy because STAR sits at redshift ~0 and
QSO at high redshift. A flat 3-class model spends the same capacity everywhere. A
cascade instead PEELS OFF the most-separable class first (stage 1), then puts a
dedicated binary model on the genuinely hard remaining pair (stage 2 -> the
GALAXY-vs-other boundary that drives the metric).

Design (per stack-aligned fold):
  * Stage-1 binary LightGBM separates the EASIEST split first. We decide EMPIRICALLY
    which of {STAR-vs-rest, QSO-vs-rest} is cleaner by a quick one-fold AUC probe and
    pick the higher-AUC split (DEFAULT STAR-vs-rest on a near tie / probe failure).
  * Stage-2 binary LightGBM separates the two REMAINING classes (trained on rest-only
    rows, applied to all rows).
  * EACH stage uses class-balanced sample weights and BALANCED-ACCURACY-aware early
    stopping (custom feval; metric='None' so logloss never drives best_iteration).
  * Original SDSS17 rows are appended to each fold's TRAIN pool at low weight (0.1),
    NEVER to validation -> OOF stays honest.

Composition of the calibrated 3-class probabilities (cols = [GALAXY, QSO, STAR]):
    P(first_class) = p1
    P(a)           = (1 - p1) * p2          # a, b = the two remaining classes, a<b
    P(b)           = (1 - p1) * (1 - p2)
  then row-renormalized. p1 = stage-1 P(first_class); p2 = stage-2 P(a | rest).

Self-contained Kaggle SCRIPT kernel (CPU, fast: ~12 small LightGBMs). Does NOT import
the user's pipeline modules / kernel_common.

Alignment contract (so the OOF stacks with the existing pool):
  * Labels GALAXY=0, QSO=1, STAR=2.
  * Folds StratifiedKFold(5, shuffle=True, random_state=42) on the integer-label
    vector over the 577347 competition rows ONLY, in train-CSV row order.
  * Test rows reordered to sample_submission id order.

Outputs to /kaggle/working/:
  oof_hier.npy    (577347, 3) float32, columns [GALAXY,QSO,STAR], train-CSV row order
  test_hier.npy   (247435, 3) float32, columns [GALAXY,QSO,STAR], sample_submission order
  results.txt     probe AUCs, chosen split, per-stage AUC, composed OOF balanced
                  accuracy + per-class recall + confusion matrix
  submission.csv  argmax -> label, sample_submission order
"""

from __future__ import annotations

import os

# Keep CPU helper libs from each grabbing every thread.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import gc
import glob
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
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
GALAXY, QSO, STAR = 0, 1, 2

RAW_NUMS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
BANDS = ["u", "g", "r", "i", "z"]

N_TRAIN_EXPECT = 577347
N_TEST_EXPECT = 247435

# Hyper-params.
ORIGINAL_WEIGHT = 0.1     # appended original rows down-weighted in each TRAIN pool
LGB_ROUNDS = 3000
LGB_EARLY = 150
PROBE_ROUNDS = 900        # quick one-fold probe to choose the stage-1 split
PROBE_EARLY = 80
EPS = 1e-6                # clip stage probabilities away from {0,1} before composing
CLIP = 1e-15              # final save-array clip

# Codes for binned-color categoricals (reconstruct the competition cats).
SPEC_MAP = {"M": 0, "G/K": 1, "A/F": 2, "O/B": 3}
POP_MAP = {"Blue_Cloud": 0, "Red_Sequence": 1}

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
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
    candidates += [
        Path(p) for p in glob.glob("/kaggle/input/**/star_classification.csv", recursive=True)
    ]
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
# Feature engineering — numeric-only matrix.
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
# Metrics / save helpers
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


# ----------------------------------------------------------------------------
# Binary LightGBM with balanced-accuracy-aware early stopping
# ----------------------------------------------------------------------------
def _feval_bin_bal_acc(preds, dataset):
    # With a BUILT-IN objective (objective='binary'), LightGBM hands feval the
    # post-sigmoid probability of the positive class -> threshold at 0.5.
    y_true = dataset.get_label().astype(int)
    y_hat = (np.asarray(preds) >= 0.5).astype(int)
    return "bal_acc", float(balanced_accuracy_score(y_true, y_hat)), True


def _lgb_binary_params() -> dict:
    # metric='None' disables the default (binary_logloss) so the custom bal_acc feval
    # is the ONLY/FIRST metric and therefore drives early stopping. first_metric_only
    # guards the same intent.
    return dict(
        objective="binary",
        metric="None",
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.05,
        reg_lambda=8.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
        first_metric_only=True,
    )


def train_binary(X_fit, y_bin, sw, X_es, y_es_bin, rounds, early):
    """Fit a binary LightGBM; early-stop on the validation binary balanced accuracy."""
    import lightgbm as lgb

    dtrain = lgb.Dataset(X_fit, label=y_bin, weight=sw)
    dvalid = lgb.Dataset(X_es, label=y_es_bin, reference=dtrain)
    booster = lgb.train(
        _lgb_binary_params(),
        dtrain,
        num_boost_round=rounds,
        valid_sets=[dvalid],
        valid_names=["valid"],
        feval=_feval_bin_bal_acc,
        callbacks=[
            lgb.early_stopping(early, first_metric_only=True),
            lgb.log_evaluation(250),
        ],
    )
    bi = booster.best_iteration or rounds
    del dtrain, dvalid
    return booster, bi


# ----------------------------------------------------------------------------
# Cascade composition
# ----------------------------------------------------------------------------
def compose(p1: np.ndarray, p2: np.ndarray, first_class: int, a: int, b: int) -> np.ndarray:
    """Compose 3-class probs from stage-1 P(first_class) and stage-2 P(a|rest)."""
    p1 = np.clip(p1.astype("float32"), EPS, 1.0 - EPS)
    p2 = np.clip(p2.astype("float32"), EPS, 1.0 - EPS)
    out = np.zeros((len(p1), N_CLASSES), dtype="float32")
    out[:, first_class] = p1
    out[:, a] = (1.0 - p1) * p2
    out[:, b] = (1.0 - p1) * (1.0 - p2)
    out = out / out.sum(axis=1, keepdims=True)
    return out.astype("float32")


def rest_pair(first_class: int) -> tuple[int, int]:
    rest = [c for c in range(N_CLASSES) if c != first_class]
    return rest[0], rest[1]  # a<b


# ----------------------------------------------------------------------------
# Empirical stage-1 split probe (one fold): pick the cleaner peel-off class.
# ----------------------------------------------------------------------------
def probe_first_class(X_comp, y_comp, X_orig, y_orig, fold) -> int:
    tr_idx, va_idx = fold
    X_fit = np.vstack([X_comp[tr_idx], X_orig])
    y_fit = np.concatenate([y_comp[tr_idx], y_orig]).astype("int64")
    n_tr = len(tr_idx)
    is_orig = np.zeros(len(y_fit), dtype=bool)
    is_orig[n_tr:] = True
    X_va, y_va = X_comp[va_idx], y_comp[va_idx]

    aucs: dict[int, float] = {}
    for fc in (STAR, QSO):
        try:
            y_bin = (y_fit == fc).astype("int64")
            sw = compute_sample_weight(class_weight="balanced", y=y_bin).astype("float32")
            sw[is_orig] *= np.float32(ORIGINAL_WEIGHT)
            y_va_bin = (y_va == fc).astype("int64")
            booster, bi = train_binary(
                X_fit, y_bin, sw, X_va, y_va_bin, PROBE_ROUNDS, PROBE_EARLY
            )
            p = booster.predict(X_va, num_iteration=bi)
            auc = float(roc_auc_score(y_va_bin, p))
            aucs[fc] = auc
            log(f"probe stage-1 {INV_MAP[fc]}-vs-rest: AUC={auc:.6f} best_iter={bi}")
            del booster
            gc.collect()
        except Exception as e:  # pragma: no cover - be robust, fall back to default
            log(f"probe {INV_MAP[fc]}-vs-rest FAILED ({e!r})")
            aucs[fc] = -1.0

    emit(f"# probe AUCs: STAR-vs-rest={aucs.get(STAR, float('nan')):.6f} "
         f"QSO-vs-rest={aucs.get(QSO, float('nan')):.6f}")
    # Default to STAR-vs-rest unless QSO-vs-rest is clearly cleaner (> 1e-4 higher AUC).
    if aucs.get(QSO, -1.0) > aucs.get(STAR, -1.0) + 1e-4:
        chosen = QSO
    else:
        chosen = STAR
    emit(f"# chosen stage-1 split: {INV_MAP[chosen]}-vs-rest (default STAR on near tie)")
    flush_results()
    return chosen


# ----------------------------------------------------------------------------
# Full 5-fold cascade
# ----------------------------------------------------------------------------
def run_cascade(X_comp, y_comp, X_orig, y_orig, X_test, folds, first_class):
    a, b = rest_pair(first_class)
    log(f"cascade: stage-1 = {INV_MAP[first_class]}-vs-rest ; "
        f"stage-2 = {INV_MAP[a]}-vs-{INV_MAP[b]}")

    oof = np.zeros((len(y_comp), N_CLASSES), dtype="float32")
    test_pred = np.zeros((len(X_test), N_CLASSES), dtype="float32")
    fold_ba: list[float] = []
    s1_aucs: list[float] = []
    s2_aucs: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        t0 = time.perf_counter()
        X_fit = np.vstack([X_comp[tr_idx], X_orig])
        y_fit = np.concatenate([y_comp[tr_idx], y_orig]).astype("int64")
        n_tr = len(tr_idx)
        is_orig = np.zeros(len(y_fit), dtype=bool)
        is_orig[n_tr:] = True
        X_va, y_va = X_comp[va_idx], y_comp[va_idx]

        # ---- Stage 1: first_class vs rest (all fit rows) ----
        y1_fit = (y_fit == first_class).astype("int64")
        sw1 = compute_sample_weight(class_weight="balanced", y=y1_fit).astype("float32")
        sw1[is_orig] *= np.float32(ORIGINAL_WEIGHT)
        y1_va = (y_va == first_class).astype("int64")
        b1, bi1 = train_binary(X_fit, y1_fit, sw1, X_va, y1_va, LGB_ROUNDS, LGB_EARLY)
        p1_va = b1.predict(X_va, num_iteration=bi1).astype("float32")
        p1_te = b1.predict(X_test, num_iteration=bi1).astype("float32")
        auc1 = float(roc_auc_score(y1_va, p1_va))
        s1_aucs.append(auc1)

        # ---- Stage 2: a vs b, trained on REST-ONLY fit rows ----
        rest_fit = y_fit != first_class
        X2_fit = X_fit[rest_fit]
        y2_fit = (y_fit[rest_fit] == a).astype("int64")
        sw2 = compute_sample_weight(class_weight="balanced", y=y2_fit).astype("float32")
        sw2[is_orig[rest_fit]] *= np.float32(ORIGINAL_WEIGHT)
        rest_va = y_va != first_class
        X2_va_es = X_va[rest_va]
        y2_va_es = (y_va[rest_va] == a).astype("int64")
        b2, bi2 = train_binary(X2_fit, y2_fit, sw2, X2_va_es, y2_va_es, LGB_ROUNDS, LGB_EARLY)
        # Apply stage-2 to ALL rows (cascade weights small rest-prob for first_class rows).
        p2_va = b2.predict(X_va, num_iteration=bi2).astype("float32")
        p2_te = b2.predict(X_test, num_iteration=bi2).astype("float32")
        auc2 = float(roc_auc_score(y2_va_es, b2.predict(X2_va_es, num_iteration=bi2)))
        s2_aucs.append(auc2)

        # ---- Compose ----
        comp_va = compose(p1_va, p2_va, first_class, a, b)
        oof[va_idx] = comp_va
        test_pred += compose(p1_te, p2_te, first_class, a, b) / N_FOLDS

        ba = float(balanced_accuracy_score(y_va, comp_va.argmax(axis=1)))
        fold_ba.append(ba)
        log(f"fold {fold}: s1_auc={auc1:.5f} (it={bi1})  s2_auc={auc2:.5f} (it={bi2})  "
            f"composed BA={ba:.6f}  ({time.perf_counter() - t0:.0f}s)")

        del X_fit, y_fit, X2_fit, b1, b2, p1_va, p1_te, p2_va, p2_te, comp_va
        gc.collect()

    return oof, test_pred, fold_ba, s1_aucs, s2_aucs


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    emit("# s6e6 hiercascade : 2-stage binary-LightGBM cascade (CPU)")
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

    # Align test to sample_submission row order.
    if ID_COL in test.columns and ID_COL in sample.columns:
        test = test.set_index(ID_COL).loc[sample[ID_COL]].reset_index()
        log("test reordered to sample_submission id order")

    # Labels in train-CSV order (GALAXY=0, QSO=1, STAR=2).
    y_comp = train[TARGET].astype(str).map(LABEL_MAP).to_numpy()
    assert not np.any(np.isnan(y_comp.astype(float))), "unmapped class label in train"
    y_comp = y_comp.astype("int64")

    # Original dataset: reconstruct cats, filter to valid classes, drop sentinel rows.
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

    # Folds depend only on y_comp + n_splits + random_state => align with all artifacts.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y_comp)), y_comp))

    # ---- Empirical probe (one fold) to choose the stage-1 peel-off class ----
    log("probing stage-1 split (STAR-vs-rest vs QSO-vs-rest) on fold 1 ...")
    first_class = probe_first_class(X_comp, y_comp, X_orig, y_orig, folds[0])

    # ---- Full cascade ----
    oof, test_pred, fold_ba, s1_aucs, s2_aucs = run_cascade(
        X_comp, y_comp, X_orig, y_orig, X_test, folds, first_class
    )

    # ---- Save artifacts (do this before the report so they survive any late error) ----
    oof_n = normalize_proba(oof)
    test_n = normalize_proba(test_pred)
    np.save(WORK / "oof_hier.npy", oof_n.astype("float32"))
    np.save(WORK / "test_hier.npy", test_n.astype("float32"))
    sub = sample.copy()
    sub[TARGET] = [INV_MAP[i] for i in test_n.argmax(axis=1)]
    sub.to_csv(WORK / "submission.csv", index=False)
    log(f"saved oof_hier.npy {oof_n.shape}, test_hier.npy {test_n.shape}, submission.csv")

    # ---- Report ----
    a, b = rest_pair(first_class)
    y_pred = oof_n.argmax(axis=1)
    ba = float(balanced_accuracy_score(y_comp, y_pred))
    rec = per_class_recall(y_comp, y_pred)
    cm = confusion_matrix(y_comp, y_pred, labels=list(range(N_CLASSES)))

    emit("")
    emit("===== hier (2-stage cascade) =====")
    emit(f"  stage-1 split: {INV_MAP[first_class]}-vs-rest")
    emit(f"  stage-2 split: {INV_MAP[a]}-vs-{INV_MAP[b]}")
    emit(f"  stage-1 per-fold AUC: {[round(x, 6) for x in s1_aucs]}  mean={np.mean(s1_aucs):.6f}")
    emit(f"  stage-2 per-fold AUC: {[round(x, 6) for x in s2_aucs]}  mean={np.mean(s2_aucs):.6f}")
    emit(f"  composed per-fold BA: {[round(x, 6) for x in fold_ba]}  mean={np.mean(fold_ba):.6f}")
    emit(f"  OVERALL OOF balanced accuracy: {ba:.6f}")
    emit("  per-class recall: "
         f"GALAXY={rec['GALAXY']:.4f} QSO={rec['QSO']:.4f} STAR={rec['STAR']:.4f}")
    emit("  confusion (rows=true GALAXY/QSO/STAR, cols=pred):")
    for i, name in enumerate(CLASSES):
        emit(f"    {name:7s} {cm[i].tolist()}")
    emit("DONE")
    flush_results()
    log("ALL DONE")


if __name__ == "__main__":
    main()
