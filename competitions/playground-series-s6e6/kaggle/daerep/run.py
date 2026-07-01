"""s6e6 daerep : swap-noise Denoising AutoEncoder (DAE) representation -> LightGBM.

The Porto-Seguro / Michael-Jahrer base-ceiling lever, ported to S6E6. A swap-noise DAE
is trained UNSUPERVISED (no labels) on the union of competition train + competition test +
ORIGINAL SDSS17 rows. Its learned representation (last encoder hidden layer + bottleneck) is
extracted and CONCATENATED with the raw all_v2 features, then a stack-aligned 5-fold LightGBM
is trained on top (original-data augmented TRAIN pool, balanced-accuracy early stopping).

Why the OOF stays honest:
  * The DAE is fit ONCE, transductively, on ALL rows but uses NO target. It is a pure
    feature transform of X. Computing DAE features for a validation row therefore leaks no
    label information (no y ever touches the DAE), so the LightGBM 5-fold OOF is honest.
  * The LightGBM itself respects the stack-alignment contract: OOF only on competition train
    rows, original rows only in each fold's TRAIN pool (never validation).

Self-contained Kaggle SCRIPT kernel. Does NOT import the user's pipeline modules.

P100 reality: stock Kaggle torch (2.10+cu128) has NO sm_60 kernels, so we install
torch==2.4.1+cu121 (covers sm_60 P100 AND sm_75 T4) BEFORE importing torch, and use the
INDEXED device 'cuda:0' (bare 'cuda' breaks torch 2.4 mem_get_info on some libs).

Contract (must match existing artifacts so the new OOF stacks):
  * Labels: GALAXY=0, QSO=1, STAR=2 (alphabetical).
  * Folds: StratifiedKFold(5, shuffle=True, random_state=42).split(X_comp, y_comp) on the
    577347 competition rows ONLY, in original train-CSV row order.

Outputs to /kaggle/working/:
  oof_dae.npy   (577347, 3) float32, columns [GALAXY,QSO,STAR], train-CSV row order
  test_dae.npy  (247435, 3) float32, columns [GALAXY,QSO,STAR], sample_submission order
  results.txt   DAE recon loss + per-fold + overall OOF balanced accuracy + per-class recall
  submission.csv argmax->label
"""

from __future__ import annotations

import os

# Keep CPU helper libs from grabbing every thread while LightGBM trains.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
# Reduce CUDA fragmentation; must be set before importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import glob
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# --- Pascal-compatible torch (install BEFORE `import torch`) ---
# Kaggle's stock torch 2.10+cu128 dropped sm_60; this batch kernel may land on a P100.
# cu121 build supports sm_60 (P100) AND sm_75 (T4). enable_internet:true is required.
import subprocess as _sp
import sys as _sys

_sp.run(
    [_sys.executable, "-m", "pip", "install", "-q", "torch==2.4.1",
     "--extra-index-url", "https://download.pytorch.org/whl/cu121"],
    check=False,
)

import torch
import torch.nn as nn

from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
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

# Original-data augmentation.
ORIGINAL_WEIGHT = 0.1          # appended original rows down-weighted in TRAIN pool
CLIP = 1e-15

# DAE hyper-parameters (swap-noise denoising autoencoder).
SWAP_P = 0.15                  # per-element input-swap corruption probability
ENC_HIDDEN = (256, 128)        # encoder hidden sizes -> 3-layer encoder (h1,h2,bottleneck)
BOTTLENECK = 96                # bottleneck dimension (~64-128)
DAE_EPOCHS = 60                # modest; small MLP on ~0.9M rows finishes in minutes
DAE_BATCH = 4096
DAE_LR = 1e-3
DAE_WD = 1e-5
DAE_EXTRACT_BATCH = 16384      # eval batch for feature extraction
# Extracted DAE features = [last encoder hidden (ENC_HIDDEN[-1]), bottleneck (BOTTLENECK)].

# LightGBM (gbdt_orig spirit; CPU — runs alongside the GPU DAE).
LGB_ROUNDS = 4000
LGB_EARLY = 200

SPEC_MAP = {"M": 0, "G/K": 1, "A/F": 2, "O/B": 3}
POP_MAP = {"Blue_Cloud": 0, "Red_Sequence": 1}

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = WORK / "results.txt"

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


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
            "/kaggle/input/datasets/cindyxue1122/s6e6-original-sdss17/star_classification.csv"
        ),
        Path(
            "/kaggle/input/datasets/fedesoriano/"
            "stellar-classification-dataset-sdss17/star_classification.csv"
        ),
    ]
    candidates += [Path(p) for p in glob.glob("/kaggle/input/**/star_classification.csv", recursive=True)]
    candidates += [Path(p) for p in glob.glob("/kaggle/input/**/stellar_classification.csv", recursive=True)]
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
# Feature engineering (all_v2 colors+redshift). Numeric-only matrix shared by the
# DAE (after StandardScaling) and by LightGBM (raw). Applied identically to comp
# train, test, and original rows.
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


# ----------------------------------------------------------------------------
# Swap-noise Denoising AutoEncoder
# ----------------------------------------------------------------------------
class DAE(nn.Module):
    """3-layer MLP encoder -> bottleneck -> mirrored decoder. ReLU activations.

    encode() returns the activations used as new features:
      (last encoder hidden [ENC_HIDDEN[-1]], bottleneck [BOTTLENECK]).
    """

    def __init__(self, d_in: int, enc_hidden=ENC_HIDDEN, bottleneck=BOTTLENECK):
        super().__init__()
        h1, h2 = enc_hidden
        self.enc1 = nn.Linear(d_in, h1)
        self.enc2 = nn.Linear(h1, h2)
        self.enc3 = nn.Linear(h2, bottleneck)   # bottleneck
        self.dec1 = nn.Linear(bottleneck, h2)
        self.dec2 = nn.Linear(h2, h1)
        self.dec3 = nn.Linear(h1, d_in)         # linear reconstruction (standardized inputs)
        self.act = nn.ReLU()

    def encode(self, x):
        h1 = self.act(self.enc1(x))
        h2 = self.act(self.enc2(h1))
        z = self.act(self.enc3(h2))
        return h2, z

    def forward(self, x):
        h2, z = self.encode(x)
        d = self.act(self.dec1(z))
        d = self.act(self.dec2(d))
        return self.dec3(d)


def make_swap_noise(xb: torch.Tensor, x_full: torch.Tensor, swap_p: float) -> torch.Tensor:
    """Per-element swap noise: with prob swap_p, replace x[i,j] with x_full[k,j] for a random
    row k drawn from the FULL (unlabeled) dataset's empirical marginal of column j."""
    b, d = xb.shape
    n = x_full.shape[0]
    rand_rows = torch.randint(0, n, (b, d), device=xb.device)
    col_idx = torch.arange(d, device=xb.device).unsqueeze(0).expand(b, d)
    swap_vals = x_full[rand_rows, col_idx]
    mask = torch.rand(b, d, device=xb.device) < swap_p
    return torch.where(mask, swap_vals, xb)


def train_dae(x_full: torch.Tensor, d_in: int, device) -> tuple[DAE, float]:
    """Train the swap-noise DAE unsupervised on all rows. Returns (model, final_recon_loss)."""
    seed_everything(SEED)
    model = DAE(d_in).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=DAE_LR, weight_decay=DAE_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=DAE_EPOCHS, eta_min=DAE_LR * 0.02)
    loss_fn = nn.MSELoss()
    n = x_full.shape[0]
    final_loss = float("nan")

    for epoch in range(1, DAE_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        running, n_batches = 0.0, 0
        for start in range(0, n, DAE_BATCH):
            idx = perm[start:start + DAE_BATCH]
            xb = x_full[idx]
            xc = make_swap_noise(xb, x_full, SWAP_P)
            recon = model(xc)
            loss = loss_fn(recon, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            n_batches += 1
        sched.step()
        final_loss = running / max(1, n_batches)
        if epoch == 1 or epoch % 5 == 0 or epoch == DAE_EPOCHS:
            log(f"DAE epoch {epoch:03d}/{DAE_EPOCHS}  recon_mse={final_loss:.6f}  "
                f"lr={sched.get_last_lr()[0]:.2e}")
    return model, final_loss


@torch.no_grad()
def extract_dae_features(model: DAE, x_full: torch.Tensor) -> np.ndarray:
    """Run the encoder on CLEAN (uncorrupted) inputs; concat [last hidden, bottleneck]."""
    model.eval()
    n = x_full.shape[0]
    chunks = []
    for start in range(0, n, DAE_EXTRACT_BATCH):
        xb = x_full[start:start + DAE_EXTRACT_BATCH]
        h2, z = model.encode(xb)
        feat = torch.cat([h2, z], dim=1)
        chunks.append(feat.float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype("float32")


# ----------------------------------------------------------------------------
# LightGBM on [raw all_v2 features | DAE features], original-augmented TRAIN.
# ----------------------------------------------------------------------------
def make_fit_data(X_comp, y_comp, X_orig, y_orig, tr_idx):
    """Fold fit matrix = competition-train rows + ALL original rows, balanced weights,
    original rows further down-weighted by ORIGINAL_WEIGHT."""
    X_fit = np.vstack([X_comp[tr_idx], X_orig])
    y_fit = np.concatenate([y_comp[tr_idx], y_orig]).astype("int64")
    sw = compute_sample_weight(class_weight="balanced", y=y_fit).astype("float32")
    sw[len(tr_idx):] *= np.float32(ORIGINAL_WEIGHT)
    return X_fit, y_fit, sw


def train_lgb(X_comp, y_comp, X_orig, y_orig, X_test, folds):
    import lightgbm as lgb

    def feval_bal_acc(y_pred_raw, dataset):
        # lgb multiclass raw preds: modern lgb (>=4) hands back a 2-D (n_rows, n_classes)
        # array; older lgb flattens column-major. Handle both.
        y_true = dataset.get_label().astype(int)
        nrows = y_true.shape[0]
        arr = np.asarray(y_pred_raw)
        preds = arr if arr.ndim == 2 else arr.reshape(N_CLASSES, nrows).T
        y_hat = preds.argmax(axis=1)
        return "bal_acc", balanced_accuracy_score(y_true, y_hat), True

    # metric="None" disables the built-in multi_logloss so the custom bal_acc feval is the
    # ONLY / FIRST metric and therefore drives early stopping (NOT logloss).
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


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    emit("# s6e6 daerep : swap-noise DAE representation -> LightGBM")
    emit(f"# seed={SEED} folds={N_FOLDS} original_weight={ORIGINAL_WEIGHT}")
    emit(f"# DAE: swap_p={SWAP_P} enc_hidden={ENC_HIDDEN} bottleneck={BOTTLENECK} "
         f"epochs={DAE_EPOCHS} batch={DAE_BATCH}")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")        # INDEXED device for torch 2.4 on P100
        props = torch.cuda.get_device_properties(0)
        log(f"GPU: {props.name} cap=sm_{props.major}{props.minor} "
            f"mem={props.total_memory / 1024 ** 3:.1f}GiB")
    else:
        device = torch.device("cpu")
        log("WARNING: CUDA not available; DAE will train on CPU (slow).")
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

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

    # Original dataset: reconstruct cats, filter to valid classes + sane magnitudes.
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

    # ---- Features (identical pipeline for all three frames) ----
    log("building all_v2 features ...")
    Xc_df = add_features(train)
    Xt_df = add_features(test)
    Xo_df = add_features(orig)
    feature_names = list(Xc_df.columns)
    assert list(Xt_df.columns) == feature_names and list(Xo_df.columns) == feature_names

    X_comp_raw = Xc_df.fillna(0.0).to_numpy(dtype="float32")
    X_test_raw = Xt_df.fillna(0.0).to_numpy(dtype="float32")
    X_orig_raw = Xo_df.fillna(0.0).to_numpy(dtype="float32")
    n_comp, n_test, n_orig = len(X_comp_raw), len(X_test_raw), len(X_orig_raw)
    d_in = X_comp_raw.shape[1]
    log(f"raw feature matrix: comp={X_comp_raw.shape} test={X_test_raw.shape} "
        f"orig={X_orig_raw.shape} d_in={d_in}")
    emit(f"# n_raw_features={d_in} n_orig_rows={n_orig}")
    del Xc_df, Xt_df, Xo_df, orig, orig_raw, train, test
    gc.collect()

    # ---- StandardScale on ALL rows (transductive, target-free) for the DAE ----
    # The DAE is a pure X-transform (no labels) -> validation DAE features leak no target,
    # so the downstream LightGBM 5-fold OOF stays honest.
    X_all_raw = np.vstack([X_comp_raw, X_test_raw, X_orig_raw])  # order: comp, test, orig
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all_raw).astype("float32")
    X_all_scaled = np.nan_to_num(X_all_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    del X_all_raw
    gc.collect()
    log(f"scaled DAE input matrix: {X_all_scaled.shape}")

    # ---- Train the swap-noise DAE (unsupervised, all rows) ----
    x_full = torch.from_numpy(X_all_scaled).to(device)  # ~0.2GB on GPU for 0.9M x ~54
    log("training swap-noise DAE (unsupervised on train+test+original) ...")
    dae_model, dae_recon = train_dae(x_full, d_in, device)
    emit(f"# DAE final reconstruction MSE = {dae_recon:.6f}")

    # ---- Extract DAE representation for all rows, then split back ----
    log("extracting DAE features ...")
    dae_all = extract_dae_features(dae_model, x_full)   # (n_all, h2+bottleneck)
    dae_dim = dae_all.shape[1]
    del x_full, dae_model, X_all_scaled
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dae_comp = dae_all[:n_comp]
    dae_test = dae_all[n_comp:n_comp + n_test]
    dae_orig = dae_all[n_comp + n_test:]
    del dae_all
    gc.collect()
    assert len(dae_comp) == n_comp and len(dae_test) == n_test and len(dae_orig) == n_orig
    log(f"DAE feature dim={dae_dim} "
        f"(last_hidden={ENC_HIDDEN[-1]} + bottleneck={BOTTLENECK})")
    emit(f"# n_dae_features={dae_dim} total_lgb_features={d_in + dae_dim}")

    # ---- Concatenate [raw all_v2 | DAE features] ----
    X_comp = np.hstack([X_comp_raw, dae_comp]).astype("float32")
    X_test = np.hstack([X_test_raw, dae_test]).astype("float32")
    X_orig = np.hstack([X_orig_raw, dae_orig]).astype("float32")
    del X_comp_raw, X_test_raw, X_orig_raw, dae_comp, dae_test, dae_orig
    gc.collect()
    log(f"LGB feature matrix: comp={X_comp.shape} test={X_test.shape} orig={X_orig.shape}")

    # ---- Folds: depend only on y_comp + n_splits + random_state => align with all artifacts.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X_comp, y_comp))

    # ---- LightGBM ----
    log("training LightGBM on [raw all_v2 | DAE] features ...")
    oof, test_pred, fs = train_lgb(X_comp, y_comp, X_orig, y_orig, X_test, folds)
    ba = report_oof("dae (LightGBM on raw+DAE)", y_comp, oof, fs)

    # ---- Save artifacts ----
    oof = normalize_proba(oof)
    test_pred = normalize_proba(test_pred)
    np.save(WORK / "oof_dae.npy", oof.astype("float32"))
    np.save(WORK / "test_dae.npy", test_pred.astype("float32"))
    sub = sample.copy()
    sub[TARGET] = [INV_MAP[i] for i in test_pred.argmax(axis=1)]
    sub.to_csv(WORK / "submission.csv", index=False)
    log(f"saved oof_dae.npy {oof.shape}, test_dae.npy {test_pred.shape}, submission.csv")

    emit("")
    emit("==================== SUMMARY ====================")
    emit(f"  DAE reconstruction MSE = {dae_recon:.6f}")
    emit(f"  dae OOF balanced accuracy = {ba:.6f}")
    emit("DONE")
    flush_results()
    log("ALL DONE")


if __name__ == "__main__":
    main()
