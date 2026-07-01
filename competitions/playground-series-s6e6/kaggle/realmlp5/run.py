"""s6e6-realmlp5 — Kaggle SCRIPT kernel.

Faithful port of the notebook `realmlp-v5-for-s6e6.ipynb` (config R2-103, reported 5-fold
local balanced accuracy ~0.969278). A single custom RealMLP-style PyTorch tabular network
(n_ens=8 in-model ensemble); no stacking / blending / post-processing.

Self-contained: pure torch + sklearn, no pip installs, no user pipeline imports.

Outputs to /kaggle/working:
  oof_realmlp5.npy   (577347, 3) float32  OOF probabilities, train-CSV row order, cols [GALAXY,QSO,STAR]
  test_realmlp5.npy  (247435, 3) float32  test probabilities, sample_submission order, mean of 5 folds
  submission.csv     argmax -> label
  results.txt        per-fold + overall balanced accuracy + per-class recall

LABEL MAP (matches existing artifacts): GALAXY=0, QSO=1, STAR=2 (alphabetical).
FOLDS: StratifiedKFold(5, shuffle=True, random_state=42).split(X, y), y = int labels in CSV order.
"""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import gc
import math
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# --- Pascal-compatible torch ---
# Kaggle's stock torch 2.10+cu128 dropped sm_60; this kernel may land on a P100.
# Install a cu121 build (supports sm_60 P100 AND sm_75 T4) BEFORE importing torch.
import sys as _sys, subprocess as _sp
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "torch==2.4.1",
         "--extra-index-url", "https://download.pytorch.org/whl/cu121"], check=False)

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)

T0 = time.perf_counter()
def log(msg):
    print(f'[{time.perf_counter() - T0:8.1f}s] {msg}', flush=True)

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Config (identifiers / paths)
# ---------------------------------------------------------------------------
MODEL_ID = 'realmlpw'   # second-seed variant: same folds (skf random_state=42), different model seeds
SEED = 42
FOLDS = 5
N_CLASSES = 3
TARGET = 'class'
ID = 'id'

CLASSES = ['GALAXY', 'QSO', 'STAR']
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}   # GALAXY=0, QSO=1, STAR=2
INV_MAP = {v: k for k, v in LABEL_MAP.items()}
CLASS_MAP = LABEL_MAP
INV_CLASS_MAP = INV_MAP

WORK = Path('/kaggle/working')
OOF_PATH = WORK / f'oof_{MODEL_ID}.npy'
PRED_PATH = WORK / f'test_{MODEL_ID}.npy'
SUB_PATH = WORK / 'submission.csv'
RESULTS_PATH = WORK / 'results.txt'

seed_everything(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)
print('Visible CUDA devices:', os.environ.get('CUDA_VISIBLE_DEVICES'))


# ---------------------------------------------------------------------------
# Load Data (competition-dir fallback)
# ---------------------------------------------------------------------------
def find_data_dir():
    candidates = [
        Path('/kaggle/input/competitions/playground-series-s6e6'),
        Path('/kaggle/input/playground-series-s6e6'),
    ]
    for p in candidates:
        if (p / 'train.csv').exists() and (p / 'test.csv').exists() and (p / 'sample_submission.csv').exists():
            return p
    raise FileNotFoundError('Could not find train.csv, test.csv, and sample_submission.csv')

DATA_DIR = find_data_dir()
log(f'DATA_DIR={DATA_DIR}')

train = pd.read_csv(DATA_DIR / 'train.csv', index_col=ID)
test = pd.read_csv(DATA_DIR / 'test.csv', index_col=ID)
sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')

train[TARGET] = train[TARGET].map(LABEL_MAP).astype('int8')
y = train[TARGET].astype(int)
X = train.drop([TARGET], axis=1)
X_test = test.copy()

log(f'X={X.shape} X_test={X_test.shape}')
print(pd.Series(y).map(INV_MAP).value_counts())


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
base_cat_cols = X.select_dtypes(include=['object']).columns.tolist()
base_num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
print('Base categorical columns:', base_cat_cols)
print('Base numeric columns:', base_num_cols)

cat_cols = base_cat_cols.copy()
num_cols = base_num_cols.copy()
category_map = {}

color_pairs = [
    ('u', 'g'),
    ('g', 'r'),
    ('r', 'i'),
    ('i', 'z'),
    ('u', 'r'),
    ('g', 'i'),
    ('r', 'z'),
]
important_combos = [
    ('alpha_cat_', 'delta_cat_'),
    ('u_cat_', 'z_cat_'),
]

def feature_engineering(df, fit=False):
    df = df.copy()

    # Smooth numeric/color-locus features.
    df['_g_div_redshift'] = (df['g'] / (df['redshift'] + 1e-6)).replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')
    df['_i_div_redshift'] = (df['i'] / (df['redshift'] + 1e-6)).replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')
    for a, b in color_pairs:
        df[f'_{a}-{b}'] = (df[a] - df[b]).astype('float32')

    mags = df[['u', 'g', 'r', 'i', 'z']].astype('float32')
    df['_mag_mean'] = mags.mean(axis=1).astype('float32')
    df['_mag_range'] = (mags.max(axis=1) - mags.min(axis=1)).astype('float32')

    shifted_redshift = df['redshift'].astype('float32') - min(0.0, float(df['redshift'].min())) + 1e-4
    df['_log1p_redshift'] = np.log1p(shifted_redshift).astype('float32')

    # Original categorical columns.
    for col in base_cat_cols:
        if fit:
            codes, uniques = pd.factorize(df[col], sort=False)
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = df[col].map(code_map).fillna(-1).astype('int32')
        df[col] = pd.Series(codes, index=df.index).astype('int32').astype('category')

    # Integer-floor categorical views of every base numeric feature.
    for col in base_num_cols:
        cat_name = f'{col}_cat_'
        floored = np.floor(df[col]).astype('float32')
        if fit:
            codes, uniques = pd.factorize(floored, sort=False)
            category_map[cat_name] = uniques
        else:
            uniques = category_map[cat_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = floored.map(code_map).fillna(-1).astype('int32')
        df[cat_name] = pd.Series(codes, index=df.index).astype('int32').astype('category')

    # Delta quantile bins from the best local run.
    for n_bins in [100, 500]:
        bin_name = f'delta_{n_bins}_quantile_bin_'
        if fit:
            kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)
            binned = kb.fit_transform(df[['delta']]).ravel().astype('int32')
            category_map[bin_name] = kb
        else:
            kb = category_map[bin_name]
            binned = kb.transform(df[['delta']]).ravel().astype('int32')
        df[bin_name] = pd.Series(binned, index=df.index).astype('int32').astype('category')

    # Interaction categories used only for fold-safe multiclass target encoding.
    combo_names = []
    for cols in important_combos:
        combo_name = '__'.join(cols) + '__'
        combo_names.append(combo_name)
        combo = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo = combo + '|' + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo, sort=False)
            category_map[combo_name] = uniques
        else:
            uniques = category_map[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo.map(code_map).fillna(-1).astype('int32')
        df[combo_name] = pd.Series(codes, index=df.index).astype('int32').astype('category')

    new_cat_cols = [c for c in df.columns if str(df[c].dtype) == 'category' and c not in base_cat_cols]
    new_num_cols = [c for c in df.columns if c.startswith('_') and str(df[c].dtype) != 'category']
    return df, new_cat_cols, new_num_cols, combo_names

X, new_cat_cols, new_num_cols, combo_names = feature_engineering(X, fit=True)
X_test, _, _, _ = feature_engineering(X_test, fit=False)

cat_cols = sorted(base_cat_cols + new_cat_cols)
num_cols = sorted(base_num_cols + new_num_cols)
X = X.reindex(sorted(X.columns), axis=1)
X_test = X_test.reindex(sorted(X_test.columns), axis=1)

print('New categorical columns:', len(new_cat_cols))
print('New numeric columns:', len(new_num_cols))
print('Total categorical columns:', len(cat_cols))
print('Base feature shape:', X.shape)
print('Target-encoded interaction columns:', combo_names)


# ---------------------------------------------------------------------------
# RealMLP Modules (verbatim from notebook)
# ---------------------------------------------------------------------------
#  Preprocessing
class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    """
    Applies a configurable sequence of numerical transforms from CONFIG["tfms"].
    Supported: 'median_center', 'robust_scale', 'smooth_clip', 'l2_normalize'.
    'one_hot' and 'embedding' are recognised but skipped (handled by the model).
    """
    def __init__(self, tfms):
        self._tfms = [t for t in tfms
                      if t in ("median_center", "robust_scale", "smooth_clip", "l2_normalize")]

    def fit(self, X: np.ndarray, y=None):
        if "median_center" in self._tfms or "robust_scale" in self._tfms:
            self._median = np.median(X, axis=0)
            q_diff = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
            zero_idx = q_diff == 0.0
            q_diff[zero_idx] = 0.5 * (X.max(axis=0)[zero_idx] - X.min(axis=0)[zero_idx])
            self._iqr_factors = 1.0 / (q_diff + 1e-30)
            self._iqr_factors[q_diff == 0.0] = 0.0
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X = X.copy().astype(np.float32)
        for tfm in self._tfms:
            if tfm == "median_center":
                X -= self._median[None, :]
            elif tfm == "robust_scale":
                X *= self._iqr_factors[None, :]
            elif tfm == "smooth_clip":
                X = X / np.sqrt(1 + (X / 3) ** 2)
            elif tfm == "l2_normalize":
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                X /= np.where(norms == 0, 1.0, norms)
        return X

#  Model components
class CategoricalFeatureLayer(nn.Module):
    def __init__(self, n_ens: int, cat_dims, embed_dim: int = 8,
                 onehot_thresh: int = 8, device=None):
        super().__init__()
        self.n_ens = n_ens
        self.cat_dims = cat_dims
        self.onehot_features = []
        self.embed_layers = nn.ModuleList()
        self._embed_feature_indices = []

        for i, dim in enumerate(cat_dims):
            if dim <= onehot_thresh:
                self.onehot_features.append(i)
            else:
                emb = nn.ModuleList(
                    [nn.Embedding(dim, embed_dim) for _ in range(n_ens)]
                )
                self.embed_layers.append(emb)
                self._embed_feature_indices.append(i)

    def forward(self, x):
        # x: (batch, n_ens, n_cat)
        batch_size, n_ens, _ = x.shape
        features = []

        if self.onehot_features:
            onehot_x    = x[:, :, self.onehot_features]
            onehot_dims = [self.cat_dims[i] for i in self.onehot_features]
            total_oh    = sum(onehot_dims)
            encoded     = torch.zeros(batch_size, n_ens, total_oh, device=x.device)
            start = 0
            for idx, dim in enumerate(onehot_dims):
                pos = onehot_x[:, :, idx : idx + 1].long()
                encoded.scatter_(2, pos + start, 1.0)
                start += dim
            features.append(encoded)

        for emb_list, feat_idx in zip(self.embed_layers, self._embed_feature_indices):
            feat_embs = []
            for model_idx in range(self.n_ens):
                indices = x[:, model_idx, feat_idx : feat_idx + 1].long()
                feat_embs.append(emb_list[model_idx](indices))    # (batch, 1, embed_dim)
            feat_combined = torch.cat(feat_embs, dim=1)           # (batch, n_ens, embed_dim)
            features.append(feat_combined)

        return torch.cat(features, dim=2)


class ScalingLayer(nn.Module):
    def __init__(self, n_ens: int, n_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))

    def forward(self, x):
        return x * self.scale[None, :, :]


class NTPLinear(nn.Module):
    def __init__(self, n_ens: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias   = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None

    def forward(self, x):
        # x: (batch, n_ens, in_features)
        # Single einsum replaces transpose  matmul  transpose
        x = torch.einsum("bki,kio->bko", x, self.weight) / math.sqrt(self.in_features)
        if self.bias is not None:
            x = x + self.bias
        return x


class PBLDEmbedding(nn.Module):
    """Periodic Basis with Learned Decay embedding for numerical features."""
    def __init__(self, n_ens: int, n_features: int,
                 hidden_dim: int = 16, out_dim: int = 4, freq_scale: float = 0.1,
                 activation=nn.GELU):
        super().__init__()
        self.n_ens      = n_ens
        self.n_features = n_features
        self.out_dim    = out_dim
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) / math.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(n_ens, n_features, out_dim - 1))
        self.act = activation()
        nn.init.uniform_(self.b1, -math.pi, math.pi)

    def forward(self, x):
        # x: (batch, n_ens, n_features)
        # All operations are fully vectorised over n_features  no Python loop.

        # (batch, n_ens, n_features, 1) * (n_ens, n_features, hidden)
        #  periodic: (batch, n_ens, n_features, hidden)
        periodic = torch.cos(
            2 * math.pi * (
                x.unsqueeze(-1) * self.w1.unsqueeze(0)   # Broadcast over batch
                + self.b1.unsqueeze(0)
            )
        )

        # (batch, n_ens, n_features, hidden) @ (n_ens, n_features, hidden, out-1)
        #  transformed: (batch, n_ens, n_features, out-1)
        transformed = self.act(
            torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2)
            + self.b2.unsqueeze(0)
        )

        # Concatenate raw feature (residual) with transformed output
        # x: (batch, n_ens, n_features)  unsqueeze  (batch, n_ens, n_features, 1)
        feat = torch.cat([x.unsqueeze(-1), transformed], dim=-1)
        # (batch, n_ens, n_features, out_dim)  flatten last two dims
        #  (batch, n_ens, n_features * out_dim)
        return feat.flatten(start_dim=2)

#  Model
class RealMLP(nn.Module):
    def __init__(self, output_dim: int, cat_dims, n_numerical: int, cfg: dict):
        super().__init__()
        n_ens      = cfg["n_ens"]
        embed_dim  = cfg["embed_dim"]
        self.n_ens = n_ens

        self.cate = CategoricalFeatureLayer(
            n_ens=n_ens, cat_dims=cat_dims, embed_dim=embed_dim,
            onehot_thresh=cfg["onehot_thresh"],
        )
        self.num_embed = PBLDEmbedding(
            n_ens=n_ens,
            n_features=n_numerical,
            hidden_dim=cfg["pbld_hidden_dim"],
            out_dim=cfg["pbld_out_dim"],
            freq_scale=cfg["pbld_freq_scale"],
            activation=cfg["pbld_activation"],
        )

        num_emb_dim = n_numerical * cfg["pbld_out_dim"]
        cat_emb_dim = sum(
            c if c <= cfg["onehot_thresh"] else embed_dim for c in cat_dims
        )
        total_dim = num_emb_dim + cat_emb_dim
        hidden_dims = cfg["hidden_dims"]

        act = cfg["activation"]

        # Build layers, tracking which NTPLinear is the "first layer"
        # so we can give it a separate lr group.
        # Each hidden position gets its own Dropout instance (shared instance
        # would only register once in nn.Sequential and break the scheduler).
        layers = []
        if cfg["add_front_scale"]:
            layers.append(ScalingLayer(n_ens=n_ens, n_features=total_dim))

        self._dropout_modules = []    # Kept for live p_drop_sched updates
        in_dim = total_dim
        for i, out_dim_h in enumerate(hidden_dims):
            linear = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=out_dim_h)
            if i == 0:
                self.first_linear = linear    # Reference for separate lr group
            drop = nn.Dropout(cfg["dropout"])
            self._dropout_modules.append(drop)
            layers += [linear, act(), drop]
            in_dim = out_dim_h

        self.hidden = nn.Sequential(*layers)
        self.output_layer = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=output_dim)

    def forward(self, x_num, x_cat):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_num = self.num_embed(x_num)
        x_cat = self.cate(x_cat)
        combined = torch.cat([x_num, x_cat], dim=2)
        x = self.hidden(combined)
        x = self.output_layer(x)
        return F.softmax(x, dim=2)    # (batch, n_ens, output_dim)

#  Schedule helpers
def apply_schedule(init_value: float, progress: float, sched: str,
                   flat_ratio: float = 0.3) -> float:
    """
    Supported schedules:
      'constant'     no decay
      'cos'          cosine from init to 0
      'flat_cos'     flat for flat_ratio, then cosine to 0
      'flat_anneal'  flat for flat_ratio, then linear to 0
      'sqrt_cos'     sqrt of cosine annealing (slower decay)
      'expm4t'       exponential decay: init * exp(-4 * progress)
    """
    if sched == "constant":
        return init_value
    elif sched == "cos":
        return init_value * (math.cos(math.pi * progress) + 1) / 2
    elif sched == "flat_cos":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (math.cos(math.pi * t) + 1) / 2
    elif sched == "flat_anneal":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (1 - t)
    elif sched == "sqrt_cos":
        return init_value * math.sqrt((math.cos(math.pi * progress) + 1) / 2)
    elif sched == "expm4t":
        return init_value * math.exp(-4 * progress)
    else:
        raise ValueError(f"Unknown schedule: '{sched}'")

#  Per-parameter-group builder
def get_parameter_groups(model: RealMLP, p: dict):
    """
    Five groups with independent lr / wd:
      0  ScalingLayer params  (scale.*)
      1  PBLD / num_embed params
      2  first hidden linear weight
      3  all other weights
      4  all biases (excluding those already in groups 0-2)
    Note: PBLD has its own bias params (b1, b2) which belong to group 1, not 4.
    The ordering of checks therefore is: scale  num_embed  first_w  bias  other_w.
    """
    first_linear_weight_id = id(model.first_linear.weight)

    scale_p, pbld_p, first_w_p, other_w_p, bias_p = [], [], [], [], []
    for name, param in model.named_parameters():
        if "num_embed" in name:
            # All PBLD params (weights and biases) get their own lr group
            pbld_p.append(param)
        elif "scale" in name:
            scale_p.append(param)
        elif id(param) == first_linear_weight_id:
            first_w_p.append(param)
        elif "bias" in name:
            bias_p.append(param)
        else:
            other_w_p.append(param)

    LR = p["lr"]
    WD = p["weight_decay"]
    return [
        {"params": scale_p,   "lr": LR * p["lr_scale_mult"],         "weight_decay":  WD * p["wd_scale_mult"],         "group": "scale"},
        {"params": pbld_p,    "lr": LR * p["pbld_lr_factor"],        "weight_decay":  WD,                              "group": "pbld"},
        {"params": first_w_p, "lr": LR * p["first_layer_lr_factor"], "weight_decay":  WD * p["first_layer_wd_factor"], "group": "first_w"},
        {"params": other_w_p, "lr": LR,                              "weight_decay":  WD,                              "group": "other_w"},
        {"params": bias_p,    "lr": LR * p["lr_bias_mult"],          "weight_decay":  WD * p["wd_bias_mult"],          "group": "bias"},
    ]

#  Multiclass label-smoothed cross-entropy with optional class weights
def smooth_ce_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    ls: float = 0.0,
    class_weights: torch.Tensor = None,
    focal_gamma: float = 0.0,
    loss_prob_multipliers: torch.Tensor = None,
) -> torch.Tensor:
    """
    y_true        : (N,)    long
    y_pred        : (N, C)  probabilities
    class_weights : (C,)    per-class weight tensor (optional)
    """
    n_classes = y_pred.size(1)
    if loss_prob_multipliers is not None:
        y_pred = y_pred * loss_prob_multipliers[None, :]
        y_pred = y_pred / y_pred.sum(dim=1, keepdim=True).clamp_min(1e-15)
    y_smooth  = torch.full_like(y_pred, ls / n_classes)
    y_smooth.scatter_(1, y_true.unsqueeze(1), 1.0 - ls + ls / n_classes)
    per_sample_loss = -(y_smooth * torch.log(y_pred.clamp(1e-15, 1))).sum(dim=1)
    if focal_gamma > 0:
        pt = y_pred.gather(1, y_true.unsqueeze(1)).squeeze(1).clamp(1e-15, 1.0)
        per_sample_loss = per_sample_loss * torch.pow(1.0 - pt, focal_gamma)
    if class_weights is not None:
        sample_weights = class_weights[y_true]
        return (per_sample_loss * sample_weights).sum() / sample_weights.sum()
    return per_sample_loss.mean()

#  Sklearn-compatible wrapper
class RealMLP_TD_Classifier(BaseEstimator):
    """
    Sklearn-compatible wrapper around RealMLP, matching the interface of
    pytabkit's RealMLP_TD_Classifier:
    model = RealMLP_TD_Classifier(**CONFIG)
    model.fit(X_train, y_train, X_val, y_val, cat_col_names=CATS)
    proba = model.predict_proba(X_test)
    """
    def __init__(self, **kwargs):
        # Accept any subset of CONFIG keys; fall back to CONFIG defaults
        self.params = {**CONFIG, **kwargs}

    def fit(self, X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val,
            cat_col_names=None, X_test: pd.DataFrame = None):
        p   = self.params
        dev = torch.device(p["device"] if torch.cuda.is_available() else "cpu")
        verbose = p["verbosity"]
        cat_col_names = cat_col_names or []
        num_col_names = [c for c in X_train.columns if c not in cat_col_names]

        #  Split num / cat
        X_tr_num  = X_train[num_col_names].values.astype(np.float32)
        X_val_num = X_val[num_col_names].values.astype(np.float32)
        X_tr_cat  = X_train[cat_col_names].values.astype(np.int64)
        X_val_cat = X_val[cat_col_names].values.astype(np.int64)
        y_tr      = np.asarray(y_train)
        y_v       = np.asarray(y_val)

        #  Numerical preprocessing
        self.preprocessor_ = NumericalPreprocessor(p["tfms"])
        self.preprocessor_.fit(X_tr_num)
        X_tr_num  = self.preprocessor_.transform(X_tr_num)
        X_val_num = self.preprocessor_.transform(X_val_num)

        #  Categorical dims
        self.cat_col_names_ = cat_col_names
        self.num_col_names_ = num_col_names
        if cat_col_names:
            all_cat = [X_tr_cat, X_val_cat]
            if X_test is not None:
                all_cat.append(X_test[cat_col_names].values.astype(np.int64))
            cat_dims = (np.concatenate(all_cat, axis=0).max(axis=0) + 1).tolist()
        else:
            cat_dims = []
        self.cat_dims_ = cat_dims

        # Clamp indices to [0, dim-1]  -1 codes (unseen pandas categories)
        # wrap to huge ints on GPU and cause device-side assert
        if cat_dims:
            cat_max = np.array(cat_dims) - 1
            X_tr_cat  = np.clip(X_tr_cat,  0, cat_max)
            X_val_cat = np.clip(X_val_cat, 0, cat_max)

        #  Class weights
        classes       = np.unique(y_tr)
        self.classes_ = classes
        weights_np    = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
        cw_power = float(p.get("class_weight_power", 1.0))
        if cw_power != 1.0:
            weights_np = np.power(weights_np, cw_power)
        cw_mult = p.get("class_weight_multipliers", None)
        if cw_mult is not None:
            cw_mult = np.asarray(cw_mult, dtype="float64")
            if len(cw_mult) != len(classes):
                raise ValueError("class_weight_multipliers must match number of classes")
            weights_np = weights_np * cw_mult
        class_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=dev)
        loss_prior_power = float(p.get("loss_prior_power", 0.0))
        loss_prob_multipliers = None
        if loss_prior_power != 0.0:
            class_counts = np.bincount(y_tr, minlength=len(classes)).astype("float64")
            class_counts = class_counts / np.exp(np.log(class_counts).mean())
            loss_mult_np = np.power(class_counts, loss_prior_power)
            loss_prob_multipliers = torch.as_tensor(loss_mult_np, dtype=torch.float32, device=dev)

        #  Build model
        n_classes    = len(classes)
        self.model_  = RealMLP(
            output_dim=n_classes,
            cat_dims=cat_dims,
            n_numerical=X_tr_num.shape[1],
            cfg=p,
        ).to(dev)

        param_groups = get_parameter_groups(self.model_, p)
        # Store the base lr on each group so the scheduler can scale from it
        for g in param_groups:
            g["lr_base"] = g["lr"]
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(p["mom"], p["sq_mom"]),
        )

        #  To tensors
        Xtn = torch.as_tensor(X_tr_num,  dtype=torch.float32, device=dev)
        Xtc = torch.as_tensor(X_tr_cat,  dtype=torch.long,    device=dev)
        ytt = torch.as_tensor(y_tr,      dtype=torch.long,    device=dev)
        Xvn = torch.as_tensor(X_val_num, dtype=torch.float32, device=dev)
        Xvc = torch.as_tensor(X_val_cat, dtype=torch.long,    device=dev)

        n_ens       = p["n_ens"]
        train_bs    = p["train_bs"]
        eval_bs     = p["eval_bs"]
        epochs      = p["epochs"]
        lr_sched    = p["lr_sched"]
        flat_ratio  = p["flat_ratio"]
        numeric_noise_std = float(p.get("numeric_noise_std", 0.0))
        ema_decay = float(p.get("ema_decay", 0.0))
        total_steps = epochs * len(y_tr)
        train_order = np.arange(len(y_tr))
        sample_weight_power = float(p.get("sample_weight_power", 0.0))
        sampler_rng = np.random.default_rng(int(p.get("random_state", 0)) + 2027)
        sample_probs = None
        if sample_weight_power > 0:
            class_counts = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
            sample_probs = np.power(1.0 / np.clip(class_counts[y_tr], 1.0, None), sample_weight_power)
            sample_probs = sample_probs / sample_probs.sum()

        best_score      = -np.inf
        best_epoch      = 0
        best_val_probs  = None
        best_state      = None
        ema_state = None
        if ema_decay > 0:
            ema_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}

        #  Epoch loop
        for epoch in range(epochs):
            self.model_.train()
            if sample_probs is not None:
                epoch_order = sampler_rng.choice(len(y_tr), size=len(y_tr), replace=True, p=sample_probs)
            else:
                epoch_order = train_order
            for start in range(0, len(y_tr), train_bs):
                progress  = (epoch * len(y_tr) + start) / total_steps
                idx_batch = epoch_order[start : start + train_bs]

                # Update lr for each param group using its base lr
                for g in optimizer.param_groups:
                    g["lr"] = apply_schedule(g["lr_base"], progress, lr_sched, flat_ratio)

                optimizer.zero_grad()
                x_num_batch = Xtn[idx_batch]
                if numeric_noise_std > 0:
                    x_num_batch = x_num_batch + torch.randn_like(x_num_batch) * numeric_noise_std
                y_pred = self.model_(x_num_batch, Xtc[idx_batch])    # (bs, n_ens, C)

                ls_val   = apply_schedule(p["ls_eps"],  progress, p["ls_eps_sched"],  flat_ratio)
                drop_val = apply_schedule(p["dropout"], progress, p["p_drop_sched"],  flat_ratio)
                for dm in self.model_._dropout_modules:
                    dm.p = drop_val

                loss = smooth_ce_loss(
                    ytt[idx_batch].repeat_interleave(n_ens),
                    y_pred.reshape(-1, n_classes),
                    ls=ls_val,
                    class_weights=class_weights,
                    focal_gamma=float(p.get("focal_gamma", 0.0)),
                    loss_prob_multipliers=loss_prob_multipliers,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), p["grad_clip"])
                optimizer.step()
                if ema_state is not None:
                    with torch.no_grad():
                        model_state = self.model_.state_dict()
                        for key, value in model_state.items():
                            if torch.is_floating_point(value):
                                ema_state[key].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
                            else:
                                ema_state[key].copy_(value)

            if sample_probs is None:
                np.random.shuffle(train_order)

            #  Validation
            self.model_.eval()
            live_state = None
            if ema_state is not None:
                live_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                self.model_.load_state_dict(ema_state, strict=True)
            with torch.no_grad():
                val_probs = np.concatenate([
                    self.model_(Xvn[s : s + eval_bs], Xvc[s : s + eval_bs])
                        .mean(dim=1).cpu().numpy()
                    for s in range(0, len(y_v), eval_bs)
                ], axis=0)
            if live_state is not None:
                self.model_.load_state_dict(live_state, strict=True)

            score_probs = val_probs
            eval_mult = p.get("eval_class_multipliers", None)
            if eval_mult is not None:
                eval_mult = np.asarray(eval_mult, dtype=np.float32)
                if len(eval_mult) != val_probs.shape[1]:
                    raise ValueError("eval_class_multipliers must match number of classes")
                score_probs = val_probs * eval_mult[None, :]
                score_probs /= np.clip(score_probs.sum(axis=1, keepdims=True), 1e-12, None)

            epoch_score = balanced_accuracy_score(y_v, np.argmax(score_probs, axis=1))
            improved    = epoch_score > best_score
            if improved:
                best_score     = epoch_score
                best_epoch     = epoch + 1
                best_val_probs = score_probs.copy()
                state_src = ema_state if ema_state is not None else self.model_.state_dict()
                best_state = {k: v.detach().clone() for k, v in state_src.items()}

            if verbose >= 2:
                print(
                    f"  epoch {epoch + 1}/{epochs}  "
                    f"score = {epoch_score:.5f}  "
                    f"best = {best_score:.5f}  "
                    f"ls = {ls_val:.4f}  drop = {drop_val:.4f}"
                    + (" " if improved else "")
                )

            #  Early stopping
            if p["use_early_stopping"]:
                patience = (best_epoch * p["early_stopping_multiplicative_patience"]
                            + p["early_stopping_additive_patience"])
                if (epoch + 1) > patience:
                    if verbose >= 1:
                        print(f"  Early stopping at epoch {epoch + 1} "
                              f"(best epoch {best_epoch})")
                    break

        #  Restore best weights
        if best_state is not None:
            self.model_.load_state_dict(best_state, strict=True)
        self.best_score_     = best_score
        self.best_val_probs_ = best_val_probs
        self._dev            = dev
        if verbose >= 1:
            print(f"   best score: {best_score:.5f}  (epoch {best_epoch})")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        eval_bs = self.params["eval_bs"]
        X_num = self.preprocessor_.transform(
            X[self.num_col_names_].values.astype(np.float32)
        )
        X_cat = X[self.cat_col_names_].values.astype(np.int64)
        # Clamp to valid embedding range  guards against -1 codes (unseen
        # categories in pandas category dtype) which wrap to large ints on GPU
        X_cat = np.clip(X_cat, 0, np.array(self.cat_dims_) - 1)
        Xn = torch.as_tensor(X_num, dtype=torch.float32, device=self._dev)
        Xc = torch.as_tensor(X_cat, dtype=torch.long,    device=self._dev)
        self.model_.eval()
        with torch.no_grad():
            return np.concatenate([
                self.model_(Xn[s : s + eval_bs], Xc[s : s + eval_bs])
                    .mean(dim=1).cpu().numpy()
                for s in range(0, len(X_num), eval_bs)
            ], axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# R2-103 Configuration (verbatim from notebook)
# ---------------------------------------------------------------------------
CONFIG = {
    # Model architecture
    'n_ens': 8,
    'embed_dim': 7,
    'onehot_thresh': 10,
    'hidden_dims': [768, 768, 768],
    'dropout': 0.044,
    'p_drop_sched': 'expm4t',
    'activation': nn.GELU,
    'add_front_scale': True,

    # PBLD / periodic numerical embedding
    'pbld_hidden_dim': 16,
    'pbld_out_dim': 5,
    'pbld_freq_scale': 2.33,
    'pbld_activation': nn.PReLU,
    'pbld_lr_factor': 0.115,

    # Optimizer and training objective
    'lr': 0.01,
    'mom': 0.9,
    'sq_mom': 0.98,
    'lr_sched': 'flat_cos',
    'flat_ratio': 0.20,
    'first_layer_lr_factor': 1.0,
    'first_layer_wd_factor': 0.1,
    'lr_scale_mult': 10.0,
    'lr_bias_mult': 0.1,
    'weight_decay': 0.0125,
    'wd_scale_mult': 0.1,
    'wd_bias_mult': 0.5,
    'grad_clip': 1.0,
    'class_weight_power': 0.0,
    'class_weight_multipliers': None,
    'sample_weight_power': 0.0,
    'loss_prior_power': 1.075,
    'focal_gamma': 0.0,
    'eval_class_multipliers': None,

    # Label smoothing
    'ls_eps': 0.04,
    'ls_eps_sched': 'cos',

    # Preprocessing
    'tfms': ['median_center', 'robust_scale'],

    # Training loop
    'epochs': 6,
    'train_bs': 256,
    'eval_bs': 10240,
    'numeric_noise_std': 0.0,
    'ema_decay': 0.997875,
    'verbosity': 2,

    # Early stopping
    'use_early_stopping': False,
    'early_stopping_additive_patience': 10,
    'early_stopping_multiplicative_patience': 1,

    # Device and seed
    'device': str(DEVICE),
    'random_state': SEED,
}

TE = True
print('FOLDS:', FOLDS)
print('Target encoding:', TE)
print('R2-103 key config:', {k: CONFIG[k] for k in ['flat_ratio', 'dropout', 'weight_decay', 'loss_prior_power', 'class_weight_power', 'ema_decay', 'first_layer_lr_factor']})


# ---------------------------------------------------------------------------
# 5-Fold CV
# ---------------------------------------------------------------------------
def metric(y_true, y_pred_proba):
    return balanced_accuracy_score(y_true, np.argmax(y_pred_proba, axis=1))

def make_target_encoder(seed):
    try:
        return TargetEncoder(target_type='multiclass', cv=5, smooth='auto', shuffle=True, random_state=seed)
    except TypeError:
        return TargetEncoder(cv=5, smooth='auto', shuffle=True, random_state=seed)

# FOLD ALIGNMENT: StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)
# with y the integer labels (GALAXY=0,QSO=1,STAR=2) in original train-CSV row order.
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros((len(X), N_CLASSES), dtype='float32')
test_preds = np.zeros((len(X_test), N_CLASSES), dtype='float32')
fold_rows = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    # skf above still uses random_state=SEED=42 (folds aligned); only the MODEL seed differs
    # from the first variant so this is a genuinely diverse second seed of the same recipe.
    fold_seed = 999 + fold * 100
    seed_everything(fold_seed)
    CONFIG['random_state'] = fold_seed

    X_tr = X.iloc[tr_idx].copy()
    X_val = X.iloc[val_idx].copy()
    X_tst = X_test.copy()
    y_tr = y.iloc[tr_idx]
    y_val = y.iloc[val_idx]

    te_names = []
    if TE:
        encoder = make_target_encoder(fold_seed)
        tr_enc = encoder.fit_transform(X_tr[combo_names], y_tr)
        val_enc = encoder.transform(X_val[combo_names])
        tst_enc = encoder.transform(X_tst[combo_names])

        te_names = [f'_{col}TE_class{cls}' for col in combo_names for cls in range(N_CLASSES)]
        X_tr[te_names] = np.asarray(tr_enc, dtype='float32')
        X_val[te_names] = np.asarray(val_enc, dtype='float32')
        X_tst[te_names] = np.asarray(tst_enc, dtype='float32')

    X_tr = X_tr.reindex(sorted(X_tr.columns), axis=1)
    X_val = X_val.reindex(sorted(X_val.columns), axis=1)
    X_tst = X_tst.reindex(sorted(X_tst.columns), axis=1)

    if fold == 1:
        print('Number of training features:', X_tr.shape[1])
        print('Number of categorical columns:', len(cat_cols))
        print('Target-encoded columns:', len(te_names))

    print('\n' + '#' * 80)
    print(f'Fold {fold}/{FOLDS} | train={len(X_tr)} valid={len(X_val)}')
    print('#' * 80)

    model = RealMLP_TD_Classifier(**CONFIG)
    model.fit(
        X_tr,
        y_tr,
        X_val,
        y_val,
        cat_col_names=cat_cols,
        X_test=X_tst,
    )

    oof[val_idx] = model.best_val_probs_.astype('float32')
    test_preds += model.predict_proba(X_tst).astype('float32') / FOLDS

    fold_score = metric(y_val, oof[val_idx])
    fold_rows.append({
        'fold': fold,
        'score': fold_score,
        'n_train': len(X_tr),
        'n_valid': len(X_val),
        'n_features': X_tr.shape[1],
    })
    print(f'Fold {fold} balanced accuracy: {fold_score:.6f}')

    del model, X_tr, X_val, X_tst, y_tr, y_val
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

fold_scores = pd.DataFrame(fold_rows)
overall_score = metric(y, oof)
print('=' * 80)
print(fold_scores.to_string(index=False))
print(f'Mean fold balanced accuracy: {fold_scores.score.mean():.6f}')
print(f'Overall OOF balanced accuracy: {overall_score:.6f}')
print('=' * 80)


# ---------------------------------------------------------------------------
# CV Diagnostics (per-class recall = diagonal of row-normalized confusion matrix)
# ---------------------------------------------------------------------------
oof_labels = np.argmax(oof, axis=1)
cm = confusion_matrix(y, oof_labels, labels=np.arange(N_CLASSES))
cm_df = pd.DataFrame(cm, index=[f'true_{c}' for c in CLASSES], columns=[f'pred_{c}' for c in CLASSES])
print('Confusion matrix:')
print(cm_df.to_string())

per_class_recall = {}
for k, name in INV_MAP.items():
    mask = (y.values == k)
    per_class_recall[name] = float((oof_labels[mask] == k).mean())
print('Per-class OOF recall:')
print(per_class_recall)


# ---------------------------------------------------------------------------
# Save artifacts: oof_realmlp5.npy + test_realmlp5.npy + submission.csv + results.txt
# ---------------------------------------------------------------------------
# Row-normalize and clip away from exactly 0 (matches stacking-pipeline conventions).
def normalize_clip(p):
    p = np.clip(p.astype('float32'), 1e-15, None)
    p = p / p.sum(axis=1, keepdims=True)
    return p.astype('float32')

oof = normalize_clip(oof)
test_preds = normalize_clip(test_preds)

assert oof.shape == (len(X), N_CLASSES), oof.shape
assert test_preds.shape == (len(X_test), N_CLASSES), test_preds.shape

np.save(OOF_PATH, oof)
np.save(PRED_PATH, test_preds)

# Submission rows follow sample_submission order; test_preds was built in X_test order,
# which equals test.csv order, which equals sample_submission order.
submission = sub.copy()
submission[TARGET] = np.argmax(test_preds, axis=1)
submission[TARGET] = submission[TARGET].map(INV_MAP)
submission.to_csv(SUB_PATH, index=False)

# Write results.txt (and echo same content to stdout).
lines = []
lines.append('s6e6-realmlp5  (RealMLP R2-103 port)')
lines.append(f'Label map: {LABEL_MAP}  (col order [GALAXY,QSO,STAR])')
lines.append(f'Folds: StratifiedKFold({FOLDS}, shuffle=True, random_state={SEED})')
lines.append('')
lines.append('Per-fold balanced accuracy:')
for r in fold_rows:
    lines.append(f"  fold {r['fold']}: {r['score']:.6f}  (train={r['n_train']} valid={r['n_valid']} feats={r['n_features']})")
lines.append('')
lines.append(f'Mean fold balanced accuracy:    {fold_scores.score.mean():.6f}')
lines.append(f'Overall OOF balanced accuracy:  {overall_score:.6f}')
lines.append('')
lines.append('Per-class OOF recall:')
for name in CLASSES:
    lines.append(f'  {name}: {per_class_recall[name]:.6f}')
lines.append('')
lines.append('Confusion matrix (rows=true, cols=pred):')
lines.append(cm_df.to_string())
lines.append('')
lines.append(f'Saved OOF:        {OOF_PATH} {oof.shape}')
lines.append(f'Saved test preds: {PRED_PATH} {test_preds.shape}')
lines.append(f'Saved submission: {SUB_PATH} {submission.shape}')
lines.append('')
lines.append(f'FINAL: realmlp5 overall OOF balanced accuracy = {overall_score:.6f}')

report = '\n'.join(lines)
with open(RESULTS_PATH, 'w') as f:
    f.write(report + '\n')
print(report)
print(submission[TARGET].value_counts())
log('Done.')
