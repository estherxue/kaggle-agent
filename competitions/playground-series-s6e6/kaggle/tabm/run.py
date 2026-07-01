"""Exp: TabM tabular model (5-fold OOF) — a parameter-efficient deep ensemble MLP
(BatchEnsemble-style rank-1 ensembling) with a periodic (PBLD) numeric embedding and
one-hot/embedding categorical layer. A distinct neural family vs the GBDTs/RealMLP/TabPFN
already in the stack, to raise the ensemble's oracle ceiling.

SELF-CONTAINED: does NOT import the user's pipeline modules. Feature engineering and the
TabM model are ported verbatim from the source notebook (s6e6-tabm.ipynb). No external
package install is required — TabM here is plain PyTorch (no pytabkit / tabm pip dependency),
so Kaggle's pre-installed, GPU-matched torch is used as-is (no "no kernel image" risk).

Label mapping: GALAXY=0, QSO=1, STAR=2 (matches all existing artifacts).
Fold split: StratifiedKFold(5, shuffle=True, random_state=42).split(X, y) with integer
labels in ORIGINAL train-CSV row order — byte-identical to train_oof.py, so oof_tabm.npy
aligns row-for-row with every other oof_*.npy.

Outputs (to /kaggle/working/):
  oof_tabm.npy   (577347, 3) float32  OOF probabilities, [GALAXY,QSO,STAR], CSV row order
  test_tabm.npy  (247435, 3) float32  test probabilities, sample_submission order, fold-avg
  submission.csv argmax->label
  results.txt    per-fold / overall balanced accuracy + per-class recall

Internet ON, GPU ON.
"""
import math
import os
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_class_weight

# --- Pascal-compatible torch ---
# Kaggle's stock torch 2.10+cu128 dropped sm_60, but this batch kernel may land on a P100.
# Install a cu121 build (supports sm_60 P100 AND sm_75 T4) BEFORE importing torch.
import sys as _sys, subprocess as _sp
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "torch==2.4.1",
         "--extra-index-url", "https://download.pytorch.org/whl/cu121"], check=False)

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
print("PyTorch version:", torch.__version__)


def seed_everything(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ── Paths / data load ─────────────────────────────────────────────────────────
COMP = "/kaggle/input/competitions/playground-series-s6e6"
if not os.path.isdir(COMP):
    COMP = "/kaggle/input/playground-series-s6e6"
WORK = "/kaggle/working"
os.makedirs(WORK, exist_ok=True)

train = pd.read_csv(os.path.join(COMP, "train.csv"))
test = pd.read_csv(os.path.join(COMP, "test.csv"))
sample_sub = pd.read_csv(os.path.join(COMP, "sample_submission.csv"))
print("Train shape:", train.shape)
print("Test shape :", test.shape)

ID = "id"
TARGET = "class"
CLASS_ORDER = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {"GALAXY": 0, "QSO": 1, "STAR": 2}
INV_LABEL_MAP = {0: "GALAXY", 1: "QSO", 2: "STAR"}

# CRITICAL: integer labels in ORIGINAL CSV row order drive the StratifiedKFold split,
# which is what makes oof_tabm.npy align with every other artifact.
train[TARGET] = train[TARGET].map(LABEL_MAP)
X = train.drop([ID, TARGET], axis=1)
y = train[TARGET]
X_test = test.drop([ID], axis=1)
test_id = test[ID]

# Align test row order to sample_submission (defines the official test row order).
# load above keeps test.csv order; reorder X_test/test_id to sample_submission ids.
if not test_id.equals(sample_sub[ID]):
    order = sample_sub[ID].tolist()
    pos = {v: i for i, v in enumerate(test_id.tolist())}
    perm = [pos[v] for v in order]
    X_test = X_test.iloc[perm].reset_index(drop=True)
    test_id = test_id.iloc[perm].reset_index(drop=True)
assert test_id.tolist() == sample_sub[ID].tolist(), "test order must match sample_submission"

del train, test
print("X      init shape:", X.shape)
print("X_test init shape:", X_test.shape, "\n")

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
print("init len(cat_cols):", len(cat_cols))
print("init len(num_cols):", len(num_cols), "\n")

# ── Feature engineering (verbatim from notebook) ───────────────────────────────
category_map = {}
important_combos = [
    ("alpha_cat_", "delta_cat_"),
    ("u_cat_", "z_cat_"),
]
important_combos = sorted(important_combos)


def feature_engineering(df, fit=False):
    # Arithmetic interaction
    df["_g_/_redshift"] = (df["g"] / (df["redshift"] + 1e-6)).astype("float32")
    df["_i_/_redshift"] = (df["i"] / (df["redshift"] + 1e-6)).astype("float32")

    # Categorize string cats
    for col in cat_cols:
        if fit:
            codes, uniques = df[col].factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = df[col].map(code_map).fillna(-1).astype("int32")
        df[col] = codes
        df[col] = df[col].astype("category")

    # Categorize numericals
    for col in num_cols:
        cat_name = f"{col}_cat_"
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype("int32")
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype("category")

    # Create interaction categories
    combo_names = []
    for cols in important_combos:
        combo_name = "_".join(cols) + "_"
        combo_names.append(combo_name)
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + "_" + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            category_map[combo_name] = uniques
        else:
            uniques = category_map[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype("int32")
        df[combo_name] = codes
        df[combo_name] = df[combo_name].astype("category")

    new_cat_cols = [col for col in df.columns if col.endswith("_")]
    new_num_cols = [col for col in df.columns if col.startswith("_")]
    return df, new_cat_cols, new_num_cols, combo_names


X, new_cat_cols, new_num_cols, combo_names = feature_engineering(X, fit=True)
X_test, new_cat_cols, new_num_cols, combo_names = feature_engineering(X_test, fit=False)
cat_cols += new_cat_cols
num_cols += new_num_cols
print("len(new_cat_cols):", len(new_cat_cols))
print("len(new_num_cols):", len(new_num_cols), "\n")

cat_cols = sorted(cat_cols)
X = X.reindex(sorted(X.columns), axis=1)
X_test = X_test.reindex(sorted(X_test.columns), axis=1)
print("prep len(cat_cols):", len(cat_cols))
print("prep len(num_cols):", len(num_cols), "\n")
print("X      prep shape:", X.shape)
print("X_test prep shape:", X_test.shape, "\n")


# ── Preprocessing (verbatim) ───────────────────────────────────────────────────
class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, tfms):
        self._tfms = [
            t
            for t in tfms
            if t in ("median_center", "robust_scale", "smooth_clip", "l2_normalize")
        ]

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


# ── Model components (TabM, verbatim) ──────────────────────────────────────────
class TabMCategoricalFeatureLayer(nn.Module):
    def __init__(self, n_ens: int, cat_dims, embed_dim: int = 8, onehot_thresh: int = 8, device=None):
        super().__init__()
        self.n_ens = n_ens
        self.cat_dims = cat_dims
        self.onehot_features = []

        self.embed_layers = nn.ModuleList()
        self.embed_scales = nn.ParameterList()
        self._embed_feature_indices = []

        for i, dim in enumerate(cat_dims):
            if dim <= onehot_thresh:
                self.onehot_features.append(i)
            else:
                self.embed_layers.append(nn.Embedding(dim, embed_dim))
                self.embed_scales.append(nn.Parameter(torch.ones(n_ens, embed_dim)))
                self._embed_feature_indices.append(i)

    def forward(self, x):
        batch_size, n_ens, _ = x.shape
        features = []

        if self.onehot_features:
            onehot_x = x[:, :, self.onehot_features]
            onehot_dims = [self.cat_dims[i] for i in self.onehot_features]
            total_oh = sum(onehot_dims)
            encoded = torch.zeros(batch_size, n_ens, total_oh, device=x.device)
            start = 0
            for idx, dim in enumerate(onehot_dims):
                pos = onehot_x[:, :, idx : idx + 1].long()
                encoded.scatter_(2, pos + start, 1.0)
                start += dim
            features.append(encoded)

        for emb, scale, feat_idx in zip(
            self.embed_layers, self.embed_scales, self._embed_feature_indices
        ):
            indices = x[:, :, feat_idx].long()
            feat_embs = emb(indices)
            feat_combined = feat_embs * scale.unsqueeze(0)
            features.append(feat_combined)

        return torch.cat(features, dim=2)


class ScalingLayer(nn.Module):
    def __init__(self, n_ens: int, n_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))

    def forward(self, x):
        return x * self.scale[None, :, :]


class TabMLinear(nn.Module):
    """Parameter-efficient ensemble linear layer."""

    def __init__(self, n_ens: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(in_features, out_features))

        self.scale_in = nn.Parameter(torch.ones(n_ens, in_features))
        self.scale_out = nn.Parameter(torch.ones(n_ens, out_features))

        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
            self.bias_ens = nn.Parameter(torch.zeros(n_ens, out_features))
        else:
            self.register_parameter("bias", None)
            self.register_parameter("bias_ens", None)

    def forward(self, x):
        x = x * self.scale_in.unsqueeze(0)
        x = torch.einsum("bki,io->bko", x, self.weight) / math.sqrt(self.in_features)
        x = x * self.scale_out.unsqueeze(0)
        if self.bias is not None:
            x = x + self.bias.view(1, 1, -1) + self.bias_ens.unsqueeze(0)
        return x


class TabMPBLDEmbedding(nn.Module):
    """Periodic Basis with Learned Decay modified for parameter-efficient ensembling."""

    def __init__(self, n_ens: int, n_features: int, hidden_dim: int = 16, out_dim: int = 4,
                 freq_scale: float = 0.1, activation=nn.GELU):
        super().__init__()
        self.n_ens = n_ens
        self.n_features = n_features
        self.out_dim = out_dim

        self.w1 = nn.Parameter(torch.randn(n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_features, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_features, hidden_dim, out_dim - 1) / math.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(n_features, out_dim - 1))

        self.scale_out = nn.Parameter(torch.ones(n_ens, n_features, out_dim - 1))

        self.act = activation()
        nn.init.uniform_(self.b1, -math.pi, math.pi)

    def forward(self, x):
        periodic = torch.cos(
            2
            * math.pi
            * (
                x.unsqueeze(-1) * self.w1.view(1, 1, self.n_features, -1)
                + self.b1.view(1, 1, self.n_features, -1)
            )
        )

        transformed = self.act(
            torch.einsum("bkfh,fhd->bkfd", periodic, self.w2)
            + self.b2.view(1, 1, self.n_features, -1)
        )

        transformed = transformed * self.scale_out.unsqueeze(0)

        feat = torch.cat([x.unsqueeze(-1), transformed], dim=-1)
        return feat.flatten(start_dim=2)


# ── Model (verbatim) ────────────────────────────────────────────────────────────
class TabMModel(nn.Module):
    def __init__(self, output_dim: int, cat_dims, n_numerical: int, cfg: dict):
        super().__init__()
        n_ens = cfg["n_ens"]
        embed_dim = cfg["embed_dim"]
        self.n_ens = n_ens

        self.cate = TabMCategoricalFeatureLayer(
            n_ens=n_ens,
            cat_dims=cat_dims,
            embed_dim=embed_dim,
            onehot_thresh=cfg["onehot_thresh"],
        )
        self.num_embed = TabMPBLDEmbedding(
            n_ens=n_ens,
            n_features=n_numerical,
            hidden_dim=cfg["pbld_hidden_dim"],
            out_dim=cfg["pbld_out_dim"],
            freq_scale=cfg["pbld_freq_scale"],
            activation=cfg["pbld_activation"],
        )

        num_emb_dim = n_numerical * cfg["pbld_out_dim"]
        cat_emb_dim = sum(c if c <= cfg["onehot_thresh"] else embed_dim for c in cat_dims)
        total_dim = num_emb_dim + cat_emb_dim
        hidden_dims = cfg["hidden_dims"]

        act = cfg["activation"]

        layers = []
        if cfg["add_front_scale"]:
            layers.append(ScalingLayer(n_ens=n_ens, n_features=total_dim))

        self._dropout_modules = []
        in_dim = total_dim
        for i, out_dim_h in enumerate(hidden_dims):
            linear = TabMLinear(n_ens=n_ens, in_features=in_dim, out_features=out_dim_h)
            if i == 0:
                self.first_linear = linear
            drop = nn.Dropout(cfg["dropout"])
            self._dropout_modules.append(drop)
            layers += [linear, act(), drop]
            in_dim = out_dim_h

        self.hidden = nn.Sequential(*layers)
        self.output_layer = TabMLinear(n_ens=n_ens, in_features=in_dim, out_features=output_dim)

    def forward(self, x_num, x_cat):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_num = self.num_embed(x_num)
        x_cat = self.cate(x_cat)
        combined = torch.cat([x_num, x_cat], dim=2)
        x = self.hidden(combined)
        x = self.output_layer(x)
        return F.softmax(x, dim=2)


# ── Helpers & sklearn wrapper (verbatim) ───────────────────────────────────────
def apply_schedule(init_value: float, progress: float, sched: str, flat_ratio: float = 0.3) -> float:
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
    raise ValueError(f"Unknown schedule: '{sched}'")


def get_parameter_groups(model: TabMModel, p: dict):
    first_linear_weight_id = id(model.first_linear.weight)

    scale_p, pbld_p, first_w_p, other_w_p, bias_p = [], [], [], [], []
    for name, param in model.named_parameters():
        if "num_embed" in name:
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
        {"params": scale_p, "lr": LR * p["lr_scale_mult"], "weight_decay": WD * p["wd_scale_mult"], "group": "scale"},
        {"params": pbld_p, "lr": LR * p["pbld_lr_factor"], "weight_decay": WD, "group": "pbld"},
        {"params": first_w_p, "lr": LR * p["first_layer_lr_factor"], "weight_decay": WD * p["first_layer_wd_factor"], "group": "first_w"},
        {"params": other_w_p, "lr": LR, "weight_decay": WD, "group": "other_w"},
        {"params": bias_p, "lr": LR * p["lr_bias_mult"], "weight_decay": WD * p["wd_bias_mult"], "group": "bias"},
    ]


def smooth_ce_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    ls: float = 0.0,
    class_weights: torch.Tensor = None,
) -> torch.Tensor:
    n_classes = y_pred.size(1)
    y_smooth = torch.full_like(y_pred, ls / n_classes)
    y_smooth.scatter_(1, y_true.unsqueeze(1), 1.0 - ls + ls / n_classes)
    per_sample_loss = -(y_smooth * torch.log(y_pred.clamp(1e-15, 1))).sum(dim=1)
    if class_weights is not None:
        sample_weights = class_weights[y_true]
        return (per_sample_loss * sample_weights).sum() / sample_weights.sum()
    return per_sample_loss.mean()


class TabM_TD_Classifier(BaseEstimator):
    def __init__(self, **kwargs):
        self.params = {**CONFIG, **kwargs}

    def fit(self, X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val,
            cat_col_names=None, ckpt_path: str = "tabm_ckpt.pth", X_test: pd.DataFrame = None):
        p = self.params
        dev = torch.device(p["device"] if torch.cuda.is_available() else "cpu")
        verbose = p["verbosity"]
        cat_col_names = cat_col_names or []
        num_col_names = [c for c in X_train.columns if c not in cat_col_names]

        X_tr_num = X_train[num_col_names].values.astype(np.float32)
        X_val_num = X_val[num_col_names].values.astype(np.float32)
        X_tr_cat = X_train[cat_col_names].values.astype(np.int64)
        X_val_cat = X_val[cat_col_names].values.astype(np.int64)
        y_tr = np.asarray(y_train)
        y_v = np.asarray(y_val)

        self.preprocessor_ = NumericalPreprocessor(p["tfms"])
        self.preprocessor_.fit(X_tr_num)
        X_tr_num = self.preprocessor_.transform(X_tr_num)
        X_val_num = self.preprocessor_.transform(X_val_num)

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

        if cat_dims:
            cat_max = np.array(cat_dims) - 1
            X_tr_cat = np.clip(X_tr_cat, 0, cat_max)
            X_val_cat = np.clip(X_val_cat, 0, cat_max)

        classes = np.unique(y_tr)
        self.classes_ = classes
        weights_np = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
        class_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=dev)

        n_classes = len(classes)
        self.model_ = TabMModel(
            output_dim=n_classes,
            cat_dims=cat_dims,
            n_numerical=X_tr_num.shape[1],
            cfg=p,
        ).to(dev)

        param_groups = get_parameter_groups(self.model_, p)
        for g in param_groups:
            g["lr_base"] = g["lr"]
        optimizer = torch.optim.AdamW(param_groups, betas=(p["mom"], p["sq_mom"]))

        Xtn = torch.as_tensor(X_tr_num, dtype=torch.float32, device=dev)
        Xtc = torch.as_tensor(X_tr_cat, dtype=torch.long, device=dev)
        ytt = torch.as_tensor(y_tr, dtype=torch.long, device=dev)
        Xvn = torch.as_tensor(X_val_num, dtype=torch.float32, device=dev)
        Xvc = torch.as_tensor(X_val_cat, dtype=torch.long, device=dev)

        n_ens = p["n_ens"]
        train_bs = p["train_bs"]
        eval_bs = p["eval_bs"]
        epochs = p["epochs"]
        lr_sched = p["lr_sched"]
        flat_ratio = p["flat_ratio"]
        total_steps = epochs * len(y_tr)
        train_order = np.arange(len(y_tr))

        best_score = -np.inf
        best_epoch = 0
        best_val_probs = None
        self.ckpt_path_ = ckpt_path

        for epoch in range(epochs):
            self.model_.train()
            for start in range(0, len(y_tr), train_bs):
                progress = (epoch * len(y_tr) + start) / total_steps
                idx_batch = train_order[start : start + train_bs]

                for g in optimizer.param_groups:
                    g["lr"] = apply_schedule(g["lr_base"], progress, lr_sched, flat_ratio)

                optimizer.zero_grad()
                y_pred = self.model_(Xtn[idx_batch], Xtc[idx_batch])

                ls_val = apply_schedule(p["ls_eps"], progress, p["ls_eps_sched"], flat_ratio)
                drop_val = apply_schedule(p["dropout"], progress, p["p_drop_sched"], flat_ratio)
                for dm in self.model_._dropout_modules:
                    dm.p = drop_val

                loss = smooth_ce_loss(
                    ytt[idx_batch].repeat_interleave(n_ens),
                    y_pred.reshape(-1, n_classes),
                    ls=ls_val,
                    class_weights=class_weights,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), p["grad_clip"])
                optimizer.step()

            np.random.shuffle(train_order)

            self.model_.eval()
            with torch.no_grad():
                val_probs = np.concatenate(
                    [
                        self.model_(Xvn[s : s + eval_bs], Xvc[s : s + eval_bs]).mean(dim=1).cpu().numpy()
                        for s in range(0, len(y_v), eval_bs)
                    ],
                    axis=0,
                )

            epoch_score = balanced_accuracy_score(y_v, np.argmax(val_probs, axis=1))
            improved = epoch_score > best_score
            if improved:
                best_score = epoch_score
                best_epoch = epoch + 1
                best_val_probs = val_probs.copy()
                torch.save(self.model_.state_dict(), ckpt_path)

            if verbose >= 2:
                print(
                    f"  epoch {epoch + 1}/{epochs}  "
                    f"score = {epoch_score:.5f}  "
                    f"best = {best_score:.5f}  "
                    f"ls = {ls_val:.4f}  drop = {drop_val:.4f}"
                    + (" *" if improved else "")
                )

            if p["use_early_stopping"]:
                patience = (
                    best_epoch * p["early_stopping_multiplicative_patience"]
                    + p["early_stopping_additive_patience"]
                )
                if (epoch + 1) > patience:
                    if verbose >= 1:
                        print(f"  Early stopping at epoch {epoch + 1} (best epoch {best_epoch})")
                    break

        self.model_.load_state_dict(torch.load(ckpt_path))
        self.best_score_ = best_score
        self.best_val_probs_ = best_val_probs
        self._dev = dev
        if verbose >= 1:
            print(f"  -> best score: {best_score:.5f}  (epoch {best_epoch})")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        eval_bs = self.params["eval_bs"]
        X_num = self.preprocessor_.transform(X[self.num_col_names_].values.astype(np.float32))
        X_cat = X[self.cat_col_names_].values.astype(np.int64)
        X_cat = np.clip(X_cat, 0, np.array(self.cat_dims_) - 1)
        Xn = torch.as_tensor(X_num, dtype=torch.float32, device=self._dev)
        Xc = torch.as_tensor(X_cat, dtype=torch.long, device=self._dev)
        self.model_.eval()
        with torch.no_grad():
            return np.concatenate(
                [
                    self.model_(Xn[s : s + eval_bs], Xc[s : s + eval_bs]).mean(dim=1).cpu().numpy()
                    for s in range(0, len(X_num), eval_bs)
                ],
                axis=0,
            )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ── CONFIG (verbatim) ───────────────────────────────────────────────────────────
CONFIG = {
    # --- Model architecture ---
    "n_ens": 32,
    "embed_dim": 7,
    "onehot_thresh": 10,
    "hidden_dims": [1024, 512, 512],
    "dropout": 0.0,
    "p_drop_sched": "constant",
    "activation": nn.SiLU,
    "add_front_scale": True,
    # --- PBLD (periodic) embedding for numericals ---
    "pbld_hidden_dim": 20,
    "pbld_out_dim": 5,
    "pbld_freq_scale": 5.0,
    "pbld_activation": nn.PReLU,
    "pbld_lr_factor": 0.093,
    # --- Optimizer ---
    "lr": 0.01,
    "mom": 0.9,
    "sq_mom": 0.98,
    "lr_sched": "flat_cos",
    "flat_ratio": 0.3,
    "first_layer_lr_factor": 1.0,
    "first_layer_wd_factor": 0.1,
    "lr_scale_mult": 20.0,
    "lr_bias_mult": 0.1,
    "weight_decay": 0.013,
    "wd_scale_mult": 0.1,
    "wd_bias_mult": 0.5,
    "grad_clip": 1.0,
    # --- Label smoothing ---
    "ls_eps": 0.04,
    "ls_eps_sched": "cos",
    # --- Preprocessing ---
    "tfms": ["median_center", "robust_scale"],
    # --- Training loop ---
    "epochs": 25,
    "train_bs": 256,
    "eval_bs": 10240,
    "verbosity": 2,
    # --- Early stopping ---
    "use_early_stopping": True,
    "early_stopping_additive_patience": 5,
    "early_stopping_multiplicative_patience": 1,
    # --- Device ---
    "device": "cuda",
    "random_state": 42,
}

FOLDS = 5
SEED = 42
TE = True
n_classes = y.nunique()


def metric(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return balanced_accuracy_score(y_true, y_pred)


# ── 5-fold OOF ───────────────────────────────────────────────────────────────
# CRITICAL: this split is forced to StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)
# with integer labels in CSV order, so oof_tabm.npy aligns with every other artifact.
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros((len(X), n_classes), dtype=np.float64)
test_preds = np.zeros((len(X_test), n_classes), dtype=np.float64)

fold_scores = []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr = X.iloc[tr_idx].copy()
    X_val = X.iloc[val_idx].copy()
    X_tst = X_test.copy()

    # Target encoding (fit on this fold's train only, applied to val/test)
    fold_cat_cols = list(cat_cols)
    if TE:
        te_cols = combo_names
        encoder = TargetEncoder(cv=FOLDS, smooth="auto", shuffle=True, random_state=SEED)
        tr_enc = encoder.fit_transform(X_tr[te_cols], y.iloc[tr_idx])
        val_enc = encoder.transform(X_val[te_cols])
        tst_enc = encoder.transform(X_tst[te_cols])

        te_names = [f"_{col}TE_class{cls}" for col in te_cols for cls in range(n_classes)]
        X_tr[te_names] = tr_enc
        X_val[te_names] = val_enc
        X_tst[te_names] = tst_enc

    if fold == 1:
        print("len(FEATURES):", len(X_tr.columns.tolist()), "\n")
    print("#" * 16)
    print(f"### Fold {fold}/{FOLDS} ...")
    print("#" * 16)

    model = TabM_TD_Classifier(**CONFIG)
    model.fit(
        X_tr,
        y.iloc[tr_idx],
        X_val,
        y.iloc[val_idx],
        cat_col_names=fold_cat_cols,
        ckpt_path=os.path.join(WORK, f"model_fold{fold}.pth"),
        X_test=X_tst,
    )

    val_probs = model.best_val_probs_
    if val_probs.ndim == 3:
        val_probs = val_probs.mean(axis=1)
    oof_preds[val_idx] = val_probs

    tst_probs = model.predict_proba(X_tst)
    if tst_probs.ndim == 3:
        tst_probs = tst_probs.mean(axis=1)
    test_preds += tst_probs / FOLDS

    fold_score = metric(y.iloc[val_idx], oof_preds[val_idx])
    fold_scores.append(fold_score)
    print(f"\nFold {fold} | Score: {fold_score:.5f}\n")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

overall_ba = metric(y, oof_preds)
print("=" * 26)
print(f"Overall OOF Score: {overall_ba:.5f}")
print("=" * 26, "\n")


# ── Normalize / clip / save in canonical [GALAXY,QSO,STAR] order ────────────────
def clip_norm(p):
    p = np.clip(p.astype(np.float64), 1e-15, None)
    p = p / p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


# Columns are already in label order 0=GALAXY,1=QSO,2=STAR (n_classes columns,
# softmax over class index), matching CLASS_ORDER. clip + row-normalize.
oof_out = clip_norm(oof_preds)
test_out = clip_norm(test_preds)

assert oof_out.shape == (577347, 3), f"unexpected oof shape {oof_out.shape}"
assert test_out.shape == (247435, 3), f"unexpected test shape {test_out.shape}"

np.save(os.path.join(WORK, "oof_tabm.npy"), oof_out)
np.save(os.path.join(WORK, "test_tabm.npy"), test_out)
print("Saved oof_tabm.npy and test_tabm.npy")

# Submission (argmax -> label), in sample_submission row order
sub = pd.DataFrame({ID: test_id, TARGET: np.argmax(test_out, axis=1)})
sub[TARGET] = sub[TARGET].map(INV_LABEL_MAP)
sub.to_csv(os.path.join(WORK, "submission.csv"), index=False)

# ── Per-class recall (on OOF) + results.txt ────────────────────────────────────
y_arr = y.to_numpy()
oof_argmax = oof_out.argmax(1)
recalls = {
    CLASS_ORDER[c]: round(float((oof_argmax[y_arr == c] == c).mean()), 5)
    for c in range(n_classes)
}

lines = []
lines.append("=== s6e6-tabm: TabM (parameter-efficient deep ensemble MLP) ===")
lines.append(f"PyTorch: {torch.__version__}  device: {device}")
lines.append(f"X={X.shape}  X_test={X_test.shape}  n_classes={n_classes}")
lines.append(f"Fold split: StratifiedKFold(5, shuffle=True, random_state=42) on integer labels (CSV order)")
lines.append("Label map: GALAXY=0, QSO=1, STAR=2")
lines.append("")
for i, s in enumerate(fold_scores, 1):
    lines.append(f"fold {i}/5  balanced_accuracy = {s:.5f}")
lines.append("")
lines.append(f"Overall OOF balanced_accuracy = {overall_ba:.5f}")
lines.append(f"Per-class OOF recall: {recalls}")
lines.append("")
lines.append("Saved: oof_tabm.npy (577347,3), test_tabm.npy (247435,3), submission.csv")
lines.append(f"FINAL SUMMARY: tabm OOF BA={overall_ba:.5f}  recalls={recalls}")

report = "\n".join(lines)
with open(os.path.join(WORK, "results.txt"), "w") as fh:
    fh.write(report + "\n")
print(report)
print(f"\nFINAL SUMMARY: tabm OOF BA={overall_ba:.5f}  recalls={recalls}")
