"""Logistic-regression stacking meta-learner over base-model OOF probabilities.

Simple weighted averaging can't exploit a weak-but-diverse member (e.g. logreg raises the
oracle ceiling +0.0066 but drags a flat blend down). A meta-learner can learn to trust each
base model only where it's reliable, converting that extra coverage into balanced accuracy.

Meta-features = log-probabilities of each base model. Meta-OOF is produced by an inner
StratifiedKFold(seed) over the (already out-of-fold) base OOF, so the reported score is honest.
Reports BA before/after class-bias search; compares to the flat-blend best (0.96609).

Usage:
    python stack.py --models lgb,xgb,cat,hgb,logreg
    python stack.py --models lgb,xgb,cat,hgb,logreg,mlp --output submissions/stack.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import ARTIFACTS, CLASS_ORDER, DATA, ID_COL, N_SPLITS, RANDOM_STATE, SUBMISSIONS, TARGET_COL
from features import load_data

EPS = 1e-15


def _logp(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1 - EPS))


def prob_to_logit(p: np.ndarray, clip: float = 30.0) -> np.ndarray:
    """One-vs-rest logit log(p/(1-p)), clipped. cdeotte's GPU-LR stacker feeds these
    (not log-probs) into a class-weighted multinomial LR — slightly stronger separation.
    Cast to float64 + a float32-safe epsilon so clip(1-eps) doesn't round to 1.0 (which
    would make 1-p=0 → inf). The outer clip then just guards extreme-confidence rows."""
    p = np.clip(p.astype(np.float64), 1e-7, 1.0 - 1e-7)
    return np.clip(np.log(p / (1.0 - p)), -clip, clip)


def build_meta(kind: str, seed: int, C: float, scale: bool = False):
    """Build a stacking meta-learner. logreg is the proven default; mlp adds non-linear
    capacity (the meta-features are only ~3*n_models dims so a small net + strong alpha).
    scale=True standardizes the (logit) features before logreg — slightly better + cures
    lbfgs non-convergence at large model counts (best meta-tune config: scale + C=3)."""
    if kind == "logreg":
        lr = LogisticRegression(
            C=C, max_iter=4000, class_weight="balanced", n_jobs=1, random_state=seed
        )
        return make_pipeline(StandardScaler(), lr) if scale else lr
    if kind == "mlp":
        # StandardScaler: log-prob features span very different ranges; NNs need scaling.
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                alpha=1e-3,
                batch_size=4096,
                learning_rate_init=1e-3,
                early_stopping=True,
                n_iter_no_change=15,
                max_iter=300,
                random_state=seed,
            ),
        )
    raise ValueError(f"unknown meta kind: {kind}")


def load_stack(models: list[str], which: str, feat: str = "logp") -> np.ndarray:
    """Concatenate per-model features for stacking. which in {'oof','test'};
    feat in {'logp' (log-probabilities, default), 'logit' (one-vs-rest logits)}."""
    transform = prob_to_logit if feat == "logit" else _logp
    mats = []
    for m in models:
        path = ARTIFACTS / f"{which}_{m}.npy"
        arr = np.load(path)
        mats.append(transform(arr))
    return np.concatenate(mats, axis=1)


def search_bias(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    logp = _logp(proba)
    res = differential_evolution(
        lambda b: -balanced_accuracy_score(y, (logp + b).argmax(1)),
        bounds=[(-1.0, 1.0)] * proba.shape[1],
        seed=42, tol=1e-6, maxiter=100, polish=True,
    )
    return res.x


def search_scale_shift(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Fit per-class scale+shift: argmax_c(a_c * logp_c + b_c), 6 params for 3-class."""
    logp = _logp(proba)
    nc = proba.shape[1]
    res = differential_evolution(
        lambda p: -balanced_accuracy_score(y, (p[:nc] * logp + p[nc:]).argmax(1)),
        bounds=[(0.2, 3.0)] * nc + [(-2.0, 2.0)] * nc,
        seed=42, tol=1e-6, maxiter=200, polish=True,
    )
    return res.x


def apply_scale_shift(proba: np.ndarray, params: np.ndarray) -> np.ndarray:
    logp = _logp(proba)
    nc = proba.shape[1]
    return (params[:nc] * logp + params[nc:]).argmax(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LogReg stacking meta-learner for S6E6")
    p.add_argument("--models", default="lgb,xgb,cat,hgb,logreg")
    p.add_argument("--meta", choices=["logreg", "mlp"], default="logreg",
                   help="Meta-learner family (logreg=proven default, mlp=non-linear)")
    p.add_argument("--calib", choices=["bias", "scale_shift"], default="bias",
                   help="Calibration method: additive class bias (default) or per-class scale+shift")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--feat", choices=["logp", "logit"], default="logp",
                   help="Base-model feature transform fed to the meta-learner")
    p.add_argument("--meta-seeds", type=int, default=1,
                   help="Average the meta-learner over this many seeds (seed, seed+1, ...) to cut meta variance")
    p.add_argument("--scale", action="store_true",
                   help="StandardScale the meta-features before logreg (best meta-tune config: --scale --C 3)")
    p.add_argument("--output", type=Path, help="Write a submission CSV from the full-data meta-fit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    y = np.load(ARTIFACTS / "y_train.npy")
    n_classes = len(CLASS_ORDER)

    X = load_stack(models, "oof", feat=args.feat)
    meta_seeds = [args.seed + i for i in range(max(1, args.meta_seeds))]
    print(f"Stacking {models} -> meta-features {X.shape}  (meta={args.meta}, feat={args.feat}, meta_seeds={meta_seeds})")

    # Inner StratifiedKFold meta-OOF, averaged over meta_seeds to reduce meta variance.
    meta_oof = np.zeros((len(y), n_classes))
    for ms in meta_seeds:
        oof_s = np.zeros((len(y), n_classes))
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=ms)
        for tr, va in skf.split(X, y):
            meta = build_meta(args.meta, ms, args.C, scale=args.scale)
            meta.fit(X[tr], y[tr])
            oof_s[va] = meta.predict_proba(X[va])
        meta_oof += oof_s / len(meta_seeds)

    ba_before = balanced_accuracy_score(y, meta_oof.argmax(1))
    if args.calib == "scale_shift":
        calib_params = search_scale_shift(y, meta_oof)
        pred_calib = apply_scale_shift(meta_oof, calib_params)
        nc = n_classes
        print(f"  scale: {dict(zip(CLASS_ORDER, np.round(calib_params[:nc], 4)))}")
        print(f"  shift: {dict(zip(CLASS_ORDER, np.round(calib_params[nc:], 4)))}")
    else:
        calib_params = search_bias(y, meta_oof)
        pred_calib = (_logp(meta_oof) + calib_params).argmax(1)
        print(f"  class bias: {dict(zip(CLASS_ORDER, np.round(calib_params, 4)))}")
    ba_after = balanced_accuracy_score(y, pred_calib)
    recalls = {CLASS_ORDER[c]: round(float((pred_calib[y == c] == c).mean()), 4)
               for c in range(n_classes)}
    print(f"Meta-OOF BA before calib: {ba_before:.5f}")
    print(f"Meta-OOF BA after  calib: {ba_after:.5f}  recalls={recalls}  (calib={args.calib})")
    print(f"  (flat-blend best to beat: 0.96609)")

    if args.output:
        _, test, _ = load_data(DATA)
        Xt = load_stack(models, "test", feat=args.feat)
        test_proba = np.zeros((len(Xt), n_classes))
        for ms in meta_seeds:
            meta_full = build_meta(args.meta, ms, args.C, scale=args.scale)
            meta_full.fit(X, y)
            test_proba += meta_full.predict_proba(Xt) / len(meta_seeds)
        if args.calib == "scale_shift":
            pred = apply_scale_shift(test_proba, calib_params)
        else:
            pred = (_logp(test_proba) + calib_params).argmax(1)
        out = args.output if args.output.is_absolute() else (SUBMISSIONS.parent / args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: [CLASS_ORDER[i] for i in pred]}).to_csv(out, index=False)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
