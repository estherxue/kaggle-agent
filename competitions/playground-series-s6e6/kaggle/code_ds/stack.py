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


def build_meta(kind: str, seed: int, C: float):
    """Build a stacking meta-learner. logreg is the proven default; mlp adds non-linear
    capacity (the meta-features are only ~3*n_models dims so a small net + strong alpha).
    Neither MLP nor scaled-input models use class_weight — the downstream bias search
    (differential evolution on balanced accuracy) handles class balance instead."""
    if kind == "logreg":
        return LogisticRegression(
            C=C, max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=seed
        )
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


def load_stack(models: list[str], which: str) -> np.ndarray:
    """Concatenate log-prob features for the given models. which in {'oof','test'}."""
    mats = []
    for m in models:
        path = ARTIFACTS / f"{which}_{m}.npy"
        arr = np.load(path)
        mats.append(_logp(arr))
    return np.concatenate(mats, axis=1)


def search_bias(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    logp = _logp(proba)
    res = differential_evolution(
        lambda b: -balanced_accuracy_score(y, (logp + b).argmax(1)),
        bounds=[(-1.0, 1.0)] * proba.shape[1],
        seed=42, tol=1e-6, maxiter=100, polish=True,
    )
    return res.x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LogReg stacking meta-learner for S6E6")
    p.add_argument("--models", default="lgb,xgb,cat,hgb,logreg")
    p.add_argument("--meta", choices=["logreg", "mlp"], default="logreg",
                   help="Meta-learner family (logreg=proven default, mlp=non-linear)")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--output", type=Path, help="Write a submission CSV from the full-data meta-fit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    y = np.load(ARTIFACTS / "y_train.npy")
    n_classes = len(CLASS_ORDER)

    X = load_stack(models, "oof")
    print(f"Stacking {models} -> meta-features {X.shape}  (meta={args.meta})")

    meta_oof = np.zeros((len(y), n_classes))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=args.seed)
    for tr, va in skf.split(X, y):
        meta = build_meta(args.meta, args.seed, args.C)
        meta.fit(X[tr], y[tr])
        meta_oof[va] = meta.predict_proba(X[va])

    ba_before = balanced_accuracy_score(y, meta_oof.argmax(1))
    bias = search_bias(y, meta_oof)
    ba_after = balanced_accuracy_score(y, (_logp(meta_oof) + bias).argmax(1))
    recalls = {CLASS_ORDER[c]: round(float((( _logp(meta_oof)+bias).argmax(1)[y == c] == c).mean()), 4)
               for c in range(n_classes)}
    print(f"Meta-OOF BA before bias: {ba_before:.5f}")
    print(f"Meta-OOF BA after  bias: {ba_after:.5f}  recalls={recalls}")
    print(f"  class bias: {dict(zip(CLASS_ORDER, np.round(bias, 4)))}")
    print(f"  (flat-blend best to beat: 0.96609)")

    if args.output:
        _, test, _ = load_data(DATA)
        Xt = load_stack(models, "test")
        meta_full = build_meta(args.meta, args.seed, args.C)
        meta_full.fit(X, y)
        test_proba = meta_full.predict_proba(Xt)
        pred = (_logp(test_proba) + bias).argmax(1)
        out = args.output if args.output.is_absolute() else (SUBMISSIONS.parent / args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: [CLASS_ORDER[i] for i in pred]}).to_csv(out, index=False)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
