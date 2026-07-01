"""Nested-CV honest evaluation of the stacking + class-calibration pipeline.

The plain stack.py fits the class bias via differential_evolution on the FULL meta-OOF and
then reports BA on that SAME data — optimistic (the calibration sees the rows it's scored on).
Here every test row is held out of BOTH the meta-learner fit AND the calibration fit:

  outer 5-fold:
    on outer-train: inner 5-fold -> honest meta-OOF -> fit class calibration (scale+shift)
                    + fit the final meta-learner on all outer-train
    apply meta + frozen calibration to outer-test  -> predictions never seen during fitting
  concat outer-test predictions -> honest full-OOF BA

This gives a generalization estimate that resists the noise-limited overfitting we hit at the
4th decimal. Compares additive-bias vs per-class scale+shift calibration.

Usage:
    python nested_cv.py --models lgb_multi,xgb_multi,cat_multi,hgb_multi,logreg_multi,mlp_multi,specialist
"""
from __future__ import annotations

import argparse
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from config import ARTIFACTS, CLASS_ORDER, N_SPLITS, RANDOM_STATE
from stack import build_meta, load_stack, _logp


def fit_additive_bias(y, proba):
    logp = _logp(proba)
    res = differential_evolution(
        lambda b: -balanced_accuracy_score(y, (logp + b).argmax(1)),
        bounds=[(-1.0, 1.0)] * proba.shape[1], seed=42, tol=1e-6, maxiter=100, polish=True)
    return ("bias", res.x)


def fit_scale_shift(y, proba):
    logp = _logp(proba); nc = proba.shape[1]
    res = differential_evolution(
        lambda p: -balanced_accuracy_score(y, (p[:nc] * logp + p[nc:]).argmax(1)),
        bounds=[(0.2, 3.0)] * nc + [(-2.0, 2.0)] * nc, seed=42, tol=1e-6, maxiter=200, polish=True)
    return ("scale_shift", res.x)


def apply_calib(kind_params, proba):
    kind, p = kind_params
    logp = _logp(proba); nc = proba.shape[1]
    if kind == "bias":
        return (logp + p).argmax(1)
    return (p[:nc] * logp + p[nc:]).argmax(1)


def inner_meta_oof(Xtr, ytr, meta_kind, seed):
    """Honest meta-OOF over the outer-train block via an inner StratifiedKFold."""
    oof = np.zeros((len(ytr), len(CLASS_ORDER)))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed + 1)
    for itr, iva in skf.split(Xtr, ytr):
        m = build_meta(meta_kind, seed, 1.0); m.fit(Xtr[itr], ytr[itr])
        oof[iva] = m.predict_proba(Xtr[iva])
    return oof


def nested_cv(X, y, meta_kind, calib_fit, seed=RANDOM_STATE):
    pred = np.zeros(len(y), dtype=int)
    outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for tr, te in outer.split(X, y):
        inner_oof = inner_meta_oof(X[tr], y[tr], meta_kind, seed)
        calib = calib_fit(y[tr], inner_oof)              # calibration fit on held-out inner-OOF
        meta = build_meta(meta_kind, seed, 1.0); meta.fit(X[tr], y[tr])
        pred[te] = apply_calib(calib, meta.predict_proba(X[te]))
    return balanced_accuracy_score(y, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--meta", choices=["logreg", "mlp"], default="logreg")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    y = np.load(ARTIFACTS / "y_train.npy")
    X = load_stack(models, "oof")
    print(f"Nested-CV  models={models}  meta={args.meta}  feats={X.shape[1]}")
    ba_bias = nested_cv(X, y, args.meta, fit_additive_bias, args.seed)
    ba_ss = nested_cv(X, y, args.meta, fit_scale_shift, args.seed)
    print(f"  nested-CV BA (additive bias)   : {ba_bias:.5f}")
    print(f"  nested-CV BA (scale+shift)     : {ba_ss:.5f}")


if __name__ == "__main__":
    main()
