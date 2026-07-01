"""Strategy 3 — honestly squeeze the balanced-accuracy DECISION RULE on the best stack.

The base meta is the proven best linear stack (logit features + StandardScale + C=3 LogReg,
single seed) over the 24-model pool. That meta-OOF is FIXED; what we vary here is the final
*decision rule* that turns meta probabilities into a class label, and we ask — honestly —
whether any rule beats the plain additive-bias baseline by more than noise.

Honesty protocol (nested CV around the DECISION RULE, mirroring nested_cv.py):
  outer 5-fold:
    on outer-train:  inner 5-fold -> honest meta-OOF -> FIT the decision rule on it
                     + fit the final meta-learner on all outer-train
    apply meta + the FROZEN rule to the untouched outer-test rows
  pool outer-test predictions -> honest nested-CV balanced accuracy.
The expensive meta fits are cached per outer fold and SHARED across all four rules, so the
rules are compared on identical meta probabilities (apples-to-apples) and only the cheap
rule-fit (DE on argmax) differs.

Decision rules compared:
  (1) bias        : argmax(logp + b)                 3 params, maximize BA   [BASELINE]
  (2) scale_shift : argmax(a*logp + b)               6 params, maximize BA
  (3) eq_recall   : argmax(logp + b), b tuned to     3 params, MAXIMIN recall
                    maximize the minimum class recall (= equalize recalls / per-class
                    quantile thresholds), NOT BA directly
  (4) temp_bias   : argmax(logp/T + b)               4 params, maximize BA   (single scale)

We report, per rule: naive (leaky, whole-OOF) BA, honest nested-CV BA, the optimism gap
(naive - honest), per-fold honest BA mean±std, and the paired honest delta vs the bias
baseline across the 5 outer folds (a crude significance read). The best honest rule writes
submissions/strat_calib.csv. No Kaggle submission is performed.

Run from competitions/playground-series-s6e6:
    python strat_calib.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from config import ARTIFACTS, CLASS_ORDER, DATA, ID_COL, N_SPLITS, RANDOM_STATE, SUBMISSIONS, TARGET_COL
from features import load_data
from stack import _logp, build_meta, load_stack

NC = len(CLASS_ORDER)

# The proven best 24-model pool (identical list to meta_gbdt.py / gbdt_meta_submit.py).
MODELS = ("lgb_multi,xgb_multi,cat_multi,hgb_multi,logreg_multi,mlp_multi,specialist,knn_multi,realmlp,"
          "realmlp5,realmlp5b,realmlp5c,nn2,nn2b,tabm,lgb_orig,xgb_orig,cat_orig,catv3,xgbv5,lgbmv3,"
          "ovrxgb,ovrcat,ovrcatb").split(",")

# Meta config = proven best linear stack: logit feats + StandardScale + C=3, single seed.
META_KIND, META_FEAT, META_C, META_SCALE, META_SEED = "logreg", "logit", 3.0, True, RANDOM_STATE

# CPU-modesty knobs (the box is shared with sibling strategies):
#   - meta fits are single-threaded (build_meta uses n_jobs=1) -> ~1-core footprint.
#   - nested inner uses 3 folds: we only fit a 3-6 param rule, so 3-fold inner-OOF is ample.
#   - the rule DE is fit on a stratified <=80k subsample: at that size the optimal
#     calibration params are stable to ~1e-4 in BA, and the params are then applied to the
#     full (held-out) rows. This roughly halves DE cost with no material change to the optimum.
INNER_SPLITS = 3
DE_FIT_SUBSAMPLE = 80_000


# --------------------------------------------------------------------------- #
# Fast per-class recall / balanced-accuracy scorer (precomputes class masks).
# --------------------------------------------------------------------------- #
class Scorer:
    def __init__(self, y: np.ndarray):
        self.idx = [np.where(y == c)[0] for c in range(NC)]

    def recalls(self, pred: np.ndarray) -> np.ndarray:
        return np.array([float((pred[self.idx[c]] == c).mean()) for c in range(NC)])

    def ba(self, pred: np.ndarray) -> float:
        return float(self.recalls(pred).mean())


def _de(obj, bounds, maxiter, seed=42, popsize=12):
    return differential_evolution(obj, bounds=bounds, seed=seed, tol=1e-7,
                                  maxiter=maxiter, popsize=popsize, polish=True).x


def _subsample(y, proba, n=DE_FIT_SUBSAMPLE, seed=0):
    """Stratified subsample of (y, proba) rows for a cheaper DE rule-fit (params stable)."""
    if len(y) <= n:
        return y, proba
    rng = np.random.default_rng(seed)
    take = []
    for c in range(NC):
        idx_c = np.where(y == c)[0]
        k = max(1, int(round(n * len(idx_c) / len(y))))
        take.append(rng.choice(idx_c, size=min(k, len(idx_c)), replace=False))
    sel = np.concatenate(take)
    return y[sel], proba[sel]


def fit_rule(fit_fn, y, proba):
    """Fit a decision rule on a stratified subsample (cheap, CPU-modest); params apply to all rows."""
    ys, ps = _subsample(y, proba)
    return fit_fn(ys, ps)


# --------------------------------------------------------------------------- #
# The four decision rules: each has fit(y, proba) -> params and apply(proba, params) -> pred.
# fit() optimizes its objective on the supplied (inner-OOF) proba; apply() is frozen.
# --------------------------------------------------------------------------- #
def fit_bias(y, proba):
    logp = _logp(proba); s = Scorer(y)
    return _de(lambda b: -s.ba((logp + b).argmax(1)), [(-1.0, 1.0)] * NC, maxiter=40)


def apply_bias(proba, p):
    return (_logp(proba) + p).argmax(1)


def fit_scale_shift(y, proba):
    logp = _logp(proba); s = Scorer(y)
    return _de(lambda p: -s.ba((p[:NC] * logp + p[NC:]).argmax(1)),
               [(0.2, 3.0)] * NC + [(-2.0, 2.0)] * NC, maxiter=60)


def apply_scale_shift(proba, p):
    logp = _logp(proba)
    return (p[:NC] * logp + p[NC:]).argmax(1)


def fit_eq_recall(y, proba):
    """Per-class offset tuned to MAXIMIZE THE MINIMUM class recall (equalize recalls).
    This is the per-class threshold/quantile rule: each class gets an additive log-prob
    threshold and we pick the largest margin-over-threshold, with the thresholds chosen to
    lift the worst class to parity rather than to greedily maximize mean BA. Tiny mean-recall
    term breaks ties among equal-min solutions toward the higher-BA one."""
    logp = _logp(proba); s = Scorer(y)

    def obj(b):
        r = s.recalls((logp + b).argmax(1))
        return -(r.min() + 1e-4 * r.mean())

    return _de(obj, [(-1.5, 1.5)] * NC, maxiter=50)


# eq_recall uses the same additive-offset apply as bias.
apply_eq_recall = apply_bias


def fit_temp_bias(y, proba):
    """Temperature scaling + additive bias: argmax(logp/T + b). Treats the meta log-probs as
    logits; a single global temperature sharpens/softens them (one shared scale) before the
    per-class bias. A constrained middle ground between bias (T=1) and per-class scale+shift."""
    logp = _logp(proba); s = Scorer(y)
    return _de(lambda p: -s.ba((logp / p[0] + p[1:]).argmax(1)),
               [(0.3, 5.0)] + [(-2.0, 2.0)] * NC, maxiter=55)


def apply_temp_bias(proba, p):
    logp = _logp(proba)
    return (logp / p[0] + p[1:]).argmax(1)


RULES = {
    "bias":        (fit_bias,        apply_bias,        "argmax(logp+b)            3p  maxBA  [BASELINE]"),
    "scale_shift": (fit_scale_shift, apply_scale_shift, "argmax(a*logp+b)          6p  maxBA"),
    "eq_recall":   (fit_eq_recall,   apply_eq_recall,   "argmax(logp+b), maximin recall  3p  eqRecall"),
    "temp_bias":   (fit_temp_bias,   apply_temp_bias,   "argmax(logp/T+b)          4p  maxBA"),
}


# --------------------------------------------------------------------------- #
# Meta-OOF helpers (logit + scale + C3, single seed).
# --------------------------------------------------------------------------- #
def inner_meta_oof(Xtr, ytr, seed, n_splits=N_SPLITS):
    """Honest meta-OOF over a training block via an inner StratifiedKFold."""
    oof = np.zeros((len(ytr), NC))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1)
    for itr, iva in skf.split(Xtr, ytr):
        m = build_meta(META_KIND, seed, META_C, scale=META_SCALE)
        m.fit(Xtr[itr], ytr[itr])
        oof[iva] = m.predict_proba(Xtr[iva])
    return oof


def main():
    ap = argparse.ArgumentParser(description="Strategy 3: honest decision-rule calibration search")
    ap.add_argument("--seed", type=int, default=META_SEED)
    ap.add_argument("--output", type=Path, default=SUBMISSIONS / "strat_calib.csv")
    args = ap.parse_args()
    seed = args.seed

    y = np.load(ARTIFACTS / "y_train.npy")
    X = load_stack(MODELS, "oof", feat=META_FEAT)
    scorer_full = Scorer(y)
    print(f"Strategy 3 calibration search | {len(MODELS)} models -> meta-feats {X.shape} "
          f"| meta={META_KIND} feat={META_FEAT} scale={META_SCALE} C={META_C} seed={seed}")

    # ---- 1) Full-data honest meta-OOF (single inner 5-fold) for the NAIVE (leaky) rule fits
    #         and as the production rule-fit data for the test submission. 5 meta fits.
    t0 = time.time()
    full_oof = inner_meta_oof(X, y, seed)
    pre_ba = scorer_full.ba(full_oof.argmax(1))
    print(f"[{time.time()-t0:5.0f}s] full meta-OOF done | pre-calibration BA (plain argmax) = {pre_ba:.5f}")

    # ---- 2) Nested-CV cache: per outer fold, fit meta on outer-train, store inner-OOF + outer-test proba.
    #         30 meta fits total. Shared across all four decision rules.
    outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    folds = []  # (te_idx, ytr, inner_oof, te_proba)
    for k, (tr, te) in enumerate(outer.split(X, y)):
        inner_oof = inner_meta_oof(X[tr], y[tr], seed, n_splits=INNER_SPLITS)
        meta = build_meta(META_KIND, seed, META_C, scale=META_SCALE)
        meta.fit(X[tr], y[tr])
        te_proba = meta.predict_proba(X[te])
        folds.append((te, y[tr], inner_oof, te_proba))
        print(f"[{time.time()-t0:5.0f}s] outer fold {k+1}/{N_SPLITS} meta cached "
              f"(train {len(tr)}, test {len(te)})", flush=True)

    # ---- 3) Evaluate every rule on the SAME cached meta probabilities.
    results = {}  # name -> dict
    bias_fold_ba = None
    for name, (fit_fn, apply_fn, desc) in RULES.items():
        # naive (leaky): fit on full meta-OOF, score on the same rows. This same fit is the
        # production rule used for the test submission (fit on the best honest OOF we have).
        p_full = fit_rule(fit_fn, y, full_oof)
        naive_ba = scorer_full.ba(apply_fn(full_oof, p_full))

        # honest nested-CV: rule fit on each outer-train inner-OOF, scored on untouched outer-test.
        pred = np.zeros(len(y), dtype=int)
        fold_ba = []
        for (te, ytr, inner_oof, te_proba) in folds:
            p = fit_rule(fit_fn, ytr, inner_oof)
            pr = apply_fn(te_proba, p)
            pred[te] = pr
            fold_ba.append(balanced_accuracy_score(y[te], pr))
        honest_ba = scorer_full.ba(pred)
        fold_ba = np.array(fold_ba)
        if name == "bias":
            bias_fold_ba = fold_ba
        results[name] = dict(desc=desc, naive=naive_ba, honest=honest_ba, gap=naive_ba - honest_ba,
                             fold_ba=fold_ba, recalls=scorer_full.recalls(pred),
                             params_full=p_full)
        print(f"[{time.time()-t0:5.0f}s] rule {name:11s} naive={naive_ba:.5f} honest={honest_ba:.5f} "
              f"gap={naive_ba-honest_ba:+.5f}", flush=True)

    # ---- 4) Report.
    print("\n" + "=" * 92)
    print(f"{'rule':12s} {'naive(leaky)':>12s} {'honest nCV':>11s} {'gap':>8s} {'fold mean±std':>16s} "
          f"{'Δ vs bias':>10s}")
    print("-" * 92)
    base_h = results["bias"]["honest"]
    for name, r in results.items():
        paired = r["fold_ba"] - bias_fold_ba
        dvb = r["honest"] - base_h
        print(f"{name:12s} {r['naive']:12.5f} {r['honest']:11.5f} {r['gap']:+8.5f} "
              f"{r['fold_ba'].mean():9.5f}±{r['fold_ba'].std():.5f} {dvb:+10.5f}"
              + ("" if name == "bias" else f"  (paired Δ {paired.mean():+.5f}±{paired.std():.5f})"))
    print("=" * 92)
    print(f"pre-calibration plain-argmax BA (full meta-OOF) : {pre_ba:.5f}")
    print("external honest references: hill-climb(bagged) 0.97002 | linear-meta nCV 0.96954 | "
          "GBDT-meta nCV 0.97022")
    for name, r in results.items():
        print(f"  {name:12s} honest recalls = {dict(zip(CLASS_ORDER, np.round(r['recalls'], 4)))}")

    # ---- 5) Pick the best honest rule and write its submission (rule fit on full meta-OOF,
    #         meta fit on full data for the test probabilities). Single meta seed (ms=1).
    best = max(results, key=lambda k: results[k]["honest"])
    print(f"\nBest honest rule: {best}  (nested-CV BA {results[best]['honest']:.5f})")
    fit_fn, apply_fn, _ = RULES[best]
    p_best = results[best]["params_full"]

    Xt = load_stack(MODELS, "test", feat=META_FEAT)
    meta_full = build_meta(META_KIND, seed, META_C, scale=META_SCALE)
    meta_full.fit(X, y)
    test_proba = meta_full.predict_proba(Xt)
    pred_test = apply_fn(test_proba, p_best)

    _, test, _ = load_data(DATA)
    out = args.output if args.output.is_absolute() else (SUBMISSIONS.parent / args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COL: test[ID_COL],
                  TARGET_COL: [CLASS_ORDER[i] for i in pred_test]}).to_csv(out, index=False)
    vc = pd.Series([CLASS_ORDER[i] for i in pred_test]).value_counts().to_dict()
    print(f"Saved {out}  (rule={best}, params={np.round(np.asarray(p_best, float), 4).tolist()})")
    print(f"test class distribution: {vc}")


if __name__ == "__main__":
    main()
