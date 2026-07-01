"""Per-seed stacking experiment: feed individual seed predictions to the meta-learner so it
can exploit cross-seed *stability* (variance) as an uncertainty signal, instead of averaging
seeds first (which hides that signal). Then layer two post-processing ideas:

  1. Calibration: generalize the additive class-bias to per-class scale+shift
     decision = argmax_c (a_c * logp_c + b_c)   (6 params vs the current 3-param bias)
  2. Anomaly patching: where the base models disagree across seeds/models (high variance,
     low margin), blend the meta probs toward a robust fallback (mean of base probs).

Metric is Balanced Accuracy (argmax-based). NOTE: a single global temperature is a no-op for
BA (monotonic, doesn't change argmax) — only *per-class* transforms matter, which is why the
bias search already helps. All variants are evaluated with the SAME full-OOF protocol as the
0.96708 baseline; richer parameter sets overfit more, so only trust sizable gains.

Run locally (stacking/post-processing only — no model training):
    python exp_perseed.py --meta logreg
    python exp_perseed.py --meta mlp
"""
from __future__ import annotations

import argparse
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from config import ARTIFACTS, CLASS_ORDER, N_SPLITS, RANDOM_STATE
from stack import build_meta, _logp

EPS = 1e-15
SEEDS = [42, 2025, 3407]
BASE = ["lgb", "xgb", "cat", "hgb", "logreg", "mlp"]
BASELINE = 0.96708


def load_perseed_logp() -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (meta_features, per_seed_prob_list).
    meta_features: concatenated log-probs of every (model, seed) + specialist.
    per_seed_prob_list: raw prob arrays per (model, seed) for disagreement computation.
    """
    mats, probs = [], []
    for m in BASE:
        for s in SEEDS:
            # flat per-seed aliases (oof_<m>_s<seed>.npy); local symlinks / dataset copies
            p = np.load(ARTIFACTS / f"oof_{m}_s{s}.npy")
            mats.append(_logp(p))
            probs.append(p)
    spec = np.load(ARTIFACTS / "oof_specialist.npy")
    mats.append(_logp(spec))
    probs.append(spec)
    return np.concatenate(mats, axis=1), probs


def meta_oof(X: np.ndarray, y: np.ndarray, kind: str, seed: int) -> np.ndarray:
    n_classes = len(CLASS_ORDER)
    out = np.zeros((len(y), n_classes))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        meta = build_meta(kind, seed, 1.0)
        meta.fit(X[tr], y[tr])
        out[va] = meta.predict_proba(X[va])
    return out


def search_bias(y, proba):
    logp = _logp(proba)
    res = differential_evolution(
        lambda b: -balanced_accuracy_score(y, (logp + b).argmax(1)),
        bounds=[(-1.0, 1.0)] * proba.shape[1], seed=42, tol=1e-6, maxiter=100, polish=True,
    )
    return res.x, -res.fun


def search_scale_shift(y, proba):
    """decision = argmax_c (a_c * logp_c + b_c). 6 params (a in [0.2,3], b in [-2,2])."""
    logp = _logp(proba)
    nc = proba.shape[1]

    def neg_ba(p):
        a, b = p[:nc], p[nc:]
        return -balanced_accuracy_score(y, (a * logp + b).argmax(1))

    res = differential_evolution(
        neg_ba, bounds=[(0.2, 3.0)] * nc + [(-2.0, 2.0)] * nc,
        seed=42, tol=1e-6, maxiter=200, polish=True,
    )
    return res.x, -res.fun


def disagreement(per_seed_probs: list[np.ndarray]) -> np.ndarray:
    """Mean per-sample variance of class probs across all base (model,seed) members."""
    stack = np.stack(per_seed_probs, axis=0)          # (M, N, C)
    return stack.var(axis=0).mean(axis=1)              # (N,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", choices=["logreg", "mlp"], default="logreg")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = ap.parse_args()

    y = np.load(ARTIFACTS / "y_train.npy")
    X, per_seed_probs = load_perseed_logp()
    n_members = len(per_seed_probs)
    print(f"Per-seed stack: {len(BASE)} models x {len(SEEDS)} seeds + specialist "
          f"= {n_members} members -> {X.shape[1]} meta-features (meta={args.meta})")

    mo = meta_oof(X, y, args.meta, args.seed)
    ba_raw = balanced_accuracy_score(y, mo.argmax(1))
    bias, ba_bias = search_bias(y, mo)
    print(f"\n[meta-OOF] raw BA           : {ba_raw:.5f}")
    print(f"[+additive bias (current)] : {ba_bias:.5f}   (baseline to beat {BASELINE})")

    _, ba_ss = search_scale_shift(y, mo)
    print(f"[+per-class scale+shift]   : {ba_ss:.5f}")

    # Anomaly patching: blend meta probs toward fallback (mean of base probs) for high-disagreement
    u = disagreement(per_seed_probs)
    fallback = np.mean(per_seed_probs, axis=0)
    best = (ba_bias, "none")
    print("\n[anomaly patching] meta<-fallback blend on high-disagreement tail:")
    for q in [0.80, 0.90, 0.95]:
        thr = np.quantile(u, q)
        mask = u >= thr
        for alpha in [0.3, 0.5, 0.7, 1.0]:
            patched = mo.copy()
            patched[mask] = (1 - alpha) * mo[mask] + alpha * fallback[mask]
            _, ba_p = search_bias(y, patched)
            tag = f"top{int((1-q)*100)}% a={alpha}"
            if ba_p > best[0]:
                best = (ba_p, tag)
            print(f"  q={q:.2f} thr={thr:.4f} n={mask.sum():>6} alpha={alpha}: BA={ba_p:.5f}")
    print(f"\nbest patching: {best[1]} -> {best[0]:.5f}")
    print(f"summary  raw={ba_raw:.5f}  bias={ba_bias:.5f}  scale+shift={ba_ss:.5f}  "
          f"patch={best[0]:.5f}  | baseline={BASELINE}")


if __name__ == "__main__":
    main()
