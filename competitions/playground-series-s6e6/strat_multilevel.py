"""STRATEGY 2 - multi-level / meta-of-metas ensembling (nested-CV evaluated).

We have three level-1 metas over the strong 24-model base set, each with a different
error profile:

  * LIN  - multinomial LogReg on one-vs-rest logit features (StandardScaler, C=3).
           Project reference nested-CV ~0.96954.
  * GBDT - LightGBM multiclass on log-prob features, class-balanced.
           Project reference nested-CV ~0.97022 (the best single meta).
  * HC   - Caruana bagged hill-climb (weighted average of base OOF probabilities,
           optimizing balanced accuracy).  Project reference nested-CV ~0.97002.

This script stacks those three metas at LEVEL 2 (simple average, plus a ridge/logreg
over the 3 meta-OOFs) and reports the HONEST nested-CV balanced accuracy with FULL
nesting -- the outer-test fold is never used to fit a level-1 meta, the level-2
blender, or the calibration.  We also print the naive whole-OOF number (optimistic)
for comparison, and try additive-bias vs per-class scale+shift calibration on the
level-2 OOF.

Honesty contract (matches the project's other tools):
  * Base OOF arrays are precomputed with StratifiedKFold(5, shuffle, random_state=42)
    on integer y in CSV order (artifacts/oof_<m>.npy, test_<m>.npy).
  * OUTER StratifiedKFold(5, seed=42).  Per outer fold:
      - INNER StratifiedKFold(n_inner, seed=43) over OUTER-TRAIN only -> level-1
        meta-OOF used to (a) fit the level-2 blender and (b) fit the calibration.
      - each level-1 meta is RE-FIT on the full outer-train, then applied to the
        untouched outer-test -> level-1 features for the outer-test rows.
      - level-2 blender + calibration applied to outer-test.  Outer-test rows never
        influence any fit.
  * Standalone nested-CV of each level-1 meta is recomputed in the SAME run, so the
    level-2 vs single-meta comparison is apples-to-apples regardless of exact config.

Self-checks: the printed standalone nested-CVs should land near the project refs
(0.96954 / 0.97022 / 0.97002); if so the configs are faithful and the level-2 number
is trustworthy.

Outputs (distinct to this strategy -- does NOT touch shared tools or other strats):
  * submissions/strat_multilevel.csv   (best level-2 config, full-data refit; NOT submitted)
  * level-1 meta-OOF/test cached under scratchpad for fast reruns.

Usage:
    python strat_multilevel.py                 # full run (nested-CV + submission)
    python strat_multilevel.py --no-submit     # skip writing the CSV
    python strat_multilevel.py --inner 3 --bags 6
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from config import ARTIFACTS, CLASS_ORDER, DATA, ID_COL, N_SPLITS, RANDOM_STATE, SUBMISSIONS, TARGET_COL

EPS = 1e-15
NC = len(CLASS_ORDER)

# The strong 24-model base set (identical to meta_gbdt.py).
MODELS = ("lgb_multi,xgb_multi,cat_multi,hgb_multi,logreg_multi,mlp_multi,specialist,knn_multi,realmlp,"
          "realmlp5,realmlp5b,realmlp5c,nn2,nn2b,tabm,lgb_orig,xgb_orig,cat_orig,catv3,xgbv5,lgbmv3,"
          "ovrxgb,ovrcat,ovrcatb").split(",")

# GBDT level-1 meta config (= meta_gbdt.py "medreg", which gives the project's best
# single-meta nested-CV ~0.97022).
GBDT_CFG = dict(n_est=500, lr=0.02, leaves=15, mcs=400, ss=0.7, cs=0.5, l2=8.0)

CACHE = Path(os.environ.get(
    "STRAT_ML_CACHE",
    "/private/tmp/claude-501/-Users-xingyuanxue1122-Documents-coding-kaggle-agent/"
    "d74e21b6-1de8-4d75-a32e-7175ddd1628f/scratchpad/strat_multilevel_cache",
))


# --------------------------------------------------------------------------- #
# Fast primitives
# --------------------------------------------------------------------------- #
def _logp(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1 - EPS))


def prob_to_logit(p: np.ndarray, clip: float = 30.0) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-7, 1.0 - 1e-7)
    return np.clip(np.log(p / (1.0 - p)), -clip, clip)


def fast_ba(y: np.ndarray, pred: np.ndarray) -> float:
    """Balanced accuracy (macro recall) for integer labels -- ~10x faster than sklearn."""
    correct = np.bincount(y[pred == y], minlength=NC).astype(np.float64)
    total = np.bincount(y, minlength=NC).astype(np.float64)
    return float(np.mean(correct / np.maximum(total, 1.0)))


# --------------------------------------------------------------------------- #
# Level-1 metas
# --------------------------------------------------------------------------- #
def build_lin(seed: int = RANDOM_STATE) -> object:
    """LogReg on logit features (StandardScaler, C=3). max_iter/tol trimmed for speed;
    argmax predictions are converged well before the 4000-iter cap."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=3.0, max_iter=400, tol=1e-3, class_weight="balanced",
                           n_jobs=1, random_state=seed),
    )


def build_gbdt(seed: int = RANDOM_STATE):
    import lightgbm as lgb
    c = GBDT_CFG
    return lgb.LGBMClassifier(
        objective="multiclass", num_class=NC, n_estimators=c["n_est"], learning_rate=c["lr"],
        num_leaves=c["leaves"], min_child_samples=c["mcs"], subsample=c["ss"], subsample_freq=1,
        colsample_bytree=c["cs"], reg_lambda=c["l2"], random_state=seed, n_jobs=-1, verbose=-1,
    )


def _climb(lib: np.ndarray, y: np.ndarray, iters: int) -> np.ndarray:
    """Greedy Caruana hill-climb (max balanced accuracy), fast-BA inner loop."""
    nm = lib.shape[0]
    counts = np.zeros(nm)
    model_ba = np.array([fast_ba(y, lib[i].argmax(1)) for i in range(nm)])
    rs = lib[int(model_ba.argmax())].copy()
    counts[int(model_ba.argmax())] += 1
    nsel = 1
    for _ in range(iters):
        cur = fast_ba(y, rs.argmax(1))
        best_i, best_ba = -1, -np.inf
        for i in range(nm):
            cba = fast_ba(y, (rs + lib[i]).argmax(1))
            if cba > best_ba:
                best_ba, best_i = cba, i
        if best_ba <= cur:
            break
        counts[best_i] += 1
        rs += lib[best_i]
        nsel += 1
    return counts


def bagged_climb(lib: np.ndarray, y: np.ndarray, bags: int, iters: int, seed: int) -> np.ndarray:
    """Average Caruana climb weights over `bags` bootstraps of the model library."""
    nm = lib.shape[0]
    rng = np.random.default_rng(seed)
    agg = np.zeros(nm)
    for _ in range(max(1, bags)):
        si = rng.integers(0, nm, size=nm)
        sc = _climb(lib[si], y, iters)
        for slot, orig in enumerate(si):
            agg[orig] += sc[slot]
    tot = agg.sum()
    return np.ones(nm) / nm if tot <= 0 else agg / tot


def weighted_proba(lib: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.tensordot(w, lib, axes=(0, 0))


# --------------------------------------------------------------------------- #
# Calibration (fit on held-out OOF, fast-BA objective)
# --------------------------------------------------------------------------- #
def fit_bias(y: np.ndarray, proba: np.ndarray, seed: int = RANDOM_STATE, maxiter: int = 80):
    logp = _logp(proba)
    res = differential_evolution(
        lambda b: -fast_ba(y, (logp + b).argmax(1)),
        bounds=[(-1.0, 1.0)] * NC, seed=seed, tol=1e-6, maxiter=maxiter, polish=True)
    return ("bias", res.x)


def fit_scale_shift(y: np.ndarray, proba: np.ndarray, seed: int = RANDOM_STATE, maxiter: int = 120):
    logp = _logp(proba)
    res = differential_evolution(
        lambda p: -fast_ba(y, (p[:NC] * logp + p[NC:]).argmax(1)),
        bounds=[(0.2, 3.0)] * NC + [(-2.0, 2.0)] * NC, seed=seed, tol=1e-6, maxiter=maxiter, polish=True)
    return ("scale_shift", res.x)


def apply_calib(kp, proba: np.ndarray) -> np.ndarray:
    kind, p = kp
    logp = _logp(proba)
    if kind == "bias":
        return (logp + p).argmax(1)
    return (p[:NC] * logp + p[NC:]).argmax(1)


# --------------------------------------------------------------------------- #
# Level-2 blenders over the 3 meta probability arrays
# --------------------------------------------------------------------------- #
def l2_features(metas: list[np.ndarray]) -> np.ndarray:
    """Concatenated log-probs of the 3 metas -> (n, 9) features for the L2 learner."""
    return np.concatenate([_logp(m) for m in metas], axis=1).astype(np.float64)


def build_l2(kind: str, seed: int = RANDOM_STATE):
    """L2 learners. 'lr2' = logreg C=1; 'ridge' = strongly-regularized logreg (C=0.3,
    the multinomial ridge analogue). 'avg' is parameter-free (handled separately)."""
    C = 1.0 if kind == "lr2" else 0.3
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, tol=1e-4, class_weight="balanced",
                           n_jobs=1, random_state=seed),
    )


def l2_predict(kind: str, model, metas: list[np.ndarray]) -> np.ndarray:
    if kind == "avg":
        return np.mean(metas, axis=0)
    return model.predict_proba(l2_features(metas))


def l2_fit(kind: str, metas_tr: list[np.ndarray], y_tr: np.ndarray, seed: int = RANDOM_STATE):
    if kind == "avg":
        return None
    m = build_l2(kind, seed)
    m.fit(l2_features(metas_tr), y_tr)
    return m


# --------------------------------------------------------------------------- #
# Per-fold level-1 meta production (inner-OOF over train + refit -> test rows)
# --------------------------------------------------------------------------- #
def meta_oof_and_pred(name, tr, te, inner, y, *, logit, logp, lib, libte, bags, iters, seed):
    """Return (oof_on_tr[len(tr),NC], pred_on_te[len(te),NC]) for one level-1 meta.

    `tr`,`te` are absolute row indices into the OOF arrays. `inner` yields index pairs
    RELATIVE to `tr`. For the test rows we pass either absolute indices into the OOF
    arrays (lib==oof, libte==oof, nested-CV) or the genuine test library (submission /
    full-data path) -- the caller controls that via libte and te semantics.
    """
    oof_tr = np.zeros((len(tr), NC))
    ytr = y[tr]
    if name in ("lin", "gbdt"):
        feat = logit if name == "lin" else logp
        for itr_rel, iva_rel in inner:
            itr, iva = tr[itr_rel], tr[iva_rel]
            mdl = build_lin(seed) if name == "lin" else build_gbdt(seed)
            if name == "gbdt":
                mdl.fit(feat[itr], y[itr], sample_weight=compute_sample_weight("balanced", y[itr]))
            else:
                mdl.fit(feat[itr], y[itr])
            oof_tr[iva_rel] = mdl.predict_proba(feat[iva])
        mdl = build_lin(seed) if name == "lin" else build_gbdt(seed)
        if name == "gbdt":
            mdl.fit(feat[tr], ytr, sample_weight=compute_sample_weight("balanced", ytr))
        else:
            mdl.fit(feat[tr], ytr)
        pred_te = mdl.predict_proba(feat[te])
    elif name == "hc":
        for itr_rel, iva_rel in inner:
            itr, iva = tr[itr_rel], tr[iva_rel]
            w = bagged_climb(lib[:, itr], y[itr], bags=bags, iters=iters, seed=seed)
            oof_tr[iva_rel] = weighted_proba(lib[:, iva], w)
        w = bagged_climb(lib[:, tr], ytr, bags=bags, iters=iters, seed=seed)
        pred_te = weighted_proba(libte[:, te], w)
    else:
        raise ValueError(name)
    return oof_tr, pred_te


# --------------------------------------------------------------------------- #
# Full-data level-1 meta-OOF + test (cached) -- for naive numbers & submission
# --------------------------------------------------------------------------- #
def full_meta_oof_test(y, *, logit, logp, lib, logit_te, logp_te, lib_te, bags, iters, seed, n_inner):
    """Honest full-data level-1 meta-OOF (inner KFold) + full-data test predictions."""
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"i{n_inner}_b{bags}_it{iters}_s{seed}"
    metas_oof, metas_test = {}, {}
    skf = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(y)), y))
    for name in ("lin", "gbdt", "hc"):
        f_oof = CACHE / f"oof_{name}_{tag}.npy"
        f_te = CACHE / f"test_{name}_{tag}.npy"
        if f_oof.exists() and f_te.exists():
            metas_oof[name] = np.load(f_oof)
            metas_test[name] = np.load(f_te)
            print(f"  [cache] {name}: loaded", flush=True)
            continue
        t0 = time.time()
        oof = np.zeros((len(y), NC))
        if name in ("lin", "gbdt"):
            feat = logit if name == "lin" else logp
            feat_te = logit_te if name == "lin" else logp_te
            for tr_i, va_i in splits:
                mdl = build_lin(seed) if name == "lin" else build_gbdt(seed)
                if name == "gbdt":
                    mdl.fit(feat[tr_i], y[tr_i], sample_weight=compute_sample_weight("balanced", y[tr_i]))
                else:
                    mdl.fit(feat[tr_i], y[tr_i])
                oof[va_i] = mdl.predict_proba(feat[va_i])
            mdl = build_lin(seed) if name == "lin" else build_gbdt(seed)
            if name == "gbdt":
                mdl.fit(feat, y, sample_weight=compute_sample_weight("balanced", y))
            else:
                mdl.fit(feat, y)
            test = mdl.predict_proba(feat_te)
        else:
            for tr_i, va_i in splits:
                w = bagged_climb(lib[:, tr_i], y[tr_i], bags=bags, iters=iters, seed=seed)
                oof[va_i] = weighted_proba(lib[:, va_i], w)
            w = bagged_climb(lib, y, bags=bags, iters=iters, seed=seed)
            test = weighted_proba(lib_te, w)
        metas_oof[name] = oof
        metas_test[name] = test
        np.save(f_oof, oof)
        np.save(f_te, test)
        print(f"  {name}: full-data meta-OOF+test done ({time.time()-t0:.0f}s) "
              f"standalone OOF-BA={fast_ba(y, oof.argmax(1)):.5f}", flush=True)
    return metas_oof, metas_test


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Multi-level meta-of-metas stacking for S6E6")
    p.add_argument("--inner", type=int, default=3, help="Inner folds for level-1 meta-OOF (nested)")
    p.add_argument("--bags", type=int, default=6, help="Hill-climb bags (anti-overfit)")
    p.add_argument("--iters", type=int, default=40, help="Hill-climb max additions")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--no-submit", action="store_true", help="Skip writing the submission CSV")
    p.add_argument("--no-nested", action="store_true", help="Skip the (slow) nested-CV; naive only")
    p.add_argument("--models", type=str, default=None,
                   help="Comma-separated base model list (default: the 24-model set)")
    p.add_argument("--output", type=Path, default=SUBMISSIONS / "strat_multilevel.csv")
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    y = np.load(ARTIFACTS / "y_train.npy")
    n = len(y)
    models = args.models.split(",") if args.models else MODELS
    print(f"Loading {len(models)} base models  (n={n})", flush=True)
    oof = [np.load(ARTIFACTS / f"oof_{m}.npy").astype(np.float64) for m in models]
    test = [np.load(ARTIFACTS / f"test_{m}.npy").astype(np.float64) for m in models]
    logit = np.concatenate([prob_to_logit(o) for o in oof], axis=1).astype(np.float32)
    logp = np.concatenate([_logp(o) for o in oof], axis=1).astype(np.float32)
    logit_te = np.concatenate([prob_to_logit(o) for o in test], axis=1).astype(np.float32)
    logp_te = np.concatenate([_logp(o) for o in test], axis=1).astype(np.float32)
    lib = np.stack(oof, axis=0).astype(np.float32)
    lib_te = np.stack(test, axis=0).astype(np.float32)
    del oof, test

    META_NAMES = ("lin", "gbdt", "hc")
    L2_KINDS = ("avg", "lr2", "ridge")
    CALIBS = ("bias", "scale_shift")

    # ---- Full-data level-1 meta-OOF + test (cached) -> naive numbers + submission ---
    print("\n[1] Full-data level-1 meta-OOF + test predictions", flush=True)
    metas_oof, metas_test = full_meta_oof_test(
        y, logit=logit, logp=logp, lib=lib, logit_te=logit_te, logp_te=logp_te,
        lib_te=lib_te, bags=args.bags, iters=args.iters, seed=args.seed, n_inner=N_SPLITS)
    metas_full = [metas_oof[m] for m in META_NAMES]
    metas_full_te = [metas_test[m] for m in META_NAMES]

    # ---- Naive whole-OOF level-2 (optimistic): fit L2 + calib on full meta-OOF -----
    print("\n[2] NAIVE whole-OOF level-2 (optimistic; L2 & calib fit on same rows)", flush=True)
    naive = {}
    for kind in L2_KINDS:
        mdl = l2_fit(kind, metas_full, y, args.seed)
        proba = l2_predict(kind, mdl, metas_full)
        row = {}
        for cal in CALIBS:
            kp = fit_bias(y, proba, args.seed) if cal == "bias" else fit_scale_shift(y, proba, args.seed)
            row[cal] = fast_ba(y, apply_calib(kp, proba))
        naive[kind] = row
        print(f"  L2={kind:5s}  naive-OOF BA  bias={row['bias']:.5f}  scale_shift={row['scale_shift']:.5f}", flush=True)

    # ---- Honest nested-CV (full nesting) -------------------------------------------
    results = None
    standalone = None
    if not args.no_nested:
        print(f"\n[3] HONEST nested-CV (outer 5-fold seed={args.seed}, inner {args.inner}-fold; "
              f"level-1 refit per outer fold; no leakage)", flush=True)
        outer = StratifiedKFold(N_SPLITS, shuffle=True, random_state=args.seed)
        # accumulators of integer predictions over all outer-test rows
        l2_pred = {(k, c): np.zeros(n, dtype=int) for k in L2_KINDS for c in CALIBS}
        sa_pred = {m: np.zeros(n, dtype=int) for m in META_NAMES}  # standalone meta (bias calib)
        for fold, (tr, te) in enumerate(outer.split(np.zeros(n), y)):
            t0 = time.time()
            inner = list(StratifiedKFold(args.inner, shuffle=True, random_state=args.seed + 1)
                         .split(np.zeros(len(tr)), y[tr]))
            meta_tr, meta_te = {}, {}
            for name in META_NAMES:
                o_tr, p_te = meta_oof_and_pred(
                    name, tr, te, inner, y, logit=logit, logp=logp, lib=lib, libte=lib,
                    bags=args.bags, iters=args.iters, seed=args.seed)
                meta_tr[name], meta_te[name] = o_tr, p_te
                # standalone meta nested-CV (bias calib fit on inner-OOF of outer-train)
                kp = fit_bias(y[tr], o_tr, args.seed)
                sa_pred[name][te] = apply_calib(kp, p_te)
            metas_tr_list = [meta_tr[m] for m in META_NAMES]
            metas_te_list = [meta_te[m] for m in META_NAMES]
            for kind in L2_KINDS:
                mdl = l2_fit(kind, metas_tr_list, y[tr], args.seed)
                proba_tr = l2_predict(kind, mdl, metas_tr_list)   # in-sample on outer-train (for calib)
                proba_te = l2_predict(kind, mdl, metas_te_list)
                for cal in CALIBS:
                    kp = (fit_bias(y[tr], proba_tr, args.seed) if cal == "bias"
                          else fit_scale_shift(y[tr], proba_tr, args.seed))
                    l2_pred[(kind, cal)][te] = apply_calib(kp, proba_te)
            print(f"  outer fold {fold+1}/{N_SPLITS} done ({time.time()-t0:.0f}s)", flush=True)
        standalone = {m: fast_ba(y, sa_pred[m]) for m in META_NAMES}
        results = {(k, c): fast_ba(y, l2_pred[(k, c)]) for k in L2_KINDS for c in CALIBS}

    # ---- Report --------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("RESULTS  (balanced accuracy; HONEST = nested-CV, no leakage)")
    print("=" * 68)
    if standalone is not None:
        print("\nStandalone level-1 meta nested-CV (bias calib)  [project refs in brackets]:")
        refs = {"lin": 0.96954, "gbdt": 0.97022, "hc": 0.97002}
        for m in META_NAMES:
            print(f"  {m:5s}: {standalone[m]:.5f}   [ref {refs[m]:.5f}]")
        best_single = max(standalone.values())
        best_single_name = max(standalone, key=standalone.get)
    else:
        best_single, best_single_name = 0.97022, "gbdt(ref)"

    print("\nLevel-2 blends:")
    print(f"  {'L2':6s} {'calib':12s} {'naive-OOF':>10s} {'nested-CV(HONEST)':>18s}")
    best = (None, None, -1.0)
    for kind in L2_KINDS:
        for cal in CALIBS:
            nv = naive[kind][cal]
            nc = results[(kind, cal)] if results is not None else float("nan")
            print(f"  {kind:6s} {cal:12s} {nv:10.5f} {nc:18.5f}")
            if results is not None and nc > best[2]:
                best = (kind, cal, nc)

    print("\n" + "-" * 68)
    if results is not None:
        print(f"Best level-2 (honest nested-CV): L2={best[0]} calib={best[1]} -> {best[2]:.5f}")
        print(f"Best single level-1 meta (honest):     {best_single_name} -> {best_single:.5f}")
        print(f"Reference best single meta (project):  gbdt -> 0.97022")
        print(f"Reference hill-climb (project):        hc   -> 0.97002")
        delta = best[2] - max(best_single, 0.97022)
        verdict = ("BEATS" if delta > 0 else "does NOT beat") + " the best single meta"
        print(f"Level-2 vs best single meta: delta={delta:+.5f}  -> level-2 {verdict}.")
        print(f"NOTE: public-LB 1-sigma ~0.001-0.002; a delta below ~0.0007 is WITHIN NOISE "
              f"and unlikely to translate to LB / toward 0.972.")
    else:
        print("(nested-CV skipped: --no-nested)")

    # ---- Submission (best level-2, full-data refit) --------------------------------
    if not args.no_submit:
        # pick config by honest nested-CV if available, else by naive bias.
        if results is not None:
            kind, cal = best[0], best[1]
        else:
            kind, cal = "avg", "bias"
        print(f"\n[4] Writing submission from full-data refit: L2={kind} calib={cal}", flush=True)
        mdl = l2_fit(kind, metas_full, y, args.seed)
        proba_oof = l2_predict(kind, mdl, metas_full)
        kp = fit_bias(y, proba_oof, args.seed) if cal == "bias" else fit_scale_shift(y, proba_oof, args.seed)
        proba_te = l2_predict(kind, mdl, metas_full_te)
        pred = apply_calib(kp, proba_te)
        # ids from sample submission order
        try:
            sub = pd.read_csv(DATA / "sample_submission.csv")
            ids = sub[ID_COL]
        except Exception:
            test_df = pd.read_csv(DATA / "test.csv")
            ids = test_df[ID_COL]
        out = args.output if args.output.is_absolute() else (SUBMISSIONS.parent / args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({ID_COL: ids, TARGET_COL: [CLASS_ORDER[i] for i in pred]}).to_csv(out, index=False)
        print(f"  saved {out}  (label dist: "
              f"{dict(zip(CLASS_ORDER, np.bincount(pred, minlength=NC).tolist()))})", flush=True)

    print(f"\nTotal wall time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
