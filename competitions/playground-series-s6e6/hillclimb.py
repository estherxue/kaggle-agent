"""Caruana-style greedy hill-climbing ensemble selection for S6E6.

Greedily builds a weighted average of base-model OOF probabilities, optimizing
*balanced accuracy* of the argmax (not logloss, which is the competition metric).
Weights are integer counts (selection with replacement), as in Caruana et al.
"Ensemble Selection from Libraries of Models" (ICML 2004).

Greedy selection on a *single fixed OOF* overfits: the procedure can chase noise
in the held-out OOF rows because it is free to pick whichever combination happens
to score best on exactly those rows. This module therefore ships the mandatory
anti-overfit guards:

1. BAGGED selection -- run the climb on B bootstrap replicates of the *model
   library* and average the resulting integer-count weight vectors. This is the
   original Caruana bagging trick and it dramatically reduces selection variance.

2. NESTED CV for honest scoring -- an outer StratifiedKFold(5, seed=42); for each
   outer fold the bagged climb chooses weights using ONLY the outer-train OOF rows
   and is scored on the untouched outer-test rows. The averaged outer-test BA is
   THE honest number. We also print the naive whole-OOF climb BA (optimistic) and
   the gap between them. Weights are NEVER chosen using outer-test rows.

The final test prediction uses bagged-climb weights fit on the *full* OOF, applied
to the stacked test arrays, with an optional additive per-class bias tuned on OOF.

Artifact contract (see config.py / CLAUDE.md):
    artifacts/oof_<m>.npy   -> (577347, 3) probabilities
    artifacts/test_<m>.npy  -> (247435, 3) probabilities
    artifacts/y_train.npy   -> (577347,) int labels, GALAXY=0 / QSO=1 / STAR=2
    StratifiedKFold(5, shuffle=True, random_state=42) on integer y in CSV order.

Usage:
    python hillclimb.py --models lgb,xgb,cat,hgb
    python hillclimb.py --models lgb,xgb,cat,hgb --bags 20 --iters 100 \
        --output submissions/hc.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from config import ARTIFACTS, CLASS_ORDER, DATA, ID_COL, N_SPLITS, RANDOM_STATE, SUBMISSIONS, TARGET_COL

EPS = 1e-15


# --------------------------------------------------------------------------- #
# Loading / contract verification
# --------------------------------------------------------------------------- #
def load_library(models: list[str], which: str) -> np.ndarray:
    """Load and stack the OOF or test probability arrays for ``models``.

    Args:
        models: base-model names (resolve to ``artifacts/<which>_<m>.npy``).
        which: ``"oof"`` or ``"test"``.

    Returns:
        Array of shape ``(n_models, n_rows, n_classes)`` (float64).

    Raises:
        FileNotFoundError: a requested array is missing.
        ValueError: arrays disagree on shape or have the wrong class count.
    """
    n_classes = len(CLASS_ORDER)
    mats: list[np.ndarray] = []
    ref_rows: int | None = None
    for m in models:
        path = ARTIFACTS / f"{which}_{m}.npy"
        if not path.exists():
            raise FileNotFoundError(f"missing artifact: {path}")
        arr = np.load(path).astype(np.float64)
        if arr.ndim != 2 or arr.shape[1] != n_classes:
            raise ValueError(
                f"{path} has shape {arr.shape}, expected (n_rows, {n_classes})"
            )
        if ref_rows is None:
            ref_rows = arr.shape[0]
        elif arr.shape[0] != ref_rows:
            raise ValueError(
                f"{path} has {arr.shape[0]} rows, expected {ref_rows} (row order must match)"
            )
        mats.append(arr)
    return np.stack(mats, axis=0)


# --------------------------------------------------------------------------- #
# Core hill-climbing
# --------------------------------------------------------------------------- #
def _weighted_proba(library: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Weighted-average probabilities from integer counts.

    Args:
        library: ``(n_models, n_rows, n_classes)``.
        counts: ``(n_models,)`` non-negative integer (or float) weights.

    Returns:
        ``(n_rows, n_classes)`` normalized average. If all counts are 0,
        returns a uniform plain average (degenerate guard).
    """
    total = counts.sum()
    if total <= 0:
        w = np.ones(library.shape[0]) / library.shape[0]
    else:
        w = counts / total
    return np.tensordot(w, library, axes=(0, 0))


def _ba(y: np.ndarray, proba: np.ndarray) -> float:
    """Balanced accuracy of ``argmax(proba)`` against integer labels ``y``."""
    return balanced_accuracy_score(y, proba.argmax(axis=1))


def climb(
    library: np.ndarray,
    y: np.ndarray,
    iters: int = 100,
    init_best: bool = True,
) -> np.ndarray:
    """Greedy Caruana hill-climb maximizing balanced accuracy.

    Starts from the single best model (``init_best``), then greedily ADDS, with
    replacement, the one model whose inclusion most improves the balanced
    accuracy of the running weighted average. Stops at ``iters`` additions or
    when no single addition improves the score (early stop on no gain).

    Weights are integer counts: adding a model bumps its count by one. The final
    blend is ``counts / counts.sum()``.

    Args:
        library: ``(n_models, n_rows, n_classes)`` probability arrays.
        y: ``(n_rows,)`` integer labels.
        iters: maximum number of greedy additions.
        init_best: seed the ensemble with the single best model (counts=1 for it).

    Returns:
        ``counts`` of shape ``(n_models,)``, dtype float (integer-valued).

    Note:
        Maintains a running *sum* of selected-model probabilities so each
        candidate evaluation is O(n_rows * n_classes), not a full re-blend.
    """
    n_models = library.shape[0]
    counts = np.zeros(n_models, dtype=np.float64)

    # Per-model standalone score, used to pick the starting member.
    model_ba = np.array([_ba(y, library[i]) for i in range(n_models)])

    # running_sum = sum over selected slots of that model's proba (un-normalized).
    running_sum = np.zeros(library.shape[1:], dtype=np.float64)
    n_selected = 0

    if init_best:
        best = int(model_ba.argmax())
        counts[best] += 1.0
        running_sum += library[best]
        n_selected = 1

    for _ in range(iters):
        if n_selected >= iters:
            break
        best_gain_idx = -1
        best_gain_ba = -np.inf
        # Current score (avoid recompute when n_selected == 0).
        if n_selected == 0:
            cur_ba = -np.inf
        else:
            cur_ba = _ba(y, running_sum / n_selected)
        for i in range(n_models):
            cand_sum = running_sum + library[i]
            cand_ba = _ba(y, cand_sum / (n_selected + 1))
            if cand_ba > best_gain_ba:
                best_gain_ba = cand_ba
                best_gain_idx = i
        # Early stop: no addition strictly improves the running BA.
        if n_selected > 0 and best_gain_ba <= cur_ba:
            break
        counts[best_gain_idx] += 1.0
        running_sum += library[best_gain_idx]
        n_selected += 1

    return counts


def bagged_climb(
    library: np.ndarray,
    y: np.ndarray,
    bags: int = 20,
    iters: int = 100,
    seed: int = RANDOM_STATE,
) -> np.ndarray:
    """Caruana bagged selection: average climb weights over library bootstraps.

    Each bag draws a bootstrap sample of the MODELS (rows of ``library``, with
    replacement), runs :func:`climb` on that sub-library, then maps the resulting
    integer counts back onto the full model index space and accumulates them.
    Averaging the count vectors across bags is the anti-overfit step.

    Args:
        library: ``(n_models, n_rows, n_classes)``.
        y: ``(n_rows,)`` integer labels.
        bags: number of bootstrap replicates of the model library.
        iters: max greedy additions per climb.
        seed: RNG seed for the bootstrap draws.

    Returns:
        Averaged normalized weight vector ``(n_models,)`` summing to 1
        (uniform fallback if every bag produced empty counts).
    """
    n_models = library.shape[0]
    rng = np.random.default_rng(seed)
    agg = np.zeros(n_models, dtype=np.float64)
    for _ in range(max(1, bags)):
        sample_idx = rng.integers(0, n_models, size=n_models)
        sub_lib = library[sample_idx]
        sub_counts = climb(sub_lib, y, iters=iters, init_best=True)
        # Map bootstrap-slot counts back to original model indices.
        for slot, orig in enumerate(sample_idx):
            agg[orig] += sub_counts[slot]
    total = agg.sum()
    if total <= 0:
        return np.ones(n_models) / n_models
    return agg / total


# --------------------------------------------------------------------------- #
# Honest evaluation: nested CV
# --------------------------------------------------------------------------- #
def nested_cv_ba(
    library: np.ndarray,
    y: np.ndarray,
    bags: int = 20,
    iters: int = 100,
    n_splits: int = N_SPLITS,
    seed: int = RANDOM_STATE,
) -> float:
    """Honest balanced accuracy via nested CV around the bagged climb.

    Outer StratifiedKFold(``n_splits``, shuffle=True, random_state=``seed``).
    For each outer fold the bagged climb picks weights using ONLY the outer-TRAIN
    OOF rows, then those frozen weights are scored on the untouched outer-TEST
    rows. The mean of the per-fold outer-test balanced accuracies is returned.

    CRITICAL: outer-test rows never influence weight selection -> no leakage.

    Args:
        library: ``(n_models, n_rows, n_classes)``.
        y: ``(n_rows,)`` integer labels.
        bags: bags per inner bagged climb.
        iters: max greedy additions per climb.
        n_splits: outer fold count.
        seed: outer-split + bootstrap seed.

    Returns:
        Mean outer-test balanced accuracy (the honest number).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_scores: list[float] = []
    # Dummy X with the right row count for StratifiedKFold.split.
    X_dummy = np.zeros((library.shape[1], 1))
    for tr, te in skf.split(X_dummy, y):
        w = bagged_climb(library[:, tr], y[tr], bags=bags, iters=iters, seed=seed)
        te_proba = _weighted_proba(library[:, te], w)
        fold_scores.append(_ba(y[te], te_proba))
    return float(np.mean(fold_scores))


# --------------------------------------------------------------------------- #
# Class bias (additive log-space, optional)
# --------------------------------------------------------------------------- #
def _logp(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1 - EPS))


def search_class_bias(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Tune an additive per-class log-space bias to maximize balanced accuracy.

    Optimizes ``argmax(log(proba) + bias)`` over ``bias`` with SciPy's
    differential evolution (matching blend.py / stack.py conventions). If SciPy
    is unavailable, returns a zero bias.

    Args:
        y: ``(n_rows,)`` integer labels.
        proba: ``(n_rows, n_classes)`` probabilities.

    Returns:
        ``(n_classes,)`` additive bias vector.
    """
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        return np.zeros(proba.shape[1])
    logp = _logp(proba)
    res = differential_evolution(
        lambda b: -balanced_accuracy_score(y, (logp + b).argmax(1)),
        bounds=[(-1.0, 1.0)] * proba.shape[1],
        seed=RANDOM_STATE,
        tol=1e-6,
        maxiter=100,
        polish=True,
    )
    return res.x


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Caruana hill-climbing ensemble selection (balanced accuracy) for S6E6"
    )
    p.add_argument("--models", default=",".join(["lgb", "xgb", "cat", "hgb"]),
                   help="Comma-separated base-model names (artifacts/oof_<m>.npy)")
    p.add_argument("--bags", type=int, default=20,
                   help="Bootstrap replicates of the model library (anti-overfit)")
    p.add_argument("--iters", type=int, default=100,
                   help="Max greedy additions per climb")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--bias", action="store_true",
                   help="Also tune an additive per-class bias on the full OOF")
    p.add_argument("--output", type=Path,
                   help="Write a submission CSV from the full-OOF bagged-climb weights")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("no models specified")

    y = np.load(ARTIFACTS / "y_train.npy")
    oof_lib = load_library(models, "oof")
    print(f"Hill-climb over {models}  OOF library {oof_lib.shape}  "
          f"(bags={args.bags}, iters={args.iters})")

    # 1) Naive whole-OOF bagged climb (optimistic): weights chosen on the same
    #    rows we score on.
    naive_w = bagged_climb(oof_lib, y, bags=args.bags, iters=args.iters, seed=args.seed)
    naive_proba = _weighted_proba(oof_lib, naive_w)
    naive_ba = _ba(y, naive_proba)

    # 2) Honest nested-CV BA (weights never see the rows they're scored on).
    honest_ba = nested_cv_ba(
        oof_lib, y, bags=args.bags, iters=args.iters, n_splits=N_SPLITS, seed=args.seed
    )
    gap = naive_ba - honest_ba

    print("\nFinal bagged-climb weights (full OOF):")
    for name, w in zip(models, naive_w):
        print(f"  {name:>12s}: {w:.4f}")
    print(f"\nNaive whole-OOF BA   (optimistic): {naive_ba:.5f}")
    print(f"Nested-CV BA         (HONEST):     {honest_ba:.5f}")
    print(f"Optimism gap (naive - honest):     {gap:.5f}")
    print("NOTE: the final model pick MUST use the nested-CV (honest) number, "
          "not the naive whole-OOF score.")

    bias = np.zeros(len(CLASS_ORDER))
    if args.bias:
        bias = search_class_bias(y, naive_proba)
        biased_ba = balanced_accuracy_score(y, (_logp(naive_proba) + bias).argmax(1))
        print(f"\nClass bias {dict(zip(CLASS_ORDER, np.round(bias, 4)))}")
        print(f"Naive OOF BA after bias (optimistic): {biased_ba:.5f}")

    if args.output:
        try:
            from features import load_data
            _, test, _ = load_data(DATA)
            ids = test[ID_COL]
        except Exception as exc:  # pragma: no cover - data-dir absent in smoke test
            raise SystemExit(f"cannot load test ids for submission: {exc}")
        test_lib = load_library(models, "test")
        test_proba = _weighted_proba(test_lib, naive_w)
        pred = (_logp(test_proba) + bias).argmax(1)
        out = args.output if args.output.is_absolute() else (SUBMISSIONS.parent / args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        pd.DataFrame(
            {ID_COL: ids, TARGET_COL: [CLASS_ORDER[i] for i in pred]}
        ).to_csv(out, index=False)
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
