"""Quantify base-model correlation for a feature set: agreement, error Jaccard, oracle.

Tests whether adding the same features to all (correlated) GBDTs improves single models
without adding ensemble coverage. Reads OOF probability arrays already on disk.

Usage:
    python analyze_correlation.py            # all_v2 (live artifacts) vs all_v3 (snapshot)
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from config import ARTIFACTS, CLASS_ORDER, MODELS

Y = np.load(ARTIFACTS / "y_train.npy")


def _load_oof(folder: Path) -> dict[str, np.ndarray]:
    return {m: np.load(folder / f"oof_{m}.npy").argmax(1) for m in MODELS}


def analyze(name: str, folder: Path) -> dict:
    preds = _load_oof(folder)
    errors = {m: (preds[m] != Y) for m in MODELS}

    print(f"\n{'=' * 60}\n{name}  ({folder})\n{'=' * 60}")
    print("single-model balanced accuracy:")
    for m in MODELS:
        print(f"  {m}: BA={balanced_accuracy_score(Y, preds[m]):.5f}  err_rate={errors[m].mean():.4f}")

    print("pairwise agreement rate / error Jaccard:")
    agrees, jaccs = [], []
    for a, b in combinations(MODELS, 2):
        agree = (preds[a] == preds[b]).mean()
        inter = (errors[a] & errors[b]).sum()
        union = (errors[a] | errors[b]).sum()
        jac = inter / union if union else 0.0
        agrees.append(agree)
        jaccs.append(jac)
        print(f"  {a:>3}-{b:<3}  agree={agree:.4f}  err_jaccard={jac:.4f}")
    mean_agree, mean_jacc = float(np.mean(agrees)), float(np.mean(jaccs))
    print(f"  MEAN agree={mean_agree:.4f}  MEAN err_jaccard={mean_jacc:.4f}")

    # Oracle: any model correct. Report overall acc and balanced-accuracy ceiling.
    any_correct = np.zeros(len(Y), dtype=bool)
    for m in MODELS:
        any_correct |= (preds[m] == Y)
    oracle_acc = any_correct.mean()
    # oracle balanced accuracy: per-class recall where "correct" = any model right
    oracle_recalls = [any_correct[Y == c].mean() for c in range(len(CLASS_ORDER))]
    oracle_ba = float(np.mean(oracle_recalls))
    print(f"oracle (any-model-correct): acc={oracle_acc:.5f}  balanced_acc={oracle_ba:.5f}")
    print("  oracle per-class recall:", {CLASS_ORDER[c]: round(oracle_recalls[c], 4) for c in range(len(CLASS_ORDER))})

    return dict(mean_agree=mean_agree, mean_jacc=mean_jacc, oracle_acc=float(oracle_acc), oracle_ba=oracle_ba)


def main() -> None:
    results = {}
    results["all_v2"] = analyze("all_v2 (live)", ARTIFACTS)
    snap = ARTIFACTS / "snapshot_seed42_all_v3"
    if snap.exists():
        results["all_v3"] = analyze("all_v3 (snapshot)", snap)

    if "all_v3" in results:
        v2, v3 = results["all_v2"], results["all_v3"]
        print(f"\n{'=' * 60}\nDELTA  all_v3 - all_v2\n{'=' * 60}")
        print(f"  mean agreement : {v3['mean_agree'] - v2['mean_agree']:+.4f}  "
              f"(higher => models MORE similar)")
        print(f"  mean err Jaccard: {v3['mean_jacc'] - v2['mean_jacc']:+.4f}  "
              f"(higher => errors MORE correlated)")
        print(f"  oracle acc      : {v3['oracle_acc'] - v2['oracle_acc']:+.5f}  "
              f"(<=0 => no new coverage despite better single models)")
        print(f"  oracle bal acc  : {v3['oracle_ba'] - v2['oracle_ba']:+.5f}")


if __name__ == "__main__":
    main()
