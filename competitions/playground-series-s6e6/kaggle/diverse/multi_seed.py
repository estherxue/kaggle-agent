"""Multi-seed OOF averaging to reduce prediction variance.

Runs the full base-model pipeline (4 GBDTs + logreg + mlp) for each of the seeds
[42, 2025, 3407], saves per-seed artifacts to artifacts/seed_{s}/, then averages
probabilities across seeds into oof_{m}_multi.npy / test_{m}_multi.npy.

Seed 42 is reused from existing live artifacts (no retraining needed).
New seeds use cat_iterations=1500 (early stopping is ineffective at 2000; ~25% faster).

Live artifacts (oof_{m}.npy) are restored to seed-42 values after the run so other
scripts (blend.py etc.) see a consistent baseline.

Usage:
    python multi_seed.py [--seeds 42,2025,3407] [--cat-iterations 1500]
    python multi_seed.py --seeds 2025,3407   # skip seed 42 re-copy if already done
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from config import ARTIFACTS, CLASS_ORDER, RANDOM_STATE
from train_diverse import run_diverse
from train_oof import ALL_MODELS, TrainConfig, run_oof

DIVERSE_MODELS = ["logreg", "mlp"]
ALL_STACKING_MODELS = list(ALL_MODELS) + DIVERSE_MODELS


def _seed_dir(seed: int) -> Path:
    d = ARTIFACTS / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _seed_done(seed: int) -> bool:
    d = ARTIFACTS / f"seed_{seed}"
    return all(
        (d / f"{which}_{m}.npy").exists()
        for m in ALL_STACKING_MODELS
        for which in ("oof", "test")
    )


def copy_to_seed_dir(seed: int) -> None:
    d = _seed_dir(seed)
    for m in ALL_STACKING_MODELS:
        for which in ("oof", "test"):
            src = ARTIFACTS / f"{which}_{m}.npy"
            if src.exists():
                shutil.copy2(src, d / f"{which}_{m}.npy")
    # also copy y_train for reference
    src_y = ARTIFACTS / "y_train.npy"
    if src_y.exists():
        shutil.copy2(src_y, d / "y_train.npy")
    print(f"  → copied artifacts to seed_{seed}/")


def restore_seed(seed: int) -> None:
    """Restore live artifacts/ from a saved seed dir (undoes overwrites by later seeds)."""
    d = ARTIFACTS / f"seed_{seed}"
    if not d.exists():
        return
    for m in ALL_STACKING_MODELS:
        for which in ("oof", "test"):
            src = d / f"{which}_{m}.npy"
            if src.exists():
                shutil.copy2(src, ARTIFACTS / f"{which}_{m}.npy")
    src_y = d / "y_train.npy"
    if src_y.exists():
        shutil.copy2(src_y, ARTIFACTS / "y_train.npy")
    print(f"Restored seed {seed} artifacts to artifacts/")


def run_seed(seed: int, cat_iterations: int) -> None:
    if _seed_done(seed):
        print(f"Seed {seed} already complete — skipping.")
        return

    if seed == RANDOM_STATE:
        # Seed 42: reuse existing live artifacts (avoid ~2h cat retraining).
        print(f"\n=== Seed {seed}: copying existing live artifacts ===")
        # Run diverse for seed 42 if not already saved to seed_42/
        d = _seed_dir(seed)
        for m_name in ["logreg", "mlp"]:
            src = ARTIFACTS / f"oof_{m_name}.npy"
            if not src.exists():
                print(f"  Running {m_name} for seed {seed}...")
                run_diverse(m_name, seed=seed)
        copy_to_seed_dir(seed)
        return

    print(f"\n{'=' * 60}")
    print(f"=== Seed {seed}: training GBDTs (cat_iterations={cat_iterations}) ===")
    print(f"{'=' * 60}")

    cfg = TrainConfig(
        feature_set="all_v2",
        models=tuple(ALL_MODELS),
        seed=seed,
        cat_iterations=cat_iterations,
        n_estimators=2000,
        learning_rate=0.05,
        cat_learning_rate=0.05,
        cat_depth=8,
    )
    run_oof(cfg)

    print(f"\n=== Seed {seed}: training diverse models ===")
    for m_name in DIVERSE_MODELS:
        print(f"  --- {m_name} seed={seed} ---")
        run_diverse(m_name, seed=seed)

    copy_to_seed_dir(seed)


def average_seeds(seeds: list[int]) -> dict[str, float]:
    """Average OOF/test probabilities across seeds; return per-model OOF BA."""
    y = np.load(ARTIFACTS / "y_train.npy")
    print(f"\n=== Averaging across seeds {seeds} ===")
    scores: dict[str, float] = {}
    for m in ALL_STACKING_MODELS:
        oof_arrays, test_arrays = [], []
        for s in seeds:
            d = ARTIFACTS / f"seed_{s}"
            oof_path = d / f"oof_{m}.npy"
            test_path = d / f"test_{m}.npy"
            if not (oof_path.exists() and test_path.exists()):
                print(f"  WARNING: missing seed_{s}/oof_{m}.npy — skipping this seed for {m}")
                continue
            oof_arrays.append(np.load(oof_path))
            test_arrays.append(np.load(test_path))

        if not oof_arrays:
            print(f"  {m}: no seed data found, skipping")
            continue

        avg_oof = np.mean(oof_arrays, axis=0)
        avg_test = np.mean(test_arrays, axis=0)
        np.save(ARTIFACTS / f"oof_{m}_multi.npy", avg_oof)
        np.save(ARTIFACTS / f"test_{m}_multi.npy", avg_test)

        ba = balanced_accuracy_score(y, avg_oof.argmax(1))
        recalls = {
            CLASS_ORDER[c]: round(float((avg_oof.argmax(1)[y == c] == c).mean()), 4)
            for c in range(len(CLASS_ORDER))
        }
        print(f"  {m}_multi (n_seeds={len(oof_arrays)}): OOF BA={ba:.5f}  recalls={recalls}")
        scores[m] = ba

    return scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-seed OOF averaging for S6E6")
    p.add_argument("--seeds", default="42,2025,3407", help="Comma-separated seeds to run")
    p.add_argument("--cat-iterations", type=int, default=1500,
                   help="CatBoost iteration cap for new seeds (early stopping ineffective at 2000)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    for seed in seeds:
        run_seed(seed, cat_iterations=args.cat_iterations)

    scores = average_seeds(seeds)

    # Restore seed 42 as the live artifacts so other scripts see a consistent baseline.
    if RANDOM_STATE in seeds:
        restore_seed(RANDOM_STATE)

    print(f"\n=== Multi-seed _multi artifacts ready ===")
    print("To stack multi-seed results, run:")
    models_str = ",".join(f"{m}_multi" for m in ALL_STACKING_MODELS)
    print(f"  python3 stack.py --models {models_str}")
    print("\nTo also include specialist:")
    print(f"  python3 stack.py --models {models_str},specialist")


if __name__ == "__main__":
    main()
