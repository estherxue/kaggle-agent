"""Kaggle kernel: multi-seed training (seeds 42 + 2025 + 3407)."""
import os, shutil, sys

CODE = "/kaggle/input/s6e6-pipeline-code"
for f in os.listdir(CODE):
    if f.endswith(".py"):
        shutil.copy(os.path.join(CODE, f), f"/kaggle/working/{f}")

sys.path.insert(0, "/kaggle/working")
os.chdir("/kaggle/working")

if not os.path.exists("data"):
    os.symlink("/kaggle/input/competitions/playground-series-s6e6", "data")
os.makedirs("artifacts", exist_ok=True)

from multi_seed import average_seeds, run_seed

SEEDS = [42, 2025, 3407]
for seed in SEEDS:
    print(f"\n{'='*60}\n=== Seed {seed} ===\n{'='*60}")
    run_seed(seed, cat_iterations=1500)

average_seeds(SEEDS)
print("Multi-seed artifacts:", [f for f in sorted(os.listdir("artifacts")) if "multi" in f])
