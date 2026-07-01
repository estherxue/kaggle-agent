"""Kaggle kernel: train 4 GBDTs (lgb/xgb/cat/hgb) on seed 42."""
import os, shutil, sys

# Copy pipeline code from dataset into working dir so imports resolve
CODE = "/kaggle/input/s6e6-pipeline-code"
for f in os.listdir(CODE):
    if f.endswith(".py"):
        shutil.copy(os.path.join(CODE, f), f"/kaggle/working/{f}")

sys.path.insert(0, "/kaggle/working")
os.chdir("/kaggle/working")

# Debug: show what competition data is available
print("=== /kaggle/input/ contents ===")
for p in sorted(os.listdir("/kaggle/input")):
    print(" ", p, "->", os.listdir(f"/kaggle/input/{p}")[:5])

# Symlink competition data to ./data
# Competition data is mounted at /kaggle/input/competitions/<slug>/
COMP_DATA = "/kaggle/input/competitions/playground-series-s6e6"
if not os.path.exists("data"):
    os.symlink(COMP_DATA, "data")
    print(f"Linked {COMP_DATA} -> data")

os.makedirs("artifacts", exist_ok=True)
print("data/ contents:", os.listdir("data")[:5])

from train_oof import ALL_MODELS, TrainConfig, run_oof

cfg = TrainConfig(
    feature_set="all_v2",
    models=tuple(ALL_MODELS),
    seed=42,
    n_estimators=2000,
    learning_rate=0.05,
    cat_learning_rate=0.05,
    cat_depth=8,
    cat_iterations=1500,
)
run_oof(cfg)
print("Done. artifacts:", sorted(os.listdir("artifacts")))
