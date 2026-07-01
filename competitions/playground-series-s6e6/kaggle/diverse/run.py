"""Kaggle kernel: train non-GBDT diverse models + specialist."""
import os, shutil, sys

CODE = "/kaggle/input/s6e6-pipeline-code"
for f in os.listdir(CODE):
    if f.endswith(".py"):
        shutil.copy(os.path.join(CODE, f), f"/kaggle/working/{f}")

sys.path.insert(0, "/kaggle/working")
os.chdir("/kaggle/working")

print("=== /kaggle/input/ contents ===")
for p in sorted(os.listdir("/kaggle/input")):
    print(" ", p, "->", os.listdir(f"/kaggle/input/{p}")[:5])

COMP_DATA = "/kaggle/input/competitions/playground-series-s6e6"
if not os.path.exists("data"):
    os.symlink(COMP_DATA, "data")
    print(f"Linked {COMP_DATA} -> data")

os.makedirs("artifacts", exist_ok=True)
print("data/ contents:", os.listdir("data")[:5])

from train_diverse import run_diverse
from specialist import run_specialist

SEED = 42
FEATURE_SET = "all_v2"

for model in ["logreg", "mlp", "knn", "rf", "et"]:
    print(f"\n{'='*50}\nTraining {model} seed={SEED}\n{'='*50}")
    run_diverse(model, seed=SEED, feature_set=FEATURE_SET)

print(f"\n{'='*50}\nTraining specialist\n{'='*50}")
run_specialist(seed=SEED, feature_set=FEATURE_SET, out_name="specialist")
print("Done. artifacts:", sorted(os.listdir("artifacts")))
