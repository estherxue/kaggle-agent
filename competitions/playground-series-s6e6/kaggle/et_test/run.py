"""Remote kernel: train ExtraTrees and save oof_et.npy + test_et.npy.

Runs train_diverse.py --model et on Kaggle CPU.
Writes oof_et.npy and test_et.npy to /kaggle/working/ for easy retrieval.
"""
import os, shutil, sys

def find_input(slug, owner="cindyxue1122"):
    for cand in (f"/kaggle/input/{slug}", f"/kaggle/input/datasets/{owner}/{slug}"):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"{slug} not mounted; /kaggle/input={os.listdir('/kaggle/input')}")


CODE = find_input("s6e6-pipeline-code")
comp_data = None
for cand in ("/kaggle/input/competitions/playground-series-s6e6",
             "/kaggle/input/playground-series-s6e6"):
    if os.path.isdir(cand):
        comp_data = cand
        break
if comp_data is None:
    raise FileNotFoundError("competition data not found")
print(f"Competition data: {comp_data}, files: {os.listdir(comp_data)[:5]}")

for f in os.listdir(CODE):
    if f.endswith(".py"):
        shutil.copy(os.path.join(CODE, f), f"/kaggle/working/{f}")
sys.path.insert(0, "/kaggle/working")
os.chdir("/kaggle/working")

# Create a WRITABLE artifacts dir (no symlinks — train_diverse will write here)
os.makedirs("/kaggle/working/artifacts", exist_ok=True)

import config
from pathlib import Path
config.DATA = Path(comp_data)
config.ARTIFACTS = Path("/kaggle/working/artifacts")

import train_diverse
import numpy as np

print("Training ExtraTrees (5-fold)...")
ba = train_diverse.run_diverse("et", seed=42, feature_set="all_v2", out_name="et")
print(f"ET OOF BA: {ba:.5f}")

oof = np.load("/kaggle/working/artifacts/oof_et.npy")
test = np.load("/kaggle/working/artifacts/test_et.npy")
# Also copy to working root for easy kaggle kernels output retrieval
shutil.copy("/kaggle/working/artifacts/oof_et.npy", "/kaggle/working/oof_et.npy")
shutil.copy("/kaggle/working/artifacts/test_et.npy", "/kaggle/working/test_et.npy")
print(f"oof_et shape: {oof.shape}, test_et shape: {test.shape}")

with open("/kaggle/working/results.txt", "w") as f:
    f.write(f"ET OOF BA: {ba:.5f}\n")
    f.write(f"oof_et shape: {oof.shape}\n")
    f.write(f"test_et shape: {test.shape}\n")
print("Wrote results.txt")
