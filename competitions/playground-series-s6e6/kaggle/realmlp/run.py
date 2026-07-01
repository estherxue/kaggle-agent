"""Exp4: RealMLP base model (5-fold OOF) — a PyTorch table-tuned MLP (pytabkit), a third
neural family distinct from GBDTs and TabPFN, to add ensemble coverage.

Fold split is StratifiedKFold(N_SPLITS, shuffle=True, random_state=42) — aligns with all
existing artifacts. Internet ON (pip install pytabkit). GPU.
"""
import os, shutil, sys, time, subprocess

def find_input(slug, owner="cindyxue1122"):
    for cand in (f"/kaggle/input/{slug}", f"/kaggle/input/datasets/{owner}/{slug}"):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"dataset {slug} not mounted; /kaggle/input = {os.listdir('/kaggle/input')}")


CODE = find_input("s6e6-pipeline-code")
for f in os.listdir(CODE):
    if f.endswith(".py"):
        shutil.copy(os.path.join(CODE, f), f"/kaggle/working/{f}")
sys.path.insert(0, "/kaggle/working")
os.chdir("/kaggle/working")
COMP = "/kaggle/input/competitions/playground-series-s6e6"
if not os.path.isdir(COMP):
    COMP = "/kaggle/input/playground-series-s6e6"
if not os.path.exists("data"):
    os.symlink(COMP, "data")
os.makedirs("artifacts", exist_ok=True)

# Install pytabkit WITHOUT touching torch — Kaggle's pre-installed torch matches the GPU;
# letting pip pull pytabkit's torch dep gives "CUDA error: no kernel image" on T4/P100.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytabkit", "--no-deps"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "msgpack", "msgpack_numpy", "dask"], check=True)

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from pytabkit import RealMLP_TD_Classifier

from config import ARTIFACTS, CLASS_ORDER, DATA, N_SPLITS, RANDOM_STATE
from features import load_data, prepare_lgb_xgb
from metrics import clip_proba
from train_oof import encode_target

FEATURE_SET = "all_v2"

train, test, _ = load_data(DATA)
y, _ = encode_target(train)
n_classes = len(CLASS_ORDER)
train_f, test_f, cols = prepare_lgb_xgb(train, test, feature_set=FEATURE_SET)
X = np.nan_to_num(train_f[cols].to_numpy(np.float32), posinf=0, neginf=0)
Xt = np.nan_to_num(test_f[cols].to_numpy(np.float32), posinf=0, neginf=0)
print(f"X={X.shape} Xt={Xt.shape}")


def build():
    # CPU: Kaggle's GPU torch raised "no kernel image available" (arch mismatch) for pytabkit.
    return RealMLP_TD_Classifier(device="cpu", random_state=RANDOM_STATE, n_cv=1)


oof = np.zeros((len(y), n_classes))
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    t0 = time.time()
    clf = build()
    clf.fit(X[tr], y[tr])
    oof[va] = clf.predict_proba(X[va])
    ba = balanced_accuracy_score(y[va], oof[va].argmax(1))
    print(f"  fold {fold}/{N_SPLITS}  BA={ba:.5f}  ({time.time()-t0:.0f}s)")

oof = clip_proba(oof)
oof_ba = balanced_accuracy_score(y, oof.argmax(1))
recalls = {CLASS_ORDER[c]: round(float((oof.argmax(1)[y == c] == c).mean()), 4) for c in range(n_classes)}
print(f"\nrealmlp: OOF BA={oof_ba:.5f}  recalls={recalls}")

print("Refitting on full train for test predictions...")
full = build()
full.fit(X, y)
test_pred = clip_proba(full.predict_proba(Xt))

np.save(ARTIFACTS / "oof_realmlp.npy", oof)
np.save(ARTIFACTS / "test_realmlp.npy", test_pred)
print("Saved artifacts/oof_realmlp.npy and artifacts/test_realmlp.npy")
