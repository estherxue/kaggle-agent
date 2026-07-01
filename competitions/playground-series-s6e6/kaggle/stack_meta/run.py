"""Exp2: compare LogReg vs MLP stacking meta-learner on the current 7 multi-seed models.

Mounts:
  - s6e6-pipeline-code  (stack.py + helpers)
  - s6e6-artifacts      (oof_*.npy / test_*.npy / y_train.npy)
No competition data needed (pure OOF stacking).
Results are written to /kaggle/working/results.txt so they survive in the kernel output.
"""
import os, shutil, sys

def find_input(slug, owner="cindyxue1122"):
    """Resolve a dataset mount path across Kaggle's flat and nested input layouts."""
    for cand in (f"/kaggle/input/{slug}", f"/kaggle/input/datasets/{owner}/{slug}"):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"dataset {slug} not mounted; /kaggle/input = {os.listdir('/kaggle/input')}")


CODE = find_input("s6e6-pipeline-code")
ARTS = find_input("s6e6-artifacts")

for f in os.listdir(CODE):
    if f.endswith(".py"):
        shutil.copy(os.path.join(CODE, f), f"/kaggle/working/{f}")

sys.path.insert(0, "/kaggle/working")
os.chdir("/kaggle/working")
# Symlink artifacts dir to the mounted dataset (no 320MB copy); config.ARTIFACTS = ./artifacts
if not os.path.exists("artifacts"):
    os.symlink(ARTS, "artifacts")

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from config import ARTIFACTS, CLASS_ORDER, N_SPLITS, RANDOM_STATE
from stack import build_meta, load_stack, search_bias, _logp

MODELS = ["lgb_multi", "xgb_multi", "cat_multi", "hgb_multi",
          "logreg_multi", "mlp_multi", "specialist"]

out = open("/kaggle/working/results.txt", "w")
def log(msg=""):
    print(msg)
    out.write(msg + "\n")
    out.flush()

y = np.load(ARTIFACTS / "y_train.npy")
n_classes = len(CLASS_ORDER)
X = load_stack(MODELS, "oof")
log(f"Models: {MODELS}")
log(f"Meta-features: {X.shape}\n")

for meta_kind in ["logreg", "mlp"]:
    meta_oof = np.zeros((len(y), n_classes))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for tr, va in skf.split(X, y):
        meta = build_meta(meta_kind, RANDOM_STATE, 1.0)
        meta.fit(X[tr], y[tr])
        meta_oof[va] = meta.predict_proba(X[va])

    ba_before = balanced_accuracy_score(y, meta_oof.argmax(1))
    bias = search_bias(y, meta_oof)
    pred = (_logp(meta_oof) + bias).argmax(1)
    ba_after = balanced_accuracy_score(y, pred)
    recalls = {CLASS_ORDER[c]: round(float((pred[y == c] == c).mean()), 4) for c in range(n_classes)}
    log(f"=== meta={meta_kind} ===")
    log(f"  OOF BA before bias: {ba_before:.5f}")
    log(f"  OOF BA after  bias: {ba_after:.5f}  recalls={recalls}")
    log(f"  class bias: {dict(zip(CLASS_ORDER, np.round(bias, 4)))}\n")

log("Baseline to beat (LogReg meta, 7-model multi-seed): 0.96708")
out.close()
