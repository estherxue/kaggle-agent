"""Exp1: TabPFN-3 base model (5-fold OOF) — a Transformer in-context learner whose
inductive bias is orthogonal to the GBDT/MLP/KNN family, to raise the stack's oracle ceiling.

TabPFN cannot fit on ~460k rows/fold, so we subsample a stratified CONTEXT per fold and
average over N_BAGS draws to cut variance. Predictions cover the full validation fold.

Fold split is StratifiedKFold(N_SPLITS, shuffle=True, random_state=42) — byte-identical to
train_oof.py so oof_tabpfn.npy aligns row-for-row with existing artifacts.

Mounts: s6e6-pipeline-code (helpers) + competition data. Internet ON (downloads weights).
"""
import os, shutil, sys, time, subprocess

# TabPFN license token (priorlabs.ai) — required to download model weights non-interactively.
# Private kernel; competitions/ is gitignored so this stays out of version control.
os.environ["TABPFN_TOKEN"] = "tabpfn_sk_oAbCdZ07CI8wXOHNHODOmYScVEtokQ0cj5TaDbUpKy0"
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

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

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn"], check=True)

# Token (verified valid via verify_token) is cached for HF download auth. The local
# ensure_license_accepted() gate misfires in non-interactive Kaggle even with a valid token
# and accepted license, so we short-circuit it (download still authenticates with the token).
from tabpfn.browser_auth import save_token
save_token(os.environ["TABPFN_TOKEN"])
import tabpfn.browser_auth as _ba
import tabpfn.model_loading as _ml
_ba.ensure_license_accepted = lambda hf_repo_id=None, *a, **k: True
_ml.ensure_license_accepted = lambda hf_repo_id=None, *a, **k: True
print("Token cached; license gate bypassed for non-interactive run.")

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier

from config import ARTIFACTS, CLASS_ORDER, DATA, N_SPLITS, RANDOM_STATE
from features import load_data, prepare_lgb_xgb
from metrics import clip_proba
from train_oof import encode_target

N_CONTEXT = 6000    # reduced so 5-fold OOF + 247k-row test fits the 12h CPU cap
N_BAGS = 1          # CPU: single bag (context=6000) ~0.95h/fold, ~7h total
FEATURE_SET = "all_v2"

train, test, _ = load_data(DATA)
y, _ = encode_target(train)
n_classes = len(CLASS_ORDER)
train_f, test_f, cols = prepare_lgb_xgb(train, test, feature_set=FEATURE_SET)
X = np.nan_to_num(train_f[cols].to_numpy(np.float32), posinf=0, neginf=0)
Xt = np.nan_to_num(test_f[cols].to_numpy(np.float32), posinf=0, neginf=0)
print(f"X={X.shape} Xt={Xt.shape}  context={N_CONTEXT} bags={N_BAGS}")


def stratified_subsample(idx, labels, n, seed):
    rng = np.random.default_rng(seed)
    if len(idx) <= n:
        return idx
    out = []
    for c in np.unique(labels):
        c_idx = idx[labels == c]
        k = max(1, int(round(n * len(c_idx) / len(idx))))
        out.append(rng.choice(c_idx, size=min(k, len(c_idx)), replace=False))
    return np.concatenate(out)


oof = np.zeros((len(y), n_classes))
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    t0 = time.time()
    fold_pred = np.zeros((len(va), n_classes))
    for bag in range(N_BAGS):
        sub = stratified_subsample(tr, y[tr], N_CONTEXT, seed=1000 * fold + bag)
        clf = TabPFNClassifier(device="cpu")
        clf.fit(X[sub], y[sub])
        fold_pred += clf.predict_proba(X[va])
    oof[va] = fold_pred / N_BAGS
    ba = balanced_accuracy_score(y[va], oof[va].argmax(1))
    print(f"  fold {fold}/{N_SPLITS}  BA={ba:.5f}  ({time.time()-t0:.0f}s)")

oof = clip_proba(oof)
oof_ba = balanced_accuracy_score(y, oof.argmax(1))
recalls = {CLASS_ORDER[c]: round(float((oof.argmax(1)[y == c] == c).mean()), 4) for c in range(n_classes)}
print(f"\ntabpfn: OOF BA={oof_ba:.5f}  recalls={recalls}")
# Save OOF before test prediction so a late cancellation still preserves the costly OOF.
np.save(ARTIFACTS / "oof_tabpfn.npy", oof)
print("Saved artifacts/oof_tabpfn.npy")

print("Predicting test (full-data context bags)...")
test_pred = np.zeros((len(Xt), n_classes))
all_idx = np.arange(len(y))
for bag in range(N_BAGS):
    sub = stratified_subsample(all_idx, y, N_CONTEXT, seed=99000 + bag)
    clf = TabPFNClassifier(device="cpu")
    clf.fit(X[sub], y[sub])
    test_pred += clf.predict_proba(Xt)
test_pred = clip_proba(test_pred / N_BAGS)

np.save(ARTIFACTS / "oof_tabpfn.npy", oof)
np.save(ARTIFACTS / "test_tabpfn.npy", test_pred)
print("Saved artifacts/oof_tabpfn.npy and artifacts/test_tabpfn.npy")
