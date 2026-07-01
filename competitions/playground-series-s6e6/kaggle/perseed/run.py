"""Remote runner for the per-seed stacking experiment (exp_perseed.py).

Feeds individual seed predictions to logreg AND mlp meta-learners (so a non-linear meta can
exploit cross-seed variance as an uncertainty signal), then layers per-class scale+shift
calibration and disagreement-gated anomaly patching. Writes results to /kaggle/working/results.txt.

Mounts: s6e6-pipeline-code (exp_perseed.py + helpers) + s6e6-artifacts (per-seed oof/test).
CPU kernel (pure stacking / sklearn meta-learners).
"""
import os, shutil, sys, io, contextlib

def find_input(slug, owner="cindyxue1122"):
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
if not os.path.exists("artifacts"):
    os.symlink(ARTS, "artifacts")

import exp_perseed

results = []
for meta_kind in ["logreg", "mlp"]:
    print(f"\n{'='*60}\n=== META = {meta_kind} ===\n{'='*60}")
    buf = io.StringIO()
    sys.argv = ["exp_perseed.py", "--meta", meta_kind]
    with contextlib.redirect_stdout(buf):
        exp_perseed.main()
    text = buf.getvalue()
    print(text)
    results.append(f"=== META = {meta_kind} ===\n{text}")

with open("/kaggle/working/results.txt", "w") as f:
    f.write("\n".join(results))
print("\nWrote results.txt")
