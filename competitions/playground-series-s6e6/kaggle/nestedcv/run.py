"""Remote runner: nested-CV honest evaluation of stacking + calibration for several model sets.

Reports nested-CV BA (additive bias vs per-class scale+shift) where every test row is held out
of both the meta-learner fit and the calibration fit. Writes results.txt.

Mounts: s6e6-pipeline-code (nested_cv.py + helpers) + s6e6-artifacts (oof arrays). CPU.
"""
import os, shutil, sys, io, contextlib

def find_input(slug, owner="cindyxue1122"):
    for cand in (f"/kaggle/input/{slug}", f"/kaggle/input/datasets/{owner}/{slug}"):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"{slug} not mounted; /kaggle/input={os.listdir('/kaggle/input')}")


CODE = find_input("s6e6-pipeline-code")
ARTS = find_input("s6e6-artifacts")
for f in os.listdir(CODE):
    if f.endswith(".py"):
        shutil.copy(os.path.join(CODE, f), f"/kaggle/working/{f}")
sys.path.insert(0, "/kaggle/working")
os.chdir("/kaggle/working")
if not os.path.exists("artifacts"):
    os.symlink(ARTS, "artifacts")

import nested_cv

CONFIGS = {
    "7-multi (robust)": "lgb_multi,xgb_multi,cat_multi,hgb_multi,logreg_multi,mlp_multi,specialist",
    "9-model": "lgb_multi,xgb_multi,cat_multi,hgb_multi,logreg_multi,mlp_multi,specialist,realmlp,knn_multi",
    "9+et (10)": "lgb_multi,xgb_multi,cat_multi,hgb_multi,logreg_multi,mlp_multi,specialist,realmlp,knn_multi,et",
}

out = []
for name, models in CONFIGS.items():
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    buf = io.StringIO()
    sys.argv = ["nested_cv.py", "--models", models, "--meta", "logreg"]
    with contextlib.redirect_stdout(buf):
        nested_cv.main()
    text = buf.getvalue()
    print(text)
    out.append(f"### {name} ###\n{text}")

with open("/kaggle/working/results.txt", "w") as f:
    f.write("\n".join(out))
print("\nWrote results.txt")
