"""
Generate the Kaggle submission notebook from the canonical attack.py.

Structure matches the proven getting-started contract for this gateway comp:
  cell 1  setup: put the mounted competition package (aicomp_sdk + kaggle_evaluation) on
          sys.path.
  cell 2  materialize attack.py at /kaggle/working/attack.py (embedded, base64).
  cell 3  guarded launch: only the real competition rerun boots the grader; a plain
          "Save & Run All" writes a placeholder submission.csv so the notebook completes
          (required before you can submit).

Usage:
  python build_notebook.py --attack ../../../../multi-step-tool-attacks/attack/attack.py \
      --out submission/msta_submission.ipynb
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

SETUP = (
    "# put the mounted competition package (aicomp_sdk + kaggle_evaluation) on the path\n"
    "import sys, glob\n"
    "from pathlib import Path\n"
    "sys.argv = [sys.argv[0]]\n"
    "for c in glob.glob('/kaggle/input/**/kaggle_evaluation', recursive=True):\n"
    "    sys.path.insert(0, str(Path(c).parent))\n"
    "    break\n"
)

LAUNCH = (
    "# only the real rerun starts the grader; Save & Run All self-tests + writes placeholder\n"
    "import os, csv, importlib.util\n"
    "if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n"
    "    import kaggle_evaluation.jed_attack_134815.jed_attack_inference_server as jed\n"
    "    jed.JEDAttackInferenceServer().serve()\n"
    "else:\n"
    "    # SELF-TEST the real rerun import path so a broken submission never ships.\n"
    "    import aicomp_sdk\n"
    "    from aicomp_sdk.attacks import AttackAlgorithmBase\n"
    "    import kaggle_evaluation.jed_attack_134815.jed_attack_inference_server  # noqa: F401\n"
    "    _spec = importlib.util.spec_from_file_location('a', '/kaggle/working/attack.py')\n"
    "    _m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)\n"
    "    assert issubclass(_m.AttackAlgorithm, AttackAlgorithmBase), 'AttackAlgorithm contract'\n"
    "    print('SELF-TEST OK: aicomp_sdk + inference server import, AttackAlgorithm loads')\n"
    "    with open('/kaggle/working/submission.csv', 'w', newline='') as f:\n"
    "        w = csv.writer(f)\n"
    "        w.writerow(['Id', 'Score'])\n"
    "        for r in ['gpt_oss_public', 'gpt_oss_private', 'gemma_public', 'gemma_private']:\n"
    "            w.writerow([r, 0.0])\n"
    "    print('placeholder submission.csv written (not a competition rerun)')\n"
)


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


def build(attack_src: str) -> dict:
    b64 = base64.b64encode(attack_src.encode("utf-8")).decode("ascii")
    writer = (
        "# materialize the red-team submission at /kaggle/working/attack.py\n"
        "import base64\n"
        f"_ATTACK_B64 = {json.dumps(b64)}\n"
        "with open('/kaggle/working/attack.py', 'wb') as _f:\n"
        "    _f.write(base64.b64decode(_ATTACK_B64))\n"
        "print('wrote /kaggle/working/attack.py', len(base64.b64decode(_ATTACK_B64)), 'bytes')\n"
    )
    return {
        "cells": [code_cell(SETUP), code_cell(writer), code_cell(LAUNCH)],
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    src = args.attack.read_text()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(build(src), indent=1))
    print(f"wrote {args.out} from {args.attack} ({len(src)} chars of attack code)")


if __name__ == "__main__":
    main()
