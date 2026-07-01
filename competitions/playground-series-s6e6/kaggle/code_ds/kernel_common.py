"""Kernel-side shared helpers for the PS-S6E6 (Stellar Class) pipeline.

This module is synced into the ``s6e6-pipeline-code`` Kaggle Dataset (it lives in
``kaggle/code_ds/`` so ``sync_code.sh`` re-uploads it). Kernels add the dataset to
their sources and do ``from kernel_common import ...`` instead of copy-pasting the
same boilerplate into all ~21 ``run.py`` scripts.

Everything here is pure and importable: nothing at import time touches the network,
the Kaggle filesystem, the GPU, or runs a subprocess. The Kaggle-environment helpers
(``find_competition_root`` / ``find_dataset_file`` / ``install_pascal_torch`` /
``catboost_gpu_params``) only act when *called*, so the module imports cleanly on a
laptop for unit testing.

HARD-WON FACTS encoded here (see CLAUDE.md + experiment history):
  * Kaggle GPU is a P100 (sm_60). Stock torch 2.10+cu128 lacks an sm_60 kernel image
    ("CUDA error: no kernel image available") -> ``install_pascal_torch`` pins
    ``torch==2.4.1`` from the cu121 index and MUST run before ``import torch``.
  * cuDF / RAPIDS dropped Pascal too (crashes "invalid device ordinal") -> all feature
    engineering must be pandas/numpy. CatBoost / XGBoost GPU still work on the P100.
  * Alignment contract: OOF (577347, 3) / test (247435, 3) float32; integer labels
    GALAXY=0 / QSO=1 / STAR=2; ``StratifiedKFold(5, shuffle=True, random_state=42)``
    on integer ``y`` in train-CSV row order. ``stratkfold`` is the single source of
    truth for that split so every model's OOF rows line up.
  * Competition data lives at ``/kaggle/input/(competitions/)?<slug>/``; extra datasets
    mount nested under ``/kaggle/input/**`` -> use the glob-based finders.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Label contract (GALAXY=0 / QSO=1 / STAR=2, in CLASS_ORDER)
# ---------------------------------------------------------------------------
LABELS: List[str] = ["GALAXY", "QSO", "STAR"]
LABEL_MAP: Dict[str, int] = {name: i for i, name in enumerate(LABELS)}
INT_TO_LABEL: Dict[int, str] = {i: name for name, i in LABEL_MAP.items()}

# Canonical artifact shapes (rows fixed by the competition; 3 = num classes).
N_TRAIN = 577347
N_TEST = 247435
N_CLASSES = len(LABELS)

# Canonical fold settings (the alignment contract — do not change per-model).
N_SPLITS = 5
RANDOM_STATE = 42

# Standard Kaggle mount roots.
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

# Files that uniquely identify the competition data directory.
_COMP_MARKERS = ("train.csv", "test.csv", "sample_submission.csv")


# ---------------------------------------------------------------------------
# Kaggle-filesystem discovery (no-ops until called)
# ---------------------------------------------------------------------------
def find_competition_root(
    slug: str = "playground-series-s6e6",
    input_root: Path | str = KAGGLE_INPUT,
) -> Path:
    """Return the directory holding ``train.csv`` / ``test.csv`` / ``sample_submission.csv``.

    Tries, in order:
      1. ``<input_root>/competitions/<slug>``  (current Kaggle layout)
      2. ``<input_root>/<slug>``               (older / flat layout)
      3. a recursive glob for any directory under ``input_root`` that contains all
         three competition marker files (covers unexpected mount paths).

    Args:
        slug: Competition slug (default the S6E6 competition).
        input_root: Root to search under (default ``/kaggle/input``).

    Returns:
        Path to the competition data directory.

    Raises:
        FileNotFoundError: if no directory containing all marker files is found.
    """
    input_root = Path(input_root)
    candidates = [
        input_root / "competitions" / slug,
        input_root / slug,
    ]
    for cand in candidates:
        if all((cand / m).exists() for m in _COMP_MARKERS):
            return cand

    # Fallback: recursive glob for the train.csv marker, then verify siblings.
    for hit in sorted(input_root.glob("**/train.csv")):
        parent = hit.parent
        if all((parent / m).exists() for m in _COMP_MARKERS):
            return parent

    raise FileNotFoundError(
        f"Could not locate competition data (need {list(_COMP_MARKERS)}) "
        f"under {input_root} for slug '{slug}'."
    )


def find_dataset_file(
    filename: str,
    input_root: Path | str = KAGGLE_INPUT,
) -> Path:
    """Recursively find a single named file mounted under ``input_root``.

    Datasets mount at ``/kaggle/input/**/<owner>/<slug>/...`` so the exact path of an
    auxiliary file (e.g. the original ``star_classification.csv`` SDSS17 dataset) is
    not known up front. This globs ``<input_root>/**/<filename>`` and returns the
    first match.

    Args:
        filename: Bare file name to locate (e.g. ``"star_classification.csv"``).
        input_root: Root to search under (default ``/kaggle/input``).

    Returns:
        Path to the first matching file.

    Raises:
        FileNotFoundError: if no file with that name is found.
    """
    input_root = Path(input_root)
    matches = sorted(input_root.glob(f"**/{filename}"))
    if not matches:
        raise FileNotFoundError(f"No file named '{filename}' found under {input_root}.")
    return matches[0]


# ---------------------------------------------------------------------------
# Pascal-safe torch install (call BEFORE `import torch`)
# ---------------------------------------------------------------------------
def install_pascal_torch(
    version: str = "2.4.1",
    extra_index_url: str = "https://download.pytorch.org/whl/cu121",
    quiet: bool = True,
) -> bool:
    """pip-install a torch build that has an sm_60 (P100 / Pascal) kernel image.

    Kaggle's stock torch (2.10+cu128) was compiled without sm_60, so any torch CUDA
    op on the P100 dies with "CUDA error: no kernel image available for execution on
    the device". Installing ``torch==2.4.1`` from the cu121 wheel index fixes it.

    MUST be called *before* ``import torch`` (a torch already imported in the process
    keeps the old binary). Safe to call when torch is unused — it just (re)installs a
    wheel. After installing, some libraries need ``device="cuda:0"`` rather than the
    bare ``"cuda"`` alias.

    Args:
        version: torch version to pin.
        extra_index_url: PyTorch wheel index that carries the cu121 build.
        quiet: pass ``-q`` to pip to reduce log noise.

    Returns:
        True if the pip command exited 0, False otherwise (never raises so a kernel
        can decide to fall back to CPU).
    """
    cmd = [
        sys.executable, "-m", "pip", "install",
        f"torch=={version}",
        "--extra-index-url", extra_index_url,
    ]
    if quiet:
        cmd.append("-q")
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"install_pascal_torch failed: {exc!r}", flush=True)
        return False


# ---------------------------------------------------------------------------
# CatBoost GPU parameter guard
# ---------------------------------------------------------------------------
def catboost_gpu_params(
    base_params: Optional[dict] = None,
    thread_count: int = 4,
) -> dict:
    """Augment CatBoost params with GPU settings iff a usable GPU is present.

    CatBoost only raises about a missing GPU at ``.fit()`` time, so this probes
    ``catboost.utils.get_gpu_device_count()`` up front. CatBoost GPU *does* work on
    the P100 (unlike cuDF), so when a device is detected we set
    ``task_type="GPU", devices="0"``; otherwise we fall back to
    ``task_type="CPU", thread_count=<n>``.

    Args:
        base_params: existing CatBoost kwargs (copied, not mutated). None -> {}.
        thread_count: CPU threads to request in the CPU-fallback branch.

    Returns:
        A new dict = base_params + the resolved device settings.
    """
    params = dict(base_params or {})
    n_gpu = 0
    try:
        from catboost.utils import get_gpu_device_count

        n_gpu = int(get_gpu_device_count())
    except Exception as exc:  # catboost missing, or probe failed -> CPU
        print(f"catboost GPU probe failed ({exc!r}); using CPU", flush=True)
        n_gpu = 0

    if n_gpu > 0:
        params["task_type"] = "GPU"
        params["devices"] = "0"
    else:
        params["task_type"] = "CPU"
        params.setdefault("thread_count", thread_count)
    return params


# ---------------------------------------------------------------------------
# Canonical fold split (alignment contract)
# ---------------------------------------------------------------------------
def stratkfold(
    y: Sequence[int] | np.ndarray,
    n: int = N_SPLITS,
    seed: int = RANDOM_STATE,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return the canonical StratifiedKFold (train_idx, val_idx) splits.

    THE alignment contract for every OOF model: ``StratifiedKFold(n_splits=n,
    shuffle=True, random_state=seed)`` over integer labels in train-CSV row order.
    Using this everywhere guarantees ``oof_<model>.npy`` arrays line up row-for-row.

    Args:
        y: integer class labels (GALAXY=0 / QSO=1 / STAR=2) in CSV order.
        n: number of folds (default 5).
        seed: random_state for the shuffle (default 42).

    Returns:
        List of ``(train_idx, val_idx)`` numpy index arrays, one per fold.
    """
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    # StratifiedKFold ignores X values; pass a zero placeholder of the right length.
    return [(tr, va) for tr, va in skf.split(np.zeros(len(y)), y)]


# ---------------------------------------------------------------------------
# SDSS17 categorical reconstruction (for the original-data rows)
# ---------------------------------------------------------------------------
def _spectral_type_from_gr(g: np.ndarray, r: np.ndarray) -> np.ndarray:
    """spectral_type = cut(r-g, [-inf, -1, -0.5, 0, inf], right=True).

    right=True bins are ``(edge_i, edge_{i+1}]`` -> labelled by the first upper edge
    the value is ``<=``. Non-finite (NaN/inf) values are out-of-range and map to
    ``'nan'`` (matching ``cudf.cut``/``pd.cut`` returning NaN -> ``astype(str)``).
    """
    rg = np.asarray(r, dtype="float64") - np.asarray(g, dtype="float64")
    out = np.full(rg.shape, "O/B", dtype=object)  # (0, inf]
    out[rg <= 0.0] = "A/F"   # (-0.5, 0]
    out[rg <= -0.5] = "G/K"  # (-1, -0.5]
    out[rg <= -1.0] = "M"    # (-inf, -1]
    out[~np.isfinite(rg)] = "nan"
    return out


def _galaxy_population_from_ur(u: np.ndarray, r: np.ndarray) -> np.ndarray:
    """galaxy_population = cut(u-r, [-inf, 2.2, inf], right=True).

    ``<= 2.2`` -> 'Blue_Cloud', else 'Red_Sequence'; non-finite -> 'nan'.
    """
    ur = np.asarray(u, dtype="float64") - np.asarray(r, dtype="float64")
    out = np.full(ur.shape, "Red_Sequence", dtype=object)  # (2.2, inf]
    out[ur <= 2.2] = "Blue_Cloud"                            # (-inf, 2.2]
    out[~np.isfinite(ur)] = "nan"
    return out


def reconstruct_sdss_cats(df):
    """Add ``spectral_type`` / ``galaxy_population`` columns to original-SDSS17 rows.

    The competition train/test carry these categorical columns, but the original
    SDSS17 dataset does not. They are reconstructed from photometric colors so the
    original rows can be appended to the training pool with the same feature schema:

      * ``spectral_type``     from ``cut(r - g, [-inf, -1, -0.5, 0, inf], right=True)``
        -> labels ``M`` / ``G/K`` / ``A/F`` / ``O/B``
      * ``galaxy_population`` from ``cut(u - r, [-inf, 2.2, inf], right=True)``
        -> labels ``Blue_Cloud`` / ``Red_Sequence``

    Args:
        df: a pandas DataFrame with numeric ``u``, ``g``, ``r`` columns.

    Returns:
        The same DataFrame (mutated in place and also returned) with the two new
        string categorical columns added.
    """
    g = df["g"].to_numpy()
    r = df["r"].to_numpy()
    u = df["u"].to_numpy()
    df["spectral_type"] = _spectral_type_from_gr(g, r)
    df["galaxy_population"] = _galaxy_population_from_ur(u, r)
    return df


# ---------------------------------------------------------------------------
# OOF / test artifact saving (with verification + optional submission)
# ---------------------------------------------------------------------------
def _normalize_probs(p: np.ndarray, clip: float = 1e-7) -> np.ndarray:
    """Row-normalize and clip away from 0; return float32 (the saved-array contract)."""
    p = np.asarray(p, dtype="float32")
    p = np.clip(p, clip, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p.astype("float32")


def save_oof_test(
    oof: np.ndarray,
    test: np.ndarray,
    name: str,
    work: Path | str = KAGGLE_WORKING,
    y: Optional[np.ndarray] = None,
    sample=None,
    clip: float = 1e-7,
) -> Dict[str, str]:
    """Row-normalize, save ``oof_<name>.npy`` / ``test_<name>.npy``, write results + submission.

    Both arrays are clipped away from 0, row-normalized to sum to 1, cast to float32,
    and saved as ``oof_<name>.npy`` / ``test_<name>.npy`` in ``work``. If ``y`` is
    given, overall balanced accuracy and per-class recall are written to
    ``results.txt``. If ``sample`` (the sample_submission DataFrame) is given, a
    ``submission.csv`` is produced with argmax labels in sample-row order.

    Args:
        oof: OOF probabilities, shape ``(n_train, 3)``.
        test: test probabilities, shape ``(n_test, 3)``.
        name: model id used in the output file names.
        work: output directory (default ``/kaggle/working``).
        y: optional integer OOF labels for the results metrics block.
        sample: optional sample_submission DataFrame (needs an ``id`` column) used to
            write ``submission.csv``.
        clip: lower clip bound before normalization.

    Returns:
        Dict mapping logical output name -> written path (str). Keys: ``oof``,
        ``test``, ``results``, and ``submission`` (only if ``sample`` was given).
    """
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)

    oof_n = _normalize_probs(oof, clip=clip)
    test_n = _normalize_probs(test, clip=clip)

    oof_path = work / f"oof_{name}.npy"
    test_path = work / f"test_{name}.npy"
    np.save(oof_path, oof_n)
    np.save(test_path, test_n)

    # --- results.txt -------------------------------------------------------
    results_path = work / "results.txt"
    lines = [
        f"model: {name}",
        f"oof shape: {oof_n.shape}",
        f"test shape: {test_n.shape}",
    ]
    if y is not None:
        from sklearn.metrics import balanced_accuracy_score

        y = np.asarray(y)
        y_pred = oof_n.argmax(axis=1)
        bal_acc = float(balanced_accuracy_score(y, y_pred))
        lines.append(f"oof_balanced_accuracy: {bal_acc:.6f}")
        for i, label in enumerate(LABELS):
            mask = y == i
            recall = float((y_pred[mask] == i).mean()) if mask.sum() else 0.0
            lines.append(f"recall[{label}]: {recall:.6f}")
    results_path.write_text("\n".join(lines) + "\n")

    out: Dict[str, str] = {
        "oof": str(oof_path),
        "test": str(test_path),
        "results": str(results_path),
    }

    # --- submission.csv ----------------------------------------------------
    if sample is not None:
        import pandas as pd  # local import: pure helpers stay numpy-only

        pred_labels = [INT_TO_LABEL[i] for i in test_n.argmax(axis=1)]
        submission = sample.copy()
        target_col = "class" if "class" in submission.columns else submission.columns[-1]
        submission[target_col] = pred_labels
        sub_path = work / "submission.csv"
        submission.to_csv(sub_path, index=False)
        out["submission"] = str(sub_path)

    return out


__all__ = [
    "LABELS",
    "LABEL_MAP",
    "INT_TO_LABEL",
    "N_TRAIN",
    "N_TEST",
    "N_CLASSES",
    "N_SPLITS",
    "RANDOM_STATE",
    "find_competition_root",
    "find_dataset_file",
    "install_pascal_torch",
    "catboost_gpu_params",
    "stratkfold",
    "reconstruct_sdss_cats",
    "save_oof_test",
]
