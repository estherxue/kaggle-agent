"""Shared constants for S6E6 pipeline."""

from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
SUBMISSIONS = ROOT / "submissions"

TARGET_COL = "class"
ID_COL = "id"
CAT_COLS = ["spectral_type", "galaxy_population"]
CLASS_ORDER = ["GALAXY", "QSO", "STAR"]

RANDOM_STATE = 42
N_SPLITS = 5
N_ESTIMATORS = 300
CLIP = 1e-15
