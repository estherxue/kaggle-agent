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

# Base models contributing to the OOF stack / blend. Order is stable so saved
# artifacts (oof_<m>.npy / test_<m>.npy) and blend weights line up.
MODELS = ["lgb", "xgb", "cat", "hgb"]

RANDOM_STATE = 42
N_SPLITS = 5
# Upper cap on boosting rounds; early stopping decides the actual count per fold.
N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 100
CLIP = 1e-15
