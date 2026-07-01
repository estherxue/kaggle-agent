"""s6e6 ovrxgb: ONE-VS-REST (OvR) variant of the xgbv5 multiclass XGBoost kernel.

This kernel REUSES the ENTIRE xgbv5 feature-engineering / target-encoding / priors
pipeline VERBATIM (pure pandas + numpy, P100-safe, no cuDF/cuPy). The ONLY change is
the model decomposition: instead of one `multi:softprob` model per fold, it trains
THREE BINARY models per fold (class k vs rest, for k in {GALAXY=0, QSO=1, STAR=2})
with `objective='binary:logistic'`, then for each row collects the 3 binary
P(class=k) into a 3-column array and ROW-NORMALIZES (divide by row sum, clip to
[1e-7, 1]) into a valid 3-class probability. This adds ensemble diversity through a
different decomposition while keeping every feature and the fold contract identical
to xgbv5.

Per-binary class balancing for balanced accuracy: each binary model uses
`scale_pos_weight = n_neg / n_pos` of its own 1-vs-rest split (the QSO class is rare,
so its positive class is heavily up-weighted). Each binary model early-stops on its
own validation logloss. The base hyperparameters (learning_rate, max_leaves, max_bin,
regularization, subsample, etc.) are copied verbatim from the xgbv5 multiclass params,
with the objective swapped to binary and num_class/multiclass eval removed. The
iteration cap is lowered (n_estimators=5000, early_stopping_rounds=150) to keep
3 binary models x 5 folds within Kaggle's GPU time limit; the multiclass source used
7000/180.

Everything below the training section is the xgbv5 FE pipeline COPIED VERBATIM.

Feature engineering is pure pandas + numpy (NO cuDF / cuPy). Kaggle assigns a
P100 (sm_60) and cuDF 26.x dropped Pascal support, so cudf.read_csv crashed with
`copy_if failed: cudaErrorInvalidDevice: invalid device ordinal`. XGBoost GPU
itself works on the P100 (Pascal is supported), so ONLY the cuDF/cuPy feature-
engineering + cuML TargetEncoder layers were rewritten in pandas/numpy. The
produced features (names + semantics), the fold contract, the original-data
handling (priors/freq only; rows NOT appended), and the XGBoost params are
unchanged from the source notebook.

The pandas/numpy FE reproduces the cuDF/cuPy version 1:1:
  * color diffs / band stats / flux=10**(-0.4*mag) / sky sin/cos / curvatures:
    plain numpy vector math (the cuDF source used cp.mean/std(ddof=1)/min/max,
    NOT nan-aware -- reproduced with np.mean/std(ddof=1)/min/max here).
  * quantile bins (`qbin{q}`): np.quantile(linear) edges of probs linspace(0,1,q+1)
    on the train+test reference values, then searchsorted(side='left')-1 with the
    exact boundary/NaN handling of qcut_codes_gpu; emitted as string codes.
  * floor / scaled-floor artifact cats: np.floor on nan->0 filled arrays, sentinel
    INT32_MIN for non-finite, emitted as string codes (matches cudf .astype('str')).
  * spectral_type/galaxy_population_calc: cudf.cut(right=True) reconstructed via
    <= threshold np.select logic.
  * 2-/3-way combos: string concat with '__' separator (identical to cudf).
  * frequency features: value_counts over train+test+orig, mapped back, log1p.
  * original-prior features: groupby on ORIGINAL rows only (count + per-class mean),
    fillna with the global original-class prior.
  * fold-safe target encoding (TE_*): leave-one-inner-fold-out smoothing encoder
    (smooth=16, 7 inner stratified folds) reproducing cuML TargetEncoder
    (split_method='customize'); transform uses full-fold-train stats.
All bins / edges / vocabularies are computed on the COMBINED train+test+original
frame (one concatenated DataFrame), so codes are consistent and label-free.

Adaptations from the source notebook (only these):
  * Data-path discovery -> recursive glob that works with Kaggle's nested mounts.
  * Original-dataset path -> my mirror "cindyxue1122/s6e6-original-sdss17".
  * Original rows filtered to class in {GALAXY,QSO,STAR} and sentinel band
    magnitudes (outside (-100,100), e.g. -9999) dropped before FE (contract guard).
  * Output paths/names -> /kaggle/working/oof_xgbv5.npy, test_xgbv5.npy, results.txt.
  * Removed plotting/display.

Label mapping: GALAXY=0, QSO=1, STAR=2 (alphabetical), matching all artifacts.
The outer fold split is the contract split
StratifiedKFold(n_splits=5, shuffle=True, random_state=42) over the competition
train rows only, integer labels in original train-CSV order.

Outputs to /kaggle/working/:
  oof_ovrxgb.npy   (577347, 3) float32 ROW-NORMALIZED OOF probabilities, train-CSV row order, [GALAXY,QSO,STAR]
  test_ovrxgb.npy  (247435, 3) float32 ROW-NORMALIZED fold-averaged test probabilities, sample_submission order
  results.txt      per-fold + overall OOF balanced accuracy (argmax of normalized probs) + per-class recall
  submission.csv   argmax -> label (optional)
"""

import os

# Keep CPU helper libraries from oversubscribing while XGBoost owns the GPU.
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')

import gc
import glob
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import xgboost as xgb

from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 120)


# ---------------------------------------------------------------------------
# Constants (verbatim from source)
# ---------------------------------------------------------------------------
SEED = 42
N_SPLITS = 5
TARGET = 'class'
ID_COL = 'id'
MODEL_ID = 'ovrxgb'

CLASSES = ['GALAXY', 'QSO', 'STAR']
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}

EPS = np.float32(1e-6)
RAW_NUM_COLS = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
BANDS = ['u', 'g', 'r', 'i', 'z']

# EXP3-110 configuration. Original labels are used for prior features only;
# original rows are not appended to fold training data.
USE_ORIGINAL_ROWS = False
USE_CLASS_WEIGHTS = True
CLASS_WEIGHT_POWER = 1.0

TE_SOURCE = 'all'
TE_SMOOTH = 16.0
TE_INNER_SPLITS = 7
TE_MAX_CARDINALITY = 5000
TOP_N_FEATURES = 370

ART_LOWFREQ_COLS = [f'art_{c}_floor' for c in RAW_NUM_COLS]
ART_LOWFREQ_PAIR_COLS = ['art_alpha_floor_x_delta_floor', 'art_u_floor_x_z_floor']
ART_COLOR_BIN_SPECS = [
    ('u_g', 2.0, 'half'),
    ('g_r', 2.0, 'half'),
    ('r_i', 2.0, 'half'),
    ('i_z', 2.0, 'half'),
    ('u_r', 1.0, 'one'),
    ('redshift', 10.0, 'tenth'),
    ('alpha', 0.2, 'deg5'),
    ('delta', 0.2, 'deg5'),
]
ART_COLOR_COLS = [f'art_{c}_{tag}' for c, _, tag in ART_COLOR_BIN_SPECS]
ART_COLOR_PAIR_COLS = [
    'art_u_g_half_x_redshift_tenth',
    'art_g_r_half_x_redshift_tenth',
    'art_alpha_deg5_x_delta_deg5',
]

INT32_MIN = -2147483648

# ---- output paths (changed for Kaggle working dir + stacking artifacts) ----
WORK = Path('/kaggle/working')
WORK.mkdir(parents=True, exist_ok=True)
OOF_PATH = WORK / 'oof_ovrxgb.npy'
PRED_PATH = WORK / 'test_ovrxgb.npy'
SUB_PATH = WORK / 'submission.csv'
RESULTS_PATH = WORK / 'results.txt'

# legacy paths kept so nothing downstream breaks if referenced
LEGACY_OOF_PATH = WORK / 'train_oof' / f'{MODEL_ID}_oof.npy'
LEGACY_PRED_PATH = WORK / 'test_preds' / f'{MODEL_ID}_test_preds.npy'
for path in [LEGACY_OOF_PATH.parent, LEGACY_PRED_PATH.parent]:
    path.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

_RESULT_LINES = []


def emit(msg=''):
    print(msg, flush=True)
    _RESULT_LINES.append(str(msg))


def flush_results():
    RESULTS_PATH.write_text('\n'.join(_RESULT_LINES) + '\n')


# ---------------------------------------------------------------------------
# Load data (pure pandas)
# ---------------------------------------------------------------------------
def find_competition_root():
    candidates = [
        Path('/kaggle/input/competitions/playground-series-s6e6'),
        Path('/kaggle/input/playground-series-s6e6'),
    ]
    candidates += [Path(p).parent for p in glob.glob('/kaggle/input/**/train.csv', recursive=True)]
    seen = []
    for c in candidates:
        if c not in seen:
            seen.append(c)
    for root in seen:
        if (root / 'train.csv').exists() and (root / 'test.csv').exists():
            return root
    raise FileNotFoundError('Could not find train.csv and test.csv. Add the competition data to the notebook inputs.')


def find_original_path():
    # My mirror: cindyxue1122/s6e6-original-sdss17 (mounts nested). Recursive glob.
    candidates = [Path(p) for p in glob.glob('/kaggle/input/**/star_classification.csv', recursive=True)]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        'Could not find star_classification.csv. Add the original stellar classification dataset '
        '(cindyxue1122/s6e6-original-sdss17) to the Kaggle notebook inputs for this model.'
    )


def clean_num(s):
    # cudf.to_numeric(errors='coerce').astype('float32') equivalent.
    return pd.to_numeric(s, errors='coerce').astype('float32')


def cat_key(s):
    # cudf: s.astype('str').fillna('__NA__'). pandas Series.astype('str') turns NaN
    # into the string 'nan' (cudf did the same for float NaN), so fillna catches
    # only true None/NA entries; the result is identical to the cuDF version.
    return s.astype('str').fillna('__NA__')


def spectral_type_cut(g, r):
    # cudf.cut(r-g, [-inf,-1,-0.5,0,inf], labels=['M','G/K','A/F','O/B']) with the
    # cuDF default right=True -> intervals (edge_i, edge_{i+1}]; label by first
    # upper edge that value is <= . astype('str'); NaN/out-of-range -> 'nan'.
    rg = np.asarray(r, dtype=np.float32) - np.asarray(g, dtype=np.float32)
    out = np.full(rg.shape, 'O/B', dtype=object)
    out[rg <= 0.0] = 'A/F'
    out[rg <= -0.5] = 'G/K'
    out[rg <= -1.0] = 'M'
    out[~np.isfinite(rg)] = 'nan'
    return pd.Series(out)


def galaxy_population_cut(u, r):
    # cudf.cut(u-r, [-inf, 2.2, inf], labels=['Blue_Cloud','Red_Sequence']), right=True.
    ur = np.asarray(u, dtype=np.float32) - np.asarray(r, dtype=np.float32)
    out = np.full(ur.shape, 'Red_Sequence', dtype=object)
    out[ur <= 2.2] = 'Blue_Cloud'
    out[~np.isfinite(ur)] = 'nan'
    return pd.Series(out)


def read_competition_csv(path):
    df = pd.read_csv(str(path))
    for c in RAW_NUM_COLS:
        df[c] = clean_num(df[c])
    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype('int32')
    df['spectral_type'] = cat_key(df['spectral_type'])
    df['galaxy_population'] = cat_key(df['galaxy_population'])
    return df


def read_original_csv(path):
    orig = pd.read_csv(str(path))
    for c in RAW_NUM_COLS:
        orig[c] = clean_num(orig[c])
    # --- contract guard: keep only the three competition classes and drop rows
    # with sentinel band magnitudes (e.g. -9999) before FE. The SDSS17 uppercase
    # 'class' column already matches {GALAXY,QSO,STAR}. This removes the handful of
    # sentinel rows that would otherwise poison color/flux/prior features.
    if TARGET in orig.columns:
        orig[TARGET] = orig[TARGET].astype('str')
        orig = orig[orig[TARGET].isin(CLASSES)]
    band_ok = None
    for b in BANDS:
        if b in orig.columns:
            ok = (orig[b] > -100) & (orig[b] < 100)
            band_ok = ok if band_ok is None else (band_ok & ok)
    if band_ok is not None:
        orig = orig[band_ok]
    orig = orig.reset_index(drop=True)
    # --- end contract guard ---
    if 'spectral_type' not in orig.columns:
        orig['spectral_type'] = spectral_type_cut(orig['g'], orig['r']).values
    if 'galaxy_population' not in orig.columns:
        orig['galaxy_population'] = galaxy_population_cut(orig['u'], orig['r']).values
    orig['spectral_type'] = cat_key(orig['spectral_type'])
    orig['galaxy_population'] = cat_key(orig['galaxy_population'])
    keep = RAW_NUM_COLS + ['spectral_type', 'galaxy_population', TARGET]
    return orig[[c for c in keep if c in orig.columns]].copy()


DATA_ROOT = find_competition_root()
ORIG_PATH = find_original_path()

train = read_competition_csv(DATA_ROOT / 'train.csv')
test = read_competition_csv(DATA_ROOT / 'test.csv')
orig = read_original_csv(ORIG_PATH)
sample_path = DATA_ROOT / 'sample_submission.csv'
sample = pd.read_csv(sample_path) if sample_path.exists() else None

y = train[TARGET].map(CLASS_TO_INT).astype('int8').reset_index(drop=True)
y_orig = orig[TARGET].map(CLASS_TO_INT).astype('int8').reset_index(drop=True)
test_ids = test[ID_COL].copy()

emit(f'competition root: {DATA_ROOT}')
emit(f'original dataset: {ORIG_PATH}')
emit(f'train/test/original: {train.shape} {test.shape} {orig.shape}')
emit(str(train[TARGET].value_counts(normalize=True).sort_index()))


# ---------------------------------------------------------------------------
# Feature engineering (pure pandas / numpy -- faithful port of the cuDF/cuPy source)
# ---------------------------------------------------------------------------
def to_np(s):
    # column -> float32 numpy with NaN for nulls (== cuDF to_cupy(na_value=nan))
    return np.asarray(s, dtype=np.float32)


def add_public_features(df, style="full"):
    out = df.copy()
    for c in RAW_NUM_COLS:
        out[c] = clean_num(out[c])

    color_pairs = [
        ("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"),
        ("u", "r"), ("u", "i"), ("u", "z"), ("g", "i"),
        ("g", "z"), ("r", "z"),
    ]
    for a, b in color_pairs:
        out[f"{a}_{b}"] = (out[a] - out[b]).astype("float32")

    band_values = out[BANDS].to_numpy(dtype=np.float32)  # (n, 5)
    # NOTE: cuDF source used cp.mean/std(ddof=1)/min/max (NOT nan-aware). Bands are
    # clean after the contract guard, so plain numpy ops match exactly.
    out["mag_mean"] = np.mean(band_values, axis=1).astype("float32")
    out["mag_std"] = np.std(band_values, axis=1, ddof=1).astype("float32")
    out["mag_min"] = np.min(band_values, axis=1).astype("float32")
    out["mag_max"] = np.max(band_values, axis=1).astype("float32")
    out["mag_range"] = (out["mag_max"] - out["mag_min"]).astype("float32")

    for b in BANDS:
        out[f"redshift_{b}"] = (out["redshift"] * out[b]).astype("float32")

    alpha_rad = out["alpha"].to_numpy(dtype=np.float32) * np.float32(np.pi / 180.0)
    delta_rad = out["delta"].to_numpy(dtype=np.float32) * np.float32(np.pi / 180.0)
    out["alpha_sin"] = np.sin(alpha_rad).astype("float32")
    out["alpha_cos"] = np.cos(alpha_rad).astype("float32")
    out["delta_sin"] = np.sin(delta_rad).astype("float32")
    out["delta_cos"] = np.cos(delta_rad).astype("float32")

    spectral_map = {"O/B": 0, "A": 1, "F": 2, "G": 3, "K": 4, "M": 5}
    out["spectral_ord"] = out["spectral_type"].map(spectral_map).fillna(-1).astype("float32")

    flux_arrays = []
    for b in BANDS:
        clipped = np.clip(out[b].to_numpy(dtype=np.float32), -30, 30)
        flux = np.power(np.float32(10.0), np.float32(-0.4) * clipped).astype(np.float32)
        out[f"flux_{b}"] = flux
        flux_arrays.append(flux)
    flux_values = np.vstack(flux_arrays).T  # (n, 5)
    out["flux_mean"] = np.mean(flux_values, axis=1).astype("float32")
    out["flux_std"] = np.std(flux_values, axis=1, ddof=1).astype("float32")
    out["flux_min"] = np.min(flux_values, axis=1).astype("float32")
    out["flux_max"] = np.max(flux_values, axis=1).astype("float32")
    out["flux_range"] = (out["flux_max"] - out["flux_min"]).astype("float32")

    x = np.arange(len(BANDS), dtype=np.float32)
    x_centered = x - x.mean()
    centered = band_values - np.mean(band_values, axis=1, keepdims=True)
    out["mag_slope"] = (centered.dot(x_centered) / np.sum(x_centered ** 2)).astype("float32")
    out["mag_curvature"] = (out["u"] - 2 * out["r"] + out["z"]).astype("float32")
    out["blue_curvature"] = (out["u"] - 2 * out["g"] + out["r"]).astype("float32")
    out["red_curvature"] = (out["r"] - 2 * out["i"] + out["z"]).astype("float32")

    for c in ["u_g", "g_r", "r_i", "i_z"]:
        out[f"{c}_per_redshift"] = (out[c] / (out["redshift"].abs() + EPS)).astype("float32")

    cat_cols = ["spectral_type", "galaxy_population"]

    if style == "full":
        out["mag_argmin"] = np.argmin(band_values, axis=1).astype("int16")
        out["mag_argmax"] = np.argmax(band_values, axis=1).astype("int16")
        redshift_cp = out["redshift"].to_numpy(dtype=np.float32)
        signed_redshift_denom = np.where(
            np.abs(redshift_cp) < EPS,
            np.where(redshift_cp < 0, np.float32(-EPS), np.float32(EPS)),
            redshift_cp,
        ).astype(np.float32)
        for b in BANDS:
            out[f"{b}_over_redshift"] = (out[b] / (out["redshift"].abs() + EPS)).astype("float32")
            out[f"{b}_over_redshift_signed"] = (out[b].to_numpy(dtype=np.float32) / signed_redshift_denom).astype("float32")
        out["sky_x"] = (np.cos(delta_rad) * np.cos(alpha_rad)).astype("float32")
        out["sky_y"] = (np.cos(delta_rad) * np.sin(alpha_rad)).astype("float32")
        out["sky_z"] = np.sin(delta_rad).astype("float32")
        out["redshift_abs"] = out["redshift"].abs().astype("float32")
        sky_x = out["sky_x"].to_numpy(dtype=np.float32)
        sky_y = out["sky_y"].to_numpy(dtype=np.float32)
        sky_z = out["sky_z"].to_numpy(dtype=np.float32)
        out["redshift_sky_x"] = (redshift_cp * sky_x).astype("float32")
        out["redshift_sky_y"] = (redshift_cp * sky_y).astype("float32")
        out["redshift_sky_z"] = (redshift_cp * sky_z).astype("float32")
        redshift_abs_cp = np.abs(redshift_cp)
        out["redshift_abs_sky_x"] = (redshift_abs_cp * sky_x).astype("float32")
        out["redshift_abs_sky_y"] = (redshift_abs_cp * sky_y).astype("float32")
        out["redshift_abs_sky_z"] = (redshift_abs_cp * sky_z).astype("float32")
        distmod_proxy = (np.float32(5.0) * np.log10(redshift_abs_cp + EPS)).astype(np.float32)
        out["redshift_distmod_proxy"] = distmod_proxy
        for b in BANDS:
            out[f"{b}_absmag_proxy"] = (out[b].to_numpy(dtype=np.float32) - distmod_proxy).astype("float32")
        out["mag_mean_absmag_proxy"] = (out["mag_mean"].to_numpy(dtype=np.float32) - distmod_proxy).astype("float32")
        out["redshift_log1p_abs"] = np.log1p(out["redshift_abs"].to_numpy(dtype=np.float32)).astype("float32")
        out["redshift_is_neg"] = (out["redshift"] < 0).astype("int8")
        phys_bins = np.asarray([-np.inf, 0.05, 0.10, 0.30, 0.60, np.inf], dtype=np.float32)
        phys_codes = np.searchsorted(phys_bins, redshift_cp, side="right") - 1
        phys_codes = np.where(np.isnan(redshift_cp), -1, phys_codes).astype(np.int8)
        out["redshift_phys_bin"] = pd.Series(phys_codes, index=out.index).astype("str")
        gr_cp = out["g_r"].to_numpy(dtype=np.float32)
        gr_bins = np.asarray([-np.inf, 0.0, 0.4, 0.8, 1.2, np.inf], dtype=np.float32)
        gr_codes = np.searchsorted(gr_bins, gr_cp, side="right") - 1
        gr_codes = np.where(np.isnan(gr_cp), -1, gr_codes).astype(np.int8)
        out["g_r_color_bin"] = pd.Series(gr_codes, index=out.index).astype("str")
        out["redshift_phys_x_g_r_color"] = cat_key(out["redshift_phys_bin"]) + "__" + cat_key(out["g_r_color_bin"])
        out["spectral_type_calc"] = cat_key(pd.Series(spectral_type_cut(out["g"], out["r"]).values, index=out.index))
        out["galaxy_population_calc"] = cat_key(pd.Series(galaxy_population_cut(out["u"], out["r"]).values, index=out.index))
        out["spectral_x_pop"] = cat_key(out["spectral_type"]) + "__" + cat_key(out["galaxy_population"])
        out["spectral_calc_x_pop_calc"] = cat_key(out["spectral_type_calc"]) + "__" + cat_key(out["galaxy_population_calc"])
        out["redshift_phys_x_spectral"] = cat_key(out["redshift_phys_bin"]) + "__" + cat_key(out["spectral_type"])
        out["redshift_phys_x_pop"] = cat_key(out["redshift_phys_bin"]) + "__" + cat_key(out["galaxy_population"])
        out["redshift_phys_x_spectral_pop"] = cat_key(out["redshift_phys_bin"]) + "__" + cat_key(out["spectral_x_pop"])
        for c in ["u_g", "g_r", "r_i", "i_z", "u_r", "g_i", "r_z"]:
            out[f"{c}_x_redshift"] = (out[c] * out["redshift"]).astype("float32")
            out[f"{c}_abs"] = out[c].abs().astype("float32")
            out[f"{c}_over_redshift_signed"] = (out[c].to_numpy(dtype=np.float32) / signed_redshift_denom).astype("float32")
        ug = out["u_g"].to_numpy(dtype=np.float32)
        gr = out["g_r"].to_numpy(dtype=np.float32)
        ri = out["r_i"].to_numpy(dtype=np.float32)
        iz = out["i_z"].to_numpy(dtype=np.float32)
        out["color_plane_radius_ug_gr"] = np.sqrt(ug ** 2 + gr ** 2).astype("float32")
        out["color_plane_angle_ug_gr"] = np.arctan2(ug, gr + EPS).astype("float32")
        out["color_plane_radius_ri_iz"] = np.sqrt(ri ** 2 + iz ** 2).astype("float32")
        out["color_plane_angle_ri_iz"] = np.arctan2(ri, iz + EPS).astype("float32")
        cat_cols = [
            "spectral_type", "galaxy_population", "spectral_type_calc", "galaxy_population_calc",
            "spectral_x_pop", "spectral_calc_x_pop_calc", "redshift_phys_bin", "g_r_color_bin",
            "redshift_phys_x_g_r_color", "redshift_phys_x_spectral", "redshift_phys_x_pop",
            "redshift_phys_x_spectral_pop",
        ]

    return out.replace([np.inf, -np.inf], np.nan), cat_cols


def qcut_codes(values, ref_values, q):
    # Faithful port of qcut_codes_gpu: probs = linspace(0,1,q+1) (incl endpoints),
    # bins = unique(quantile(ref, probs)), codes = searchsorted(side='left')-1, with
    # exact boundary/NaN handling. Returns int16 numpy codes.
    ref = np.asarray(ref_values, dtype=np.float32)
    ref = ref[~np.isnan(ref)]
    vals = np.asarray(values, dtype=np.float32)
    if len(ref) < 2:
        return np.full(len(vals), -1, dtype=np.int16)
    probs = np.linspace(0, 1, q + 1, dtype=np.float32)
    bins = np.quantile(ref, probs).astype(np.float32)
    bins = np.unique(bins)
    if len(bins) <= 1:
        return np.full(len(vals), -1, dtype=np.int16)
    codes = np.searchsorted(bins, vals, side="left") - 1
    codes = np.where(vals == bins[0], 0, codes)
    codes = np.where((vals < bins[0]) | (vals > bins[-1]) | np.isnan(vals), -1, codes)
    return np.clip(codes, -1, len(bins) - 2).astype(np.int16)


def _floor_string_codes(vals, scale=None):
    # Reproduces add_lowfreq / add_color artifact cat logic:
    #   floor(where(finite, vals*scale_or_1, 0)) -> int32; non-finite -> INT32_MIN;
    #   then .astype('str').
    vals = np.asarray(vals, dtype=np.float32)
    finite = np.isfinite(vals)
    src = vals if scale is None else (vals * np.float32(scale))
    floored = np.floor(np.where(finite, src, np.float32(0.0))).astype(np.int64)
    floored = np.where(finite, floored, np.int64(INT32_MIN)).astype(np.int32)
    return pd.Series(floored).astype("str")


def add_lowfreq_artifact_features(df):
    out = df.copy()
    cat_cols = []
    for c in RAW_NUM_COLS:
        vals = clean_num(out[c]).to_numpy(dtype=np.float32)
        name = f"art_{c}_floor"
        out[name] = pd.Series(_floor_string_codes(vals).values, index=out.index)
        cat_cols.append(name)
    out["art_alpha_floor_x_delta_floor"] = cat_key(out["art_alpha_floor"]) + "__" + cat_key(out["art_delta_floor"])
    out["art_u_floor_x_z_floor"] = cat_key(out["art_u_floor"]) + "__" + cat_key(out["art_z_floor"])
    cat_cols.extend(ART_LOWFREQ_PAIR_COLS)
    return out, cat_cols


def add_color_artifact_features(df):
    out = df.copy()
    cat_cols = []
    for c, scale, tag in ART_COLOR_BIN_SPECS:
        if c not in out.columns:
            continue
        vals = clean_num(out[c]).to_numpy(dtype=np.float32)
        name = f"art_{c}_{tag}"
        out[name] = pd.Series(_floor_string_codes(vals, scale=scale).values, index=out.index)
        cat_cols.append(name)
    if "art_u_g_half" in out.columns and "art_redshift_tenth" in out.columns:
        out["art_u_g_half_x_redshift_tenth"] = cat_key(out["art_u_g_half"]) + "__" + cat_key(out["art_redshift_tenth"])
        cat_cols.append("art_u_g_half_x_redshift_tenth")
    if "art_g_r_half" in out.columns and "art_redshift_tenth" in out.columns:
        out["art_g_r_half_x_redshift_tenth"] = cat_key(out["art_g_r_half"]) + "__" + cat_key(out["art_redshift_tenth"])
        cat_cols.append("art_g_r_half_x_redshift_tenth")
    if "art_alpha_deg5" in out.columns and "art_delta_deg5" in out.columns:
        out["art_alpha_deg5_x_delta_deg5"] = cat_key(out["art_alpha_deg5"]) + "__" + cat_key(out["art_delta_deg5"])
        cat_cols.append("art_alpha_deg5_x_delta_deg5")
    return out, cat_cols


def add_quantile_bin_features(df, train_test_mask, extra_qbins=None):
    out = df.copy()
    qbin_cols = []
    cols = RAW_NUM_COLS + [c for c in ["u_g", "g_r", "r_i", "i_z", "u_r", "mag_mean", "mag_range"] if c in out.columns]
    qbins = sorted(set([16, 64, 256] + list(extra_qbins or [])))
    mask = np.asarray(train_test_mask, dtype=bool)
    for c in cols:
        s = clean_num(out[c]).to_numpy(dtype=np.float32)
        ref = s[mask]
        for q in qbins:
            name = f"{c}_qbin{q}"
            codes = qcut_codes(s, ref, q)
            out[name] = pd.Series(codes.astype(np.int16), index=out.index).astype("int16").astype("str")
            qbin_cols.append(name)
    for a, b in [("alpha_qbin64", "delta_qbin64"), ("u_g_qbin64", "g_r_qbin64"), ("redshift_qbin64", "mag_mean_qbin64")]:
        if a in out.columns and b in out.columns:
            name = f"{a}__x__{b}"
            out[name] = cat_key(out[a]) + "__" + cat_key(out[b])
            qbin_cols.append(name)
    return out, qbin_cols


def select_te_cols(df, cat_cols, source, max_card):
    cols = []
    for c in cat_cols:
        if c not in df.columns:
            continue
        card = int(cat_key(df[c]).nunique(dropna=False))
        if card > max_card:
            continue
        if source == "core":
            keep = (
                c in ["spectral_type", "galaxy_population", "spectral_type_calc", "galaxy_population_calc", "spectral_x_pop", "spectral_calc_x_pop_calc"]
                or c.endswith("_qbin16")
                or c.endswith("_qbin64")
                or "_rkey" in c
                or "__x__" in c
            )
        elif source == "qbin16":
            keep = c.endswith("_qbin16") or c in ["spectral_type", "galaxy_population", "spectral_type_calc", "galaxy_population_calc"]
        else:
            keep = True
        if keep:
            cols.append(c)
    return cols


def add_frequency_features(df, cols, fit_mask):
    out = df.copy()
    mask = np.asarray(fit_mask, dtype=bool)
    for c in cols:
        s = cat_key(out[c])
        vc = s[mask].value_counts(dropna=False)
        freq = s.map(vc).fillna(0).astype("float32")
        out[f"{c}_freq"] = freq
        out[f"{c}_freq_log1p"] = np.log1p(freq.to_numpy(dtype=np.float32)).astype("float32")
    return out


def add_original_prior_features(df, cols, orig_mask, orig_y, smooth=0.0):
    out = df.copy()
    omask = np.asarray(orig_mask, dtype=bool)
    orig_y_np = np.asarray(orig_y, dtype=np.int32)
    prior_counts = np.bincount(orig_y_np, minlength=len(CLASSES)).astype(np.float32)
    prior = prior_counts / max(float(prior_counts.sum()), 1.0)
    smooth = float(smooth or 0.0)
    smooth_tag = int(round(smooth)) if smooth else 0
    for c in cols:
        key = cat_key(out[c])
        key_orig = key[omask].reset_index(drop=True)
        tmp = pd.DataFrame({"key": key_orig.values, "y": orig_y_np})
        counts = tmp.groupby("key").size()
        out[f"orig_{c}_count"] = key.map(counts).fillna(0).astype("float32")
        for cls_idx, cls_name in INT_TO_CLASS.items():
            hit = (tmp["y"] == cls_idx).astype("float32")
            rates = tmp.assign(hit=hit).groupby("key")["hit"].mean()
            prior_val = float(prior[cls_idx])
            out[f"orig_{c}_prior_{cls_name}"] = key.map(rates).fillna(prior_val).astype("float32")
            if smooth > 0:
                smooth_rates = ((rates * counts.astype("float32")) + np.float32(smooth) * np.float32(prior_val)) / (counts.astype("float32") + np.float32(smooth))
                out[f"orig_{c}_smooth{smooth_tag}_prior_{cls_name}"] = key.map(smooth_rates).fillna(prior_val).astype("float32")
    return out


def build_feature_matrix(train, test, orig):
    train_base = train.drop(columns=[TARGET]).copy()
    test_base = test.copy()
    orig_base = orig.drop(columns=[TARGET]).copy()
    train_base['_source'] = 'train'
    test_base['_source'] = 'test'
    orig_base['_source'] = 'orig'

    all_df = pd.concat([train_base, test_base, orig_base], axis=0, ignore_index=True)
    all_df, cat_cols = add_public_features(all_df, 'full')
    train_test_mask = all_df['_source'].isin(['train', 'test'])

    all_df, artifact_cols = add_lowfreq_artifact_features(all_df)
    cat_cols += artifact_cols
    all_df, artifact_cols = add_color_artifact_features(all_df)
    cat_cols += artifact_cols
    all_df, qbin_cols = add_quantile_bin_features(all_df, train_test_mask, extra_qbins=[])
    cat_cols += qbin_cols

    cat_cols = [c for c in dict.fromkeys(cat_cols) if c in all_df.columns]

    freq_cols = select_te_cols(all_df, cat_cols, TE_SOURCE, max_card=TE_MAX_CARDINALITY * 4)
    all_df = add_frequency_features(all_df, freq_cols, all_df['_source'].isin(['train', 'test', 'orig']))

    orig_mask = all_df['_source'].eq('orig')
    prior_cols = select_te_cols(all_df, cat_cols, TE_SOURCE, max_card=TE_MAX_CARDINALITY * 2)
    all_df = add_original_prior_features(all_df, prior_cols, orig_mask, y_orig, smooth=0.0)

    all_df['is_orig'] = all_df['_source'].eq('orig').astype('int8')
    all_df['is_test'] = all_df['_source'].eq('test').astype('int8')
    all_df = all_df.drop(columns=[c for c in [ID_COL, '_source'] if c in all_df.columns]).replace([np.inf, -np.inf], np.nan)

    n_train = len(train_base)
    n_test = len(test_base)
    X = all_df.iloc[:n_train].reset_index(drop=True)
    X_test = all_df.iloc[n_train:n_train + n_test].reset_index(drop=True)
    X_orig = all_df.iloc[n_train + n_test:].reset_index(drop=True)
    cat_cols = [c for c in cat_cols if c in X.columns]

    del all_df, train_base, test_base, orig_base
    gc.collect()
    return X, X_test, X_orig, cat_cols


X, X_test, X_orig, cat_cols = build_feature_matrix(train, test, orig)
emit(f'base matrices: {X.shape} {X_test.shape} {X_orig.shape} cat_cols: {len(cat_cols)}')

del train, test, orig
gc.collect()


# ---------------------------------------------------------------------------
# Selected features (verbatim from source EXP3-030 gain rerank)
# ---------------------------------------------------------------------------
TOP_FEATURES = ['TE_art_g_r_half_x_redshift_tenth_STAR',
 'TE_art_g_r_half_x_redshift_tenth_QSO',
 'orig_art_g_r_half_x_redshift_tenth_prior_QSO',
 'redshift_u',
 'z_over_redshift',
 'TE_art_u_g_half_x_redshift_tenth_QSO',
 'TE_art_g_r_half_x_redshift_tenth_GALAXY',
 'g_z',
 'g_i',
 'orig_g_qbin64_prior_QSO',
 'redshift_g',
 'orig_g_qbin16_prior_QSO',
 'i_over_redshift',
 'redshift_abs',
 'TE_redshift_qbin16_GALAXY',
 'redshift_log1p_abs',
 'u_over_redshift',
 'TE_redshift_qbin64__x__mag_mean_qbin64_GALAXY',
 'g_i_abs',
 'orig_art_g_floor_prior_QSO',
 'u_i',
 'TE_redshift_qbin64_GALAXY',
 'orig_art_u_floor_x_z_floor_prior_QSO',
 'TE_alpha_qbin64__x__delta_qbin64_STAR',
 'art_redshift_floor_freq',
 'g_over_redshift',
 'u_r_abs',
 'redshift_is_neg',
 'orig_g_qbin256_prior_QSO',
 'TE_art_redshift_tenth_GALAXY',
 'TE_alpha_qbin64__x__delta_qbin64_GALAXY',
 'TE_redshift_qbin64_QSO',
 'art_redshift_floor',
 'orig_art_u_g_half_x_redshift_tenth_prior_QSO',
 'u_r',
 'mag_slope',
 'art_g_r_half_x_redshift_tenth_freq_log1p',
 'r_over_redshift',
 'g_qbin16',
 'TE_art_u_floor_x_z_floor_QSO',
 'TE_u_g_qbin64__x__g_r_qbin64_STAR',
 'orig_redshift_qbin64_prior_STAR',
 'flux_std',
 'redshift',
 'TE_art_u_g_half_x_redshift_tenth_GALAXY',
 'orig_redshift_qbin64_prior_GALAXY',
 'orig_g_qbin16_prior_GALAXY',
 'flux_g',
 'orig_redshift_qbin64__x__mag_mean_qbin64_prior_GALAXY',
 'mag_std',
 'TE_art_u_floor_x_z_floor_GALAXY',
 'redshift_z',
 'orig_redshift_qbin64__x__mag_mean_qbin64_prior_QSO',
 'g',
 'orig_redshift_qbin256_prior_GALAXY',
 'orig_art_i_floor_prior_QSO',
 'flux_range',
 'orig_art_redshift_floor_prior_GALAXY',
 'orig_art_r_floor_prior_QSO',
 'orig_alpha_qbin64__x__delta_qbin64_prior_STAR',
 'art_r_floor',
 'TE_redshift_qbin64_STAR',
 'TE_g_qbin64_QSO',
 'art_g_r_half_x_redshift_tenth_freq',
 'orig_alpha_qbin64__x__delta_qbin64_prior_GALAXY',
 'orig_art_i_floor_count',
 'TE_u_r_qbin64_GALAXY',
 'u_g',
 'TE_g_qbin16_QSO',
 'g_r_x_redshift',
 'z',
 'redshift_r',
 'r_z',
 'art_g_r_half_x_redshift_tenth',
 'mag_range',
 'i',
 'TE_art_u_g_half_x_redshift_tenth_STAR',
 'art_g_floor',
 'flux_i',
 'TE_art_redshift_tenth_STAR',
 'orig_r_qbin16_prior_QSO',
 'redshift_i',
 'r_z_x_redshift',
 'flux_z',
 'g_i_x_redshift',
 'orig_art_u_floor_x_z_floor_prior_GALAXY',
 'r',
 'TE_i_qbin64_QSO',
 'r_z_abs',
 'orig_g_qbin64_prior_GALAXY',
 'TE_redshift_qbin64__x__mag_mean_qbin64_QSO',
 'art_redshift_floor_freq_log1p',
 'orig_i_qbin16_prior_QSO',
 'orig_art_u_r_one_prior_QSO',
 'orig_mag_range_qbin16_prior_STAR',
 'orig_art_alpha_deg5_x_delta_deg5_prior_GALAXY',
 'TE_redshift_qbin16_STAR',
 'orig_art_redshift_tenth_prior_GALAXY',
 'TE_art_alpha_deg5_x_delta_deg5_STAR',
 'flux_r',
 'orig_art_z_floor_count',
 'TE_u_r_qbin64_QSO',
 'orig_art_alpha_deg5_x_delta_deg5_prior_STAR',
 'orig_u_qbin16_prior_QSO',
 'r_i_x_redshift',
 'art_i_floor',
 'orig_mag_range_qbin64_prior_STAR',
 'orig_z_qbin16_prior_QSO',
 'orig_redshift_qbin16_prior_GALAXY',
 'art_u_g_half',
 'art_z_floor_freq',
 'TE_art_g_floor_QSO',
 'art_r_i_half_freq_log1p',
 'orig_mag_range_qbin16_prior_GALAXY',
 'mag_max',
 'flux_min',
 'TE_g_qbin64_GALAXY',
 'orig_art_u_g_half_x_redshift_tenth_prior_STAR',
 'redshift_qbin64__x__mag_mean_qbin64',
 'TE_alpha_qbin64__x__delta_qbin64_QSO',
 'orig_i_qbin256_prior_QSO',
 'art_u_g_half_x_redshift_tenth_freq_log1p',
 'orig_art_i_floor_prior_GALAXY',
 'TE_mag_range_qbin64_QSO',
 'orig_art_r_i_half_count',
 'art_z_floor_freq_log1p',
 'color_plane_radius_ug_gr',
 'art_g_r_half',
 'art_r_i_half_freq',
 'TE_g_qbin16_GALAXY',
 'TE_redshift_qbin64__x__mag_mean_qbin64_STAR',
 'TE_art_r_floor_QSO',
 'flux_u',
 'flux_max',
 'orig_i_qbin64_prior_QSO',
 'u',
 'TE_art_alpha_deg5_x_delta_deg5_GALAXY',
 'flux_mean',
 'orig_u_g_qbin16_prior_STAR',
 'u_g_abs',
 'orig_art_u_floor_prior_QSO',
 'mag_mean_qbin16',
 'TE_art_alpha_floor_STAR',
 'art_u_floor_x_z_floor',
 'u_z',
 'redshift_qbin64__x__mag_mean_qbin64_freq_log1p',
 'orig_art_alpha_deg5_x_delta_deg5_prior_QSO',
 'orig_art_g_r_half_x_redshift_tenth_prior_STAR',
 'orig_mag_range_qbin256_prior_QSO',
 'mag_min',
 'orig_u_r_qbin16_prior_QSO',
 'orig_art_redshift_tenth_count',
 'r_i',
 'redshift_qbin16',
 'TE_i_qbin16_QSO',
 'TE_u_g_qbin64_STAR',
 'art_u_g_half_x_redshift_tenth_freq',
 'TE_art_r_floor_GALAXY',
 'redshift_qbin64__x__mag_mean_qbin64_freq',
 'TE_mag_range_qbin64_GALAXY',
 'u_qbin16',
 'art_i_floor_freq',
 'mag_range_qbin256',
 'orig_mag_range_qbin256_prior_STAR',
 'art_i_floor_freq_log1p',
 'art_u_g_half_x_redshift_tenth',
 'orig_z_qbin64_prior_QSO',
 'art_r_floor_freq_log1p',
 'alpha_sin',
 'orig_art_r_floor_prior_GALAXY',
 'TE_r_i_qbin64_QSO',
 'orig_u_qbin16_prior_STAR',
 'color_plane_radius_ri_iz',
 'TE_art_alpha_deg5_x_delta_deg5_QSO',
 'delta_cos',
 'TE_r_i_qbin16_GALAXY',
 'art_g_floor_freq',
 'orig_art_u_g_half_x_redshift_tenth_count',
 'orig_delta_qbin256_prior_STAR',
 'TE_r_i_qbin16_QSO',
 'orig_art_u_floor_x_z_floor_prior_STAR',
 'mag_range_qbin16',
 'art_delta_deg5',
 'TE_u_g_qbin16_STAR',
 'u_r_x_redshift',
 'sky_y',
 'TE_r_i_qbin64_GALAXY',
 'art_g_floor_freq_log1p',
 'orig_art_alpha_floor_prior_STAR',
 'orig_delta_qbin256_prior_GALAXY',
 'u_g_x_redshift',
 'g_r',
 'r_qbin16',
 'TE_z_qbin64_QSO',
 'art_redshift_tenth_freq_log1p',
 'alpha_qbin256',
 'orig_i_qbin16_prior_GALAXY',
 'orig_art_alpha_floor_prior_GALAXY',
 'orig_art_r_i_half_prior_GALAXY',
 'orig_art_g_floor_prior_GALAXY',
 'orig_redshift_qbin256_count',
 'TE_g_qbin64_STAR',
 'orig_u_qbin16_count',
 'art_redshift_tenth_freq',
 'z_qbin16',
 'r_qbin64',
 'TE_u_qbin16_QSO',
 'art_u_r_one_freq_log1p',
 'orig_u_qbin64_prior_QSO',
 'orig_art_alpha_deg5_prior_STAR',
 'g_qbin64',
 'TE_r_qbin16_QSO',
 'i_qbin16',
 'orig_z_qbin256_prior_QSO',
 'orig_art_delta_deg5_prior_GALAXY',
 'orig_g_qbin256_prior_GALAXY',
 'blue_curvature',
 'TE_u_qbin64_QSO',
 'orig_art_u_g_half_x_redshift_tenth_prior_GALAXY',
 'TE_art_redshift_tenth_QSO',
 'alpha',
 'art_u_r_one_freq',
 'orig_mag_range_qbin256_prior_GALAXY',
 'delta_sin',
 'art_redshift_tenth',
 'TE_art_alpha_floor_GALAXY',
 'art_alpha_floor',
 'delta',
 'TE_art_u_r_one_GALAXY',
 'color_plane_angle_ug_gr',
 'orig_art_redshift_tenth_prior_QSO',
 'orig_art_r_i_half_prior_QSO',
 'orig_art_g_floor_prior_STAR',
 'mag_curvature',
 'TE_art_u_floor_x_z_floor_STAR',
 'art_delta_floor',
 'u_qbin16_freq_log1p',
 'TE_g_r_qbin64_GALAXY',
 'TE_r_qbin64_STAR',
 'sky_z',
 'art_u_floor_freq',
 'mag_mean',
 'g_r_abs',
 'TE_z_qbin16_QSO',
 'sky_x',
 'u_qbin16_freq',
 'TE_u_qbin64_STAR',
 'orig_art_u_floor_count',
 'art_alpha_floor_x_delta_floor',
 'art_u_floor_freq_log1p',
 'orig_z_qbin16_count',
 'art_u_r_one',
 'art_u_g_half_freq',
 'TE_redshift_qbin16_QSO',
 'orig_alpha_qbin64__x__delta_qbin64_prior_QSO',
 'z_qbin16_freq',
 'TE_u_g_qbin64_QSO',
 'TE_u_g_qbin64_GALAXY',
 'orig_g_qbin64_prior_STAR',
 'orig_art_g_r_half_prior_GALAXY',
 'TE_g_qbin16_STAR',
 'orig_art_g_r_half_x_redshift_tenth_count',
 'orig_art_delta_floor_prior_QSO',
 'orig_art_u_r_one_count',
 'orig_spectral_x_pop_prior_QSO',
 'redshift_qbin256',
 'orig_art_u_g_half_prior_STAR',
 'TE_g_r_qbin64_STAR',
 'orig_art_u_r_one_prior_STAR',
 'art_delta_floor_freq_log1p',
 'TE_art_alpha_floor_QSO',
 'art_delta_floor_freq',
 'art_delta_deg5_freq',
 'art_u_g_half_freq_log1p',
 'art_delta_deg5_freq_log1p',
 'z_qbin16_freq_log1p',
 'TE_u_r_qbin16_GALAXY',
 'TE_art_i_floor_GALAXY',
 'orig_art_alpha_floor_prior_QSO',
 'orig_art_alpha_deg5_prior_GALAXY',
 'orig_art_u_g_half_count',
 'orig_art_g_floor_count',
 'TE_art_i_floor_QSO',
 'art_alpha_deg5_freq_log1p',
 'art_r_floor_freq',
 'art_alpha_deg5_freq',
 'orig_art_alpha_deg5_count',
 'TE_art_g_floor_GALAXY',
 'orig_u_r_qbin64_prior_QSO',
 'TE_u_g_qbin64__x__g_r_qbin64_QSO',
 'orig_art_u_floor_prior_STAR',
 'art_alpha_deg5_x_delta_deg5',
 'orig_art_delta_floor_prior_GALAXY',
 'art_alpha_deg5',
 'orig_art_delta_deg5_count',
 'orig_u_r_qbin256_prior_QSO',
 'orig_art_alpha_deg5_prior_QSO',
 'orig_art_g_r_half_prior_QSO',
 'TE_art_alpha_deg5_STAR',
 'orig_art_delta_floor_count',
 'TE_g_r_qbin64_QSO',
 'art_alpha_deg5_x_delta_deg5_freq',
 'art_u_floor',
 'art_alpha_deg5_x_delta_deg5_freq_log1p',
 'orig_art_delta_deg5_prior_QSO',
 'art_alpha_floor_freq_log1p',
 'art_r_i_half',
 'orig_art_r_i_half_prior_STAR',
 'orig_u_g_qbin64_prior_STAR',
 'art_alpha_floor_freq',
 'orig_art_g_r_half_x_redshift_tenth_prior_GALAXY',
 'TE_art_alpha_deg5_GALAXY',
 'orig_art_u_g_half_prior_GALAXY',
 'orig_art_u_floor_x_z_floor_count',
 'orig_art_alpha_floor_count',
 'TE_art_delta_floor_QSO',
 'TE_art_u_floor_QSO',
 'TE_art_delta_deg5_GALAXY',
 'orig_art_alpha_deg5_x_delta_deg5_count',
 'orig_art_i_floor_prior_STAR',
 'orig_art_redshift_floor_prior_QSO',
 'TE_art_delta_floor_GALAXY',
 'TE_art_alpha_deg5_QSO',
 'orig_alpha_qbin256_prior_GALAXY',
 'TE_u_g_qbin64__x__g_r_qbin64_GALAXY',
 'orig_art_delta_deg5_prior_STAR',
 'art_u_floor_x_z_floor_freq_log1p',
 'TE_art_r_floor_STAR',
 'orig_art_z_floor_prior_GALAXY',
 'TE_art_u_g_half_GALAXY',
 'TE_art_g_floor_STAR',
 'TE_art_redshift_floor_STAR',
 'orig_art_r_floor_count',
 'TE_art_r_i_half_STAR',
 'art_u_floor_x_z_floor_freq',
 'TE_art_i_floor_STAR',
 'TE_art_delta_deg5_QSO',
 'TE_art_u_floor_STAR',
 'orig_art_delta_floor_prior_STAR',
 'TE_art_r_i_half_GALAXY',
 'TE_art_u_g_half_QSO',
 'orig_art_u_g_half_prior_QSO',
 'art_alpha_floor_x_delta_floor_freq',
 'orig_art_r_floor_prior_STAR',
 'orig_art_i_z_half_prior_QSO',
 'TE_art_delta_deg5_STAR',
 'orig_alpha_qbin256_prior_STAR',
 'art_alpha_floor_x_delta_floor_freq_log1p',
 'TE_art_z_floor_STAR',
 'orig_art_z_floor_prior_QSO',
 'orig_art_u_floor_prior_GALAXY',
 'orig_art_u_r_one_prior_GALAXY',
 'TE_art_g_r_half_QSO',
 'art_g_r_half_freq_log1p',
 'TE_art_u_r_one_QSO',
 'TE_art_u_g_half_STAR',
 'TE_art_r_i_half_QSO',
 'art_z_floor',
 'TE_art_z_floor_GALAXY',
 'TE_art_z_floor_QSO',
 'TE_art_u_floor_GALAXY',
 'TE_art_g_r_half_GALAXY',
 'TE_art_delta_floor_STAR',
 'orig_art_g_r_half_prior_STAR',
 'TE_art_g_r_half_STAR',
 'TE_art_u_r_one_STAR',
 'orig_art_z_floor_prior_STAR',
 'TE_art_redshift_floor_GALAXY',
 'TE_art_redshift_floor_QSO',
 'TE_art_i_z_half_QSO']

emit(f'Selected {len(TOP_FEATURES)} top features from EXP3-030 gain rerank.')


# ---------------------------------------------------------------------------
# Fold-safe target encoding (pure pandas/numpy port of cuML TargetEncoder)
# ---------------------------------------------------------------------------
def sorted_factorize_three(train_s, valid_s, test_s):
    # Concatenate string keys, build a vocabulary sorted lexicographically, map to
    # int32 codes (unseen -> -1). Mirrors the cuDF sorted_factorize_three exactly.
    tr = cat_key(train_s).reset_index(drop=True)
    va = cat_key(valid_s).reset_index(drop=True)
    te = cat_key(test_s).reset_index(drop=True)
    vals = pd.concat([tr, va, te], ignore_index=True)
    cats = pd.Index(pd.unique(vals)).sort_values()
    mapper = pd.Series(np.arange(len(cats), dtype=np.int32), index=cats)
    codes = vals.map(mapper).fillna(-1).astype('int32').reset_index(drop=True)
    n_tr, n_va = len(tr), len(va)
    return (
        codes.iloc[:n_tr].reset_index(drop=True),
        codes.iloc[n_tr:n_tr + n_va].reset_index(drop=True),
        codes.iloc[n_tr + n_va:].reset_index(drop=True),
    )


def make_inner_fold_ids(y_train):
    y_cpu = np.asarray(y_train)
    fold_ids = np.empty(len(y_cpu), dtype=np.int32)
    inner = StratifiedKFold(n_splits=TE_INNER_SPLITS, shuffle=True, random_state=SEED + 177)
    for fold_id, (_, va_idx) in enumerate(inner.split(np.zeros(len(y_cpu), dtype=np.int8), y_cpu)):
        fold_ids[va_idx] = fold_id
    return fold_ids


def _te_encode_column(tr_codes, y_bin, fold_ids, va_codes, te_codes, global_mean):
    # Reproduces cuML TargetEncoder(smooth=TE_SMOOTH, n_folds=TE_INNER_SPLITS,
    # split_method='customize'):
    #   fit_transform : leave-one-inner-fold-out smoothing toward global_mean
    #                   enc = (sum_excl + smooth*gmean) / (count_excl + smooth)
    #   transform     : full-train-fold stats
    #                   enc = (sum_all + smooth*gmean) / (count_all + smooth)
    #                   unseen category -> global_mean
    smooth = np.float32(TE_SMOOTH)
    tr_codes = np.asarray(tr_codes, dtype=np.int64)
    y_bin = np.asarray(y_bin, dtype=np.float64)
    n_cat = int(tr_codes.max()) + 1 if len(tr_codes) and tr_codes.max() >= 0 else 0

    # totals per category over the whole fold-train set
    total_count = np.bincount(tr_codes[tr_codes >= 0], minlength=n_cat).astype(np.float64)
    total_sum = np.bincount(tr_codes[tr_codes >= 0], weights=y_bin[tr_codes >= 0], minlength=n_cat).astype(np.float64)

    # per-(category, fold) stats for leave-one-fold-out
    n_folds = TE_INNER_SPLITS
    fold_count = np.zeros((n_cat, n_folds), dtype=np.float64)
    fold_sum = np.zeros((n_cat, n_folds), dtype=np.float64)
    valid = tr_codes >= 0
    np.add.at(fold_count, (tr_codes[valid], fold_ids[valid]), 1.0)
    np.add.at(fold_sum, (tr_codes[valid], fold_ids[valid]), y_bin[valid])

    gmean = np.float64(global_mean)

    # train encoding: exclude the row's own fold
    tr_enc = np.full(len(tr_codes), gmean, dtype=np.float64)
    if n_cat > 0:
        c = tr_codes
        f = fold_ids
        cnt_excl = np.where(c >= 0, total_count[np.clip(c, 0, n_cat - 1)] - fold_count[np.clip(c, 0, n_cat - 1), f], 0.0)
        sum_excl = np.where(c >= 0, total_sum[np.clip(c, 0, n_cat - 1)] - fold_sum[np.clip(c, 0, n_cat - 1), f], 0.0)
        enc = (sum_excl + float(smooth) * gmean) / (cnt_excl + float(smooth))
        tr_enc = np.where(c >= 0, enc, gmean)

    # transform encoding (valid / test): full-train stats per category
    def transform(codes):
        codes = np.asarray(codes, dtype=np.int64)
        out = np.full(len(codes), gmean, dtype=np.float64)
        if n_cat == 0:
            return out.astype(np.float32)
        m = (codes >= 0) & (codes < n_cat)
        cc = codes[m]
        enc = (total_sum[cc] + float(smooth) * gmean) / (total_count[cc] + float(smooth))
        out[m] = enc
        return out.astype(np.float32)

    return tr_enc.astype(np.float32), transform(va_codes), transform(te_codes)


def add_fold_safe_te(X_train, y_train, X_valid, X_test_fold, te_cols):
    if not te_cols:
        return X_train, X_valid, X_test_fold, []
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test_fold = X_test_fold.copy()
    fold_ids = make_inner_fold_ids(y_train)
    y_np = np.asarray(y_train, dtype=np.int32)
    added = []

    for c in te_cols:
        if c not in X_train.columns:
            continue
        tr_codes, va_codes, te_codes = sorted_factorize_three(X_train[c], X_valid[c], X_test_fold[c])
        tr_codes = tr_codes.to_numpy()
        va_codes = va_codes.to_numpy()
        te_codes = te_codes.to_numpy()
        for cls_idx, cls_name in INT_TO_CLASS.items():
            y_bin = (y_np == cls_idx).astype(np.float32)
            global_mean = float(y_bin.mean())
            tr_vals, va_vals, te_vals = _te_encode_column(tr_codes, y_bin, fold_ids, va_codes, te_codes, global_mean)
            name = f'TE_{c}_{cls_name}'
            X_train[name] = tr_vals
            X_valid[name] = va_vals
            X_test_fold[name] = te_vals
            added.append(name)
    return X_train, X_valid, X_test_fold, added


def encode_model_categories(X_train, X_valid, X_test_fold, model_cat_cols):
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test_fold = X_test_fold.copy()
    for c in model_cat_cols:
        if c not in X_train.columns:
            continue
        tr_codes, va_codes, te_codes = sorted_factorize_three(X_train[c], X_valid[c], X_test_fold[c])
        X_train[c] = tr_codes.to_numpy()
        X_valid[c] = va_codes.to_numpy()
        X_test_fold[c] = te_codes.to_numpy()
    return X_train, X_valid, X_test_fold


def te_sources_needed_for_top_features(top_features, available_te_cols):
    needed = []
    for c in available_te_cols:
        prefix = f'TE_{c}_'
        if any(str(f).startswith(prefix) for f in top_features):
            needed.append(c)
    return needed


available_te_cols = select_te_cols(X, cat_cols, TE_SOURCE, TE_MAX_CARDINALITY)
TE_COLS = te_sources_needed_for_top_features(TOP_FEATURES, available_te_cols)
MODEL_CAT_COLS = [c for c in cat_cols if c in TOP_FEATURES]
emit(f'target-encoding sources pruned: {len(available_te_cols)} -> {len(TE_COLS)}')
emit(f'raw categorical features selected for model: {len(MODEL_CAT_COLS)}')
emit(str(TE_COLS[:25]))


# ===========================================================================
# OvR training: 3 binary XGBoost models per fold (class k vs rest).
# ---------------------------------------------------------------------------
# Everything above (FE / TE / priors / TOP_FEATURES / fold-safe TE) is the
# xgbv5 pipeline copied VERBATIM. Below is the only behavioral change: instead
# of one multiclass model per fold we train THREE binary:logistic models, take
# each model's P(class=k), stack into a 3-column array, and ROW-NORMALIZE into a
# valid 3-class probability. The FE/TE matrices (X_tr_np/X_va_np/X_te_np) are
# built ONCE per fold and shared across the 3 binary models, so the features and
# fold contract are identical to xgbv5.
# ===========================================================================

# Lowered iteration cap vs the xgbv5 multiclass source (7000 / es=180): with 3
# binary models per fold x 5 folds we cap each binary model at 5000 trees with
# early_stopping_rounds=150 to stay within Kaggle's GPU time budget. Binary
# trees are also cheaper than multiclass (single output vs 3), so this is ample
# headroom for convergence.
OVR_N_ESTIMATORS = 5000
OVR_EARLY_STOPPING_ROUNDS = 150


def make_xgb_binary_params(seed):
    # Base hyperparameters copied VERBATIM from the xgbv5 multiclass params, with
    # the objective swapped to binary:logistic, num_class / multiclass balanced-
    # error eval removed, eval_metric set to 'logloss' (for early stopping on each
    # binary model's own validation logloss), and the iteration cap lowered.
    return {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'learning_rate': 0.012,
        'n_estimators': OVR_N_ESTIMATORS,
        'early_stopping_rounds': OVR_EARLY_STOPPING_ROUNDS,
        'max_depth': 0,
        'max_leaves': 72,
        'grow_policy': 'lossguide',
        'max_bin': 960,
        'min_child_weight': 10,
        'gamma': 0.2,
        'reg_alpha': 0.30,
        'reg_lambda': 4.0,
        'subsample': 0.82,
        'colsample_bytree': 0.74,
        'colsample_bylevel': 0.86,
        'random_state': seed,
        'n_jobs': 4,
    }


def normalize_clip(p):
    # Row-normalize 3-class probabilities for stacking: clip to [1e-7, 1] then
    # divide each row by its sum so the row is a valid distribution.
    p = np.asarray(p, dtype=np.float32)
    p = np.clip(p, np.float32(1e-7), np.float32(1.0))
    p = p / p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


def prepare_selected(X_train, X_valid, X_test_fold):
    # Verbatim from xgbv5: encode the raw categorical TOP features to int codes
    # (sorted_factorize over train+valid+test) and slice/cast TOP_FEATURES to a
    # float32 matrix. Built ONCE per fold and shared across the 3 binary models so
    # the per-fold feature matrix is identical to the xgbv5 multiclass source.
    X_train, X_valid, X_test_fold = encode_model_categories(X_train, X_valid, X_test_fold, MODEL_CAT_COLS)
    missing = [c for c in TOP_FEATURES if c not in X_train.columns]
    if missing:
        emit(f'Missing {len(missing)} top features; first missing: {missing[:10]}')
    features = [c for c in TOP_FEATURES if c in X_train.columns]
    X_train_np = X_train[features].astype('float32').to_numpy(dtype=np.float32)
    X_valid_np = X_valid[features].astype('float32').to_numpy(dtype=np.float32)
    X_test_np = X_test_fold[features].astype('float32').to_numpy(dtype=np.float32)
    return X_train_np, X_valid_np, X_test_np, features


y_cpu = y.to_numpy()
oof = np.zeros((len(X), len(CLASSES)), dtype='float32')        # raw stacked binary P(class=k)
oof_fold = np.full(len(X), -1, dtype='int8')
test_pred_sum = np.zeros((len(X_test), len(CLASSES)), dtype='float32')  # fold-averaged raw stacked binary probs
fold_rows = []
importance_rows = []

# CONTRACT fold split: StratifiedKFold(5, shuffle=True, random_state=42) over the
# competition train rows only, integer labels in original train-CSV order.
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_cpu), dtype=np.int8), y_cpu), start=1):
    fold_seed = SEED + fold * 100
    emit(f'\n===== Fold {fold}/{N_SPLITS} | seed={fold_seed} =====')

    X_tr = X.iloc[tr_idx].reset_index(drop=True)
    y_tr = y.iloc[tr_idx].reset_index(drop=True)
    X_va = X.iloc[va_idx].reset_index(drop=True)
    y_va = y.iloc[va_idx].reset_index(drop=True)
    X_te = X_test.copy(deep=True)

    if USE_ORIGINAL_ROWS:
        raise ValueError('EXP3-110 should not append original rows. Keep USE_ORIGINAL_ROWS=False.')

    # ---- FE/TE computed ONCE per fold, shared by all 3 binary models (== xgbv5) ----
    X_tr, X_va, X_te, added_te = add_fold_safe_te(X_tr, y_tr, X_va, X_te, TE_COLS)
    X_tr_np, X_va_np, X_te_np, features = prepare_selected(X_tr, X_va, X_te)
    y_tr_np = y_tr.to_numpy().astype(np.int32)
    y_va_np = y_va.to_numpy().astype(np.int32)
    emit(f'training shape: {X_tr_np.shape} validation shape: {X_va_np.shape} test shape: {X_te_np.shape}')
    emit(f'TE features added before top selection: {len(added_te)}')

    # accumulate raw binary P(class=k) into 3-column arrays (column k = class k)
    va_probs = np.zeros((X_va_np.shape[0], len(CLASSES)), dtype=np.float32)
    te_probs = np.zeros((X_te_np.shape[0], len(CLASSES)), dtype=np.float32)

    for cls_idx, cls_name in INT_TO_CLASS.items():
        # binary target: class k vs rest
        y_tr_bin = (y_tr_np == cls_idx).astype(np.int32)
        y_va_bin = (y_va_np == cls_idx).astype(np.int32)
        n_pos = int(y_tr_bin.sum())
        n_neg = int(len(y_tr_bin) - n_pos)
        # scale_pos_weight = n_neg / n_pos -> up-weight the (rare) positive class so
        # the binary model is balanced for this 1-vs-rest split (helps recall of the
        # minority class, which balanced accuracy rewards). QSO is rare -> large spw.
        spw = float(n_neg) / float(max(n_pos, 1))

        binary_seed = fold_seed + cls_idx  # distinct seed per binary model
        params = make_xgb_binary_params(binary_seed)
        params['scale_pos_weight'] = spw

        emit(f'  [class {cls_name}] vs rest: pos={n_pos} neg={n_neg} scale_pos_weight={spw:.4f} seed={binary_seed}')

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr_np,
            y_tr_bin,
            eval_set=[(X_va_np, y_va_bin)],
            verbose=250,
        )

        # predict_proba returns (n, 2); column 1 is P(class==k)
        va_probs[:, cls_idx] = model.predict_proba(X_va_np)[:, 1].astype(np.float32)
        te_probs[:, cls_idx] = model.predict_proba(X_te_np)[:, 1].astype(np.float32)

        best_iter = getattr(model, 'best_iteration', None)
        emit(f'  [class {cls_name}] best_iteration={best_iter}')

        gain = model.get_booster().get_score(importance_type='gain')
        for i, f in enumerate(features):
            importance_rows.append({'fold': fold, 'class': cls_name, 'feature': f, 'gain': float(gain.get(f'f{i}', 0.0))})

        del model
        gc.collect()

    # ROW-NORMALIZE the 3 stacked binary probabilities into a valid 3-class dist.
    # Stored normalized so OOF/test arrays are directly usable (and argmax-stable).
    va_norm = normalize_clip(va_probs)
    te_norm = normalize_clip(te_probs)

    oof[va_idx] = va_norm
    oof_fold[va_idx] = fold
    test_pred_sum += te_norm / N_SPLITS

    fold_score = balanced_accuracy_score(y_cpu[va_idx], np.argmax(oof[va_idx], axis=1))
    emit(f'fold {fold} balanced accuracy: {fold_score:.8f}')
    fold_rows.append({
        'fold': fold,
        'balanced_accuracy': float(fold_score),
        'n_train': int(X_tr_np.shape[0]),
        'n_valid': int(X_va_np.shape[0]),
        'n_features': int(len(features)),
        'n_te_features': int(len(added_te)),
    })

    del X_tr, X_va, X_te, X_tr_np, X_va_np, X_te_np, y_tr, y_va, y_tr_np, y_va_np
    del va_probs, te_probs, va_norm, te_norm
    gc.collect()

    flush_results()  # checkpoint after each fold

fold_scores = pd.DataFrame(fold_rows)
feature_importance = pd.DataFrame(importance_rows)
cv_score = balanced_accuracy_score(y_cpu, np.argmax(oof, axis=1))
emit('\nFold scores:')
emit(fold_scores.to_string(index=False))
emit(f'Mean fold balanced accuracy: {fold_scores["balanced_accuracy"].mean():.8f}')
emit(f'OOF balanced accuracy: {cv_score:.8f}')

# Per-class recall (GALAXY/QSO/STAR) on the OOF predictions (argmax of normalized probs)
oof_pred_int = np.argmax(oof, axis=1)
recalls = recall_score(y_cpu, oof_pred_int, average=None, labels=[0, 1, 2])
emit('Per-class OOF recall:')
for idx, cls_name in INT_TO_CLASS.items():
    emit(f'  {cls_name}: {recalls[idx]:.8f}')


# ---------------------------------------------------------------------------
# Feature importance (top 25, mean gain across folds & binary models) -> results.txt
# ---------------------------------------------------------------------------
top_importance = (
    feature_importance.groupby('feature', as_index=False)['gain']
    .mean()
    .sort_values('gain', ascending=False)
    .reset_index(drop=True)
)
emit('\nTop 25 features by mean gain (across folds & OvR binary models):')
emit(top_importance.head(25).to_string(index=False))


# ---------------------------------------------------------------------------
# Save artifacts (already row-normalized per fold; re-normalize defensively)
# ---------------------------------------------------------------------------
oof_preds = normalize_clip(oof)
test_preds = normalize_clip(test_pred_sum)

assert oof_preds.shape == (len(X), len(CLASSES)), oof_preds.shape
assert test_preds.shape == (len(X_test), len(CLASSES)), test_preds.shape

np.save(OOF_PATH, oof_preds)
np.save(PRED_PATH, test_preds)
# legacy copies (same arrays) for any downstream that expects the notebook names
np.save(LEGACY_OOF_PATH, oof_preds)
np.save(LEGACY_PRED_PATH, test_preds)

pred_labels = [INT_TO_CLASS[i] for i in np.argmax(test_preds, axis=1)]
if sample is not None and ID_COL in sample.columns:
    submission = sample.copy()
    submission[TARGET] = pred_labels
else:
    submission = pd.DataFrame({ID_COL: test_ids.to_numpy(), TARGET: pred_labels})
submission.to_csv(SUB_PATH, index=False)

emit('\nSaved artifacts:')
emit(f'  {OOF_PATH} {oof_preds.shape}')
emit(f'  {PRED_PATH} {test_preds.shape}')
emit(f'  {SUB_PATH} {submission.shape}')
emit(str(submission['class'].value_counts().to_dict()))

flush_results()
print('DONE')

