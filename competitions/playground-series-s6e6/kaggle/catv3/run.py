"""Kaggle SCRIPT kernel: cdeotte CatBoost v3 (cat-3 / CAT2-035) for PS-S6E6.

Faithful port of cat-v3-for-s6e6.ipynb. Single 5-fold CatBoost (GPU) using a
large native-categorical feature family + the ORIGINAL SDSS17 dataset appended
at weight 0.06 (spectral_type / galaxy_population reconstructed from colors).
Local 5-fold CV ~0.96887, public LB ~0.96972.

Feature engineering is pure pandas + numpy (NO cuDF / cuPy). Kaggle assigns a
P100 (sm_60) and cuDF 26.x dropped Pascal support, so cudf.read_csv crashed with
`copy_if failed: cudaErrorInvalidDevice`. CatBoost GPU itself works on the P100
(the get_gpu_device_count guard detected the device), so ONLY the cuDF/cuPy FE
layer was rewritten; the produced features (names + semantics), the fold
contract, the original-data handling, and the CatBoost params are unchanged.

The pandas FE reproduces the cuDF version 1:1:
  * quantile bins: np.quantile(linear) edges on finite values + np.searchsorted(side='right')
  * floor/round/mod/frac cats: np.floor / np.rint (banker's rounding) on nan-filled arrays
  * 2-way / 3-way hashed combos: exact integer formulas in int64 numpy arrays
  * cudf.cut(..., right=True) reconstructions -> equivalent <= threshold np.select logic
All bins / edges / vocabularies are computed on the COMBINED train+test+original
frame (one concatenated DataFrame), so codes are consistent and label-free.

Adaptations from the source notebook (only these):
  * original SDSS path -> my mirror via recursive glob (no fedesoriano slug)
  * CatBoost GPU guard via get_gpu_device_count; single-GPU device id, CPU fallback
  * outputs -> /kaggle/working/oof_catv3.npy, test_catv3.npy, results.txt, submission.csv
  * removed plotting / IPython display
The fold contract (StratifiedKFold(5, shuffle=True, random_state=42) on competition
rows, integer labels GALAXY=0/QSO=1/STAR=2 in train-CSV order, original rows only in
each fold's TRAIN pool) is ALREADY exactly what the source notebook does -> kept verbatim.
"""

import os

# Keep CPU helper libraries from oversubscribing while CatBoost owns the GPU.
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')

import gc
import glob
import random
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import catboost as cb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

T0 = time.perf_counter()

RESULTS_PATH = '/kaggle/working/results.txt'
_results_fh = open(RESULTS_PATH, 'w')


def log(msg):
    line = f'[{time.perf_counter() - T0:8.1f}s] {msg}'
    print(line, flush=True)
    _results_fh.write(line + '\n')
    _results_fh.flush()


print('pandas:', pd.__version__)
print('numpy:', np.__version__)
print('CatBoost:', cb.__version__)

# -----------------------------------------------------------------------------
# Configuration (verbatim from source CAT2-035 settings)
# -----------------------------------------------------------------------------
MODEL_ID = 'catv3'
SEED = 42
N_SPLITS = 5
TARGET = 'class'
ID_COL = 'id'

CLASSES = ['GALAXY', 'QSO', 'STAR']
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}

RAW_NUM_COLS = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
BANDS = ['u', 'g', 'r', 'i', 'z']
BASE_CATS = ['spectral_type', 'galaxy_population']
EPS = np.float32(1e-6)

ORIGINAL_WEIGHT = 0.06
ITERATIONS = 5000
EARLY_STOPPING_ROUNDS = 260
PREDICT_BATCH_SIZE = 80_000

OOF_OUT = '/kaggle/working/oof_catv3.npy'
TEST_OUT = '/kaggle/working/test_catv3.npy'
SUB_OUT = '/kaggle/working/submission.csv'

random.seed(SEED)
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# CatBoost GPU guard (task_type='GPU' only errors at .fit(), so detect up front)
# -----------------------------------------------------------------------------
try:
    from catboost.utils import get_gpu_device_count
    N_GPU = int(get_gpu_device_count())
except Exception as e:
    print('get_gpu_device_count failed:', repr(e))
    N_GPU = 0
USE_GPU = N_GPU > 0
log(f'detected GPU device count = {N_GPU} -> task_type={"GPU" if USE_GPU else "CPU"}')

# -----------------------------------------------------------------------------
# Load Data (pure pandas)
# -----------------------------------------------------------------------------
def find_competition_root():
    candidates = [
        Path('/kaggle/input/competitions/playground-series-s6e6'),
        Path('/kaggle/input/playground-series-s6e6'),
    ]
    candidates += [Path(p).parent for p in glob.glob('/kaggle/input/**/train.csv', recursive=True)]
    seen = []
    for root in candidates:
        if root not in seen:
            seen.append(root)
    for root in seen:
        if (root / 'train.csv').exists() and (root / 'test.csv').exists():
            return root
    raise FileNotFoundError('Could not find train.csv and test.csv.')


def find_original_path():
    # Mirror dataset: cindyxue1122/s6e6-original-sdss17 ; recursive glob.
    candidates = [Path(p) for p in glob.glob('/kaggle/input/**/star_classification.csv', recursive=True)]
    seen = []
    for path in candidates:
        if path not in seen:
            seen.append(path)
    for path in seen:
        if path.exists():
            return path
    raise FileNotFoundError('Could not find star_classification.csv (original SDSS mirror).')


def clean_num(s):
    # cudf.to_numeric(errors='coerce').astype('float32') equivalent.
    return pd.to_numeric(s, errors='coerce').astype('float32')


def cat_key(s):
    # cudf: s.astype('str').fillna('__NA__'). (BASE_CATS aren't used as strings
    # downstream -- they're remapped via spec_map/pop_map -- but kept for parity.)
    return s.astype('str').fillna('__NA__')


def spectral_type_from_gr(g, r):
    # cudf.cut(r-g, [-inf,-1,-0.5,0,inf], right=True) -> [M, G/K, A/F, O/B].
    # right=True bins are (edge_i, edge_{i+1}] i.e. label by first <= upper edge.
    rg = (r - g).to_numpy()
    out = np.full(rg.shape, 'O/B', dtype=object)
    out[rg <= 0.0] = 'A/F'
    out[rg <= -0.5] = 'G/K'
    out[rg <= -1.0] = 'M'
    # NaN -> not <= any edge -> stays 'O/B' in numpy comparisons? np.nan <= x is
    # False, so NaN falls through to the default 'O/B', matching cudf.cut which
    # returns NaN for out-of-range/NaN -> astype(str) -> 'nan'. To stay faithful
    # to cudf (NaN label), mark NaN explicitly.
    nan_mask = ~np.isfinite(rg)
    out[nan_mask] = 'nan'
    return pd.Series(out, index=g.index)


def galaxy_population_from_ur(u, r):
    # cudf.cut(u-r, [-inf, 2.2, inf], right=True) -> [Blue_Cloud, Red_Sequence].
    ur = (u - r).to_numpy()
    out = np.full(ur.shape, 'Red_Sequence', dtype=object)
    out[ur <= 2.2] = 'Blue_Cloud'
    nan_mask = ~np.isfinite(ur)
    out[nan_mask] = 'nan'
    return pd.Series(out, index=u.index)


def read_competition_csv(path, is_train):
    df = pd.read_csv(str(path))
    for c in RAW_NUM_COLS:
        df[c] = clean_num(df[c])
    for c in BASE_CATS:
        if c in df.columns:
            df[c] = cat_key(df[c])
        else:
            df[c] = '__NA__'
    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype('int32')
    if is_train:
        df[TARGET] = df[TARGET].astype('str')
    return df


def read_original_csv(path):
    orig = pd.read_csv(str(path))
    keep = pd.DataFrame()
    keep[ID_COL] = (-1 - np.arange(len(orig), dtype=np.int32))
    for c in RAW_NUM_COLS:
        keep[c] = clean_num(orig[c])
    keep['spectral_type'] = spectral_type_from_gr(keep['g'], keep['r'])
    keep['galaxy_population'] = galaxy_population_from_ur(keep['u'], keep['r'])
    keep[TARGET] = orig[TARGET].astype('str').str.upper()
    keep = keep[keep[TARGET].isin(CLASSES)].reset_index(drop=True)
    # Drop sentinel band magnitudes (e.g. -9999) before feature engineering.
    band_mask = None
    for b in BANDS:
        m = (keep[b] > -100) & (keep[b] < 100)
        band_mask = m if band_mask is None else (band_mask & m)
    keep = keep[band_mask].reset_index(drop=True)
    for c in BASE_CATS:
        keep[c] = cat_key(keep[c])
    return keep


DATA_ROOT = find_competition_root()
ORIG_PATH = find_original_path()

train = read_competition_csv(DATA_ROOT / 'train.csv', is_train=True)
test = read_competition_csv(DATA_ROOT / 'test.csv', is_train=False)
original = read_original_csv(ORIG_PATH)

sample_path = DATA_ROOT / 'sample_submission.csv'
sample = pd.read_csv(sample_path) if sample_path.exists() else None

y = train[TARGET].map(CLASS_TO_INT).astype('int8').to_numpy()
y_original = original[TARGET].map(CLASS_TO_INT).astype('int8').to_numpy()
test_ids = test[ID_COL].to_numpy() if ID_COL in test.columns else np.arange(len(test), dtype=np.int32)

log(f'competition root: {DATA_ROOT}')
log(f'original dataset: {ORIG_PATH}')
log(f'train/test/original: {train.shape} {test.shape} {original.shape}')
ratios = pd.Series(y).map(INT_TO_CLASS).value_counts(normalize=True).sort_index()
log('target distribution: ' + ratios.to_dict().__str__())

# -----------------------------------------------------------------------------
# Feature Engineering (pure pandas / numpy -- faithful port of build_features_gpu)
#
# Helpers mirror the cuDF/cuPy semantics exactly:
#   * to_np: column -> float32 numpy with NaN for nulls (== to_cupy na_value=nan)
#   * finite_float: nan/inf -> 0.0  (== cuPy nan_to_num)
#   * safe_div: a / (b + EPS), then nan/inf -> 0.0
#   * cat_from_arr: float -> int via nan_to_num(-2147483648/2147483647) then int32
#     (matches cat_from_cp); integer arrays pass straight through to int32
#   * qbin_cat: np.quantile(linear) edges on finite values, np.searchsorted(right)
#   * floor_cat / round_cat: np.floor / np.rint after nan-fill with -999999.0
#   * hash2 / hash3: exact integer hash formulas in int64
# -----------------------------------------------------------------------------
INT32_MIN = -2147483648.0
INT32_MAX = 2147483647.0


def to_np(s):
    return s.to_numpy(dtype=np.float32, na_value=np.nan)


def finite_float(values):
    arr = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def safe_div(a, b):
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return finite_float(aa / (bb + np.float32(EPS)))


def cat_from_arr(values):
    arr = np.asarray(values)
    if arr.dtype.kind == 'f':
        arr = np.nan_to_num(arr, nan=INT32_MIN, posinf=INT32_MAX, neginf=INT32_MIN)
    return arr.astype(np.int32)


def qbin_codes(arr, bins):
    arr = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(arr)
    if int(valid.sum()) <= 1:
        return np.full(arr.shape, -1, dtype=np.int32)
    qs = np.linspace(0, 1, bins + 1, dtype=np.float32)[1:-1]
    # np.quantile default interpolation 'linear' matches cupy.quantile default.
    edges = np.quantile(arr[valid], qs)
    edges = np.unique(edges[np.isfinite(edges)])
    if len(edges) == 0:
        codes = np.zeros(arr.shape, dtype=np.int32)
    else:
        codes = np.searchsorted(edges, arr, side='right').astype(np.int32)
    codes = np.where(valid, codes, np.int32(-1))
    return codes.astype(np.int32)


def floor_codes(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return cat_from_arr(np.floor(np.nan_to_num(arr, nan=-999999.0)))


def round_codes(arr, decimals):
    arr = np.asarray(arr, dtype=np.float32)
    scale = np.float32(10 ** decimals)
    return cat_from_arr(np.rint(np.nan_to_num(arr, nan=-999999.0) * scale))


def hash2_codes(a, b):
    # a, b are int32 code arrays; nulls already encoded as -1 (no NaNs present).
    aa = np.asarray(a, dtype=np.int64)
    bb = np.asarray(b, dtype=np.int64)
    vals = ((aa + 1_000_003) * 1_000_003 + (bb + 9_176)) % 2_147_483_647
    return cat_from_arr(vals)


def hash3_codes(a, b, c):
    aa = np.asarray(a, dtype=np.int64)
    bb = np.asarray(b, dtype=np.int64)
    cc = np.asarray(c, dtype=np.int64)
    vals = (((aa + 1_000_003) * 1_000_003 + (bb + 9_176)) * 1_000_003 + (cc + 17_191)) % 2_147_483_647
    return cat_from_arr(vals)


def build_features(train_df, test_df, original_df):
    n_train = len(train_df)
    n_test = len(test_df)

    frames = [train_df.drop(columns=[TARGET]), test_df, original_df.drop(columns=[TARGET])]
    df = pd.concat(frames, axis=0, ignore_index=True)
    del frames

    for c in RAW_NUM_COLS:
        df[c] = df[c].astype('float32')

    # float feature columns we build; gathered then assigned in bulk at the end
    # (avoids fragmentation warnings + repeated reallocation on a wide frame).
    feat = {}
    cat = {}
    cat_cols = []

    spec_map = {'M': 0, 'G/K': 1, 'A/F': 2, 'O/B': 3}
    pop_map = {'Blue_Cloud': 0, 'Red_Sequence': 1}
    spectral_type = df['spectral_type'].astype('str').map(spec_map).fillna(-1).astype('int32').to_numpy()
    galaxy_population = df['galaxy_population'].astype('str').map(pop_map).fillna(-1).astype('int32').to_numpy()
    cat['spectral_type'] = spectral_type
    cat['galaxy_population'] = galaxy_population
    cat_cols += ['spectral_type', 'galaxy_population']

    # raw band columns as numpy float32 for vector math
    np_raw = {c: df[c].to_numpy(dtype=np.float32) for c in RAW_NUM_COLS}

    # Public-combo feature family: all optical color differences + abs differences.
    color = {}  # store color diffs by 'a_b' name for later reuse
    for a, b in combinations(BANDS, 2):
        diff = (np_raw[a] - np_raw[b]).astype(np.float32)
        feat[f'{a}_{b}'] = diff
        feat[f'{a}_{b}_abs'] = np.abs(diff).astype(np.float32)
        color[f'{a}_{b}'] = diff

    mags = np.stack([np_raw[b] for b in BANDS], axis=1)
    feat['mag_mean'] = np.nanmean(mags, axis=1).astype(np.float32)
    feat['mag_std'] = np.nanstd(mags, axis=1).astype(np.float32)
    feat['mag_min'] = np.nanmin(mags, axis=1).astype(np.float32)
    feat['mag_max'] = np.nanmax(mags, axis=1).astype(np.float32)
    mag_range = (feat['mag_max'] - feat['mag_min']).astype(np.float32)
    feat['mag_range'] = mag_range
    cat['mag_argmin'] = cat_from_arr(np.nanargmin(mags, axis=1))
    cat['mag_argmax'] = cat_from_arr(np.nanargmax(mags, axis=1))
    cat_cols += ['mag_argmin', 'mag_argmax']

    x = np.arange(len(BANDS), dtype=np.float32)
    x_center = x - x.mean()
    centered = mags - np.nanmean(mags, axis=1)[:, None]
    feat['mag_slope'] = (centered.dot(x_center) / np.sum(x_center ** 2)).astype(np.float32)
    feat['mag_curvature'] = (np_raw['u'] - 2 * np_raw['r'] + np_raw['z']).astype(np.float32)
    feat['blue_curvature'] = (np_raw['u'] - 2 * np_raw['g'] + np_raw['r']).astype(np.float32)
    feat['red_curvature'] = (np_raw['r'] - 2 * np_raw['i'] + np_raw['z']).astype(np.float32)

    redshift = np_raw['redshift']
    feat['redshift_abs'] = np.abs(redshift).astype(np.float32)
    feat['redshift_log1p_abs'] = np.log1p(np.abs(redshift)).astype(np.float32)
    feat['redshift_sq'] = (redshift ** 2).astype(np.float32)
    feat['redshift_cbrt'] = np.cbrt(redshift).astype(np.float32)
    cat['redshift_is_neg'] = cat_from_arr(redshift < 0)
    cat['redshift_lt_002'] = cat_from_arr(redshift < 0.02)
    cat['redshift_gt_07'] = cat_from_arr(redshift > 0.7)
    cat_cols += ['redshift_is_neg', 'redshift_lt_002', 'redshift_gt_07']

    feat['g_over_redshift'] = safe_div(np_raw['g'], np.abs(redshift))
    feat['i_over_redshift'] = safe_div(np_raw['i'], np.abs(redshift))
    feat['z_over_redshift'] = safe_div(np_raw['z'], np.abs(redshift))
    feat['z_over_g'] = safe_div(np_raw['z'], np_raw['g'])
    feat['z2_over_g2'] = safe_div(np_raw['z'] ** 2, np_raw['g'] ** 2)
    feat['log_z_over_log_g'] = safe_div(np.log1p(np.abs(np_raw['z'])), np.log1p(np.abs(np_raw['g'])))
    feat['sqrt_z_over_sqrt_g'] = safe_div(np.sqrt(np.abs(np_raw['z'])), np.sqrt(np.abs(np_raw['g'])))
    for b in BANDS:
        feat[f'redshift_x_{b}'] = (redshift * np_raw[b]).astype(np.float32)

    feat['ug_gr_ratio'] = safe_div(color['u_g'], color['g_r'])
    feat['gr_ri_ratio'] = safe_div(color['g_r'], color['r_i'])
    feat['ri_iz_ratio'] = safe_div(color['r_i'], color['i_z'])
    color_mat = np.stack([color[c] for c in ['u_g', 'g_r', 'r_i', 'i_z']], axis=1)
    feat['color_mean'] = np.nanmean(color_mat, axis=1).astype(np.float32)
    feat['color_std'] = np.nanstd(color_mat, axis=1).astype(np.float32)
    feat['color_abs_sum'] = np.nansum(np.abs(color_mat), axis=1).astype(np.float32)

    u_r = (np_raw['u'] - np_raw['r']).astype(np.float32)
    r_z = (np_raw['r'] - np_raw['z']).astype(np.float32)
    color['u_r'] = u_r
    color['r_z'] = r_z

    gr = np.clip(color['g_r'], -5, 5)
    ur = np.clip(u_r, -5, 8)
    feat['color_temp_gr_proxy'] = finite_float(4600.0 * ((1.0 / (0.92 * gr + 1.7)) + (1.0 / (0.92 * gr + 0.62))))
    feat['uv_excess_proxy'] = (color['u_g'] - (0.75 * color['g_r'] + 0.18)).astype(np.float32)
    feat['red_sequence_score_proxy'] = (ur - 2.2).astype(np.float32)

    alpha_rad = (np_raw['alpha'] * np.float32(np.pi / 180.0)).astype(np.float32)
    delta_rad = (np_raw['delta'] * np.float32(np.pi / 180.0)).astype(np.float32)
    feat['alpha_sin'] = np.sin(alpha_rad).astype(np.float32)
    feat['alpha_cos'] = np.cos(alpha_rad).astype(np.float32)
    feat['delta_sin'] = np.sin(delta_rad).astype(np.float32)
    feat['delta_cos'] = np.cos(delta_rad).astype(np.float32)
    feat['sky_x'] = (np.cos(delta_rad) * np.cos(alpha_rad)).astype(np.float32)
    feat['sky_y'] = (np.cos(delta_rad) * np.sin(alpha_rad)).astype(np.float32)
    feat['sky_z'] = np.sin(delta_rad).astype(np.float32)

    fluxes = []
    for b in BANDS:
        clipped = np.clip(np_raw[b], -30, 30)
        flux = np.power(np.float32(10.0), np.float32(-0.4) * clipped).astype(np.float32)
        feat[f'flux_{b}'] = flux
        feat[f'log_flux_{b}'] = np.log1p(flux).astype(np.float32)
        fluxes.append(flux)
    fv = np.stack(fluxes, axis=1)
    feat['flux_mean'] = np.nanmean(fv, axis=1).astype(np.float32)
    feat['flux_std'] = np.nanstd(fv, axis=1).astype(np.float32)
    feat['flux_range'] = (np.nanmax(fv, axis=1) - np.nanmin(fv, axis=1)).astype(np.float32)
    for a, b in [('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'z'), ('u', 'r'), ('r', 'z')]:
        col = np.clip(np_raw[a] - np_raw[b], -20, 20)
        feat[f'flux_ratio_{a}_{b}'] = np.exp(np.float32(-0.921034) * col).astype(np.float32)

    rg_raw = (np_raw['r'] - np_raw['g'])
    spec_calc = np.where(rg_raw <= -1.0, 0, np.where(rg_raw <= -0.5, 1, np.where(rg_raw <= 0.0, 2, 3)))
    pop_calc = np.where((np_raw['u'] - np_raw['r']) <= 2.2, 0, 1)
    cat['spectral_type_calc'] = cat_from_arr(spec_calc)
    cat['galaxy_population_calc'] = cat_from_arr(pop_calc)
    cat['spectral_x_pop'] = cat_from_arr(spectral_type.astype(np.int32) * 10 + galaxy_population.astype(np.int32))
    cat['spectral_calc_x_pop_calc'] = cat_from_arr(spec_calc * 10 + pop_calc)
    cat_cols += ['spectral_type_calc', 'galaxy_population_calc', 'spectral_x_pop', 'spectral_calc_x_pop_calc']

    # floor cats on raw columns
    for c in RAW_NUM_COLS:
        name = f'{c}_floor_cat'
        cat[name] = floor_codes(np_raw[c])
        cat_cols.append(name)

    # quantile-bin cats. Build a lookup for source arrays by name (raw + colors +
    # mag_range/mag_std), matching the cuDF source which read gdf[source].
    qsrc = dict(np_raw)
    qsrc['u_g'] = color['u_g']
    qsrc['g_r'] = color['g_r']
    qsrc['r_i'] = color['r_i']
    qsrc['i_z'] = color['i_z']
    qsrc['u_r'] = u_r
    qsrc['r_z'] = r_z
    qsrc['mag_range'] = mag_range
    qsrc['mag_std'] = feat['mag_std']

    q_specs = {c: [32, 100, 500] for c in RAW_NUM_COLS}
    for c in ['u_g', 'g_r', 'r_i', 'i_z', 'u_r', 'r_z', 'mag_range', 'mag_std']:
        q_specs.setdefault(c, [64])
    for c, bins_list in q_specs.items():
        for bins in bins_list:
            name = f'{c}_q{bins}_cat'
            cat[name] = qbin_codes(qsrc[c], bins)
            cat_cols.append(name)

    # round cats. Source arrays: raw cols + colors + mag_range.
    rsrc = dict(np_raw)
    rsrc['u_g'] = color['u_g']
    rsrc['g_r'] = color['g_r']
    rsrc['r_i'] = color['r_i']
    rsrc['i_z'] = color['i_z']
    rsrc['u_r'] = u_r
    rsrc['r_z'] = r_z
    rsrc['mag_range'] = mag_range
    round_specs = {
        'alpha': 1, 'delta': 1, 'u': 2, 'g': 2, 'r': 2, 'i': 2, 'z': 2,
        'redshift': 4, 'u_g': 3, 'g_r': 3, 'r_i': 3, 'i_z': 3,
        'u_r': 3, 'r_z': 3, 'mag_range': 3,
    }
    for c, dec in round_specs.items():
        name = f'{c}_round{dec}_cat'
        cat[name] = round_codes(rsrc[c], dec)
        cat_cols.append(name)

    # mod / frac / decimal cats on raw columns (absolute value of raw).
    for c in RAW_NUM_COLS:
        vals = np.abs(np_raw[c]).astype(np.float32)
        ints = np.floor(np.nan_to_num(vals, nan=0.0)).astype(np.int64)
        frac = np.floor((vals - np.floor(vals)) * 20).astype(np.int32)
        deci = np.floor((vals - np.floor(vals)) * 1000).astype(np.int32)
        for suffix, arr in [('mod10', ints % 10), ('mod100', ints % 100), ('frac20', frac), ('decimal1000', deci)]:
            name = f'{c}_{suffix}_cat'
            cat[name] = cat_from_arr(arr)
            cat_cols.append(name)

    # 2-way manual hashed combos.
    manual_combos = [
        ('alpha_floor_cat', 'delta_floor_cat'),
        ('u_floor_cat', 'z_floor_cat'),
        ('spectral_type', 'galaxy_population'),
        ('spectral_x_pop', 'redshift_q64_cat'),
        ('u_r_q64_cat', 'redshift_q64_cat'),
        ('g_r_q64_cat', 'mag_range_q64_cat'),
        ('alpha_q100_cat', 'delta_q100_cat'),
        ('u_q100_cat', 'z_q100_cat'),
        ('spectral_x_pop', 'redshift_q100_cat'),
    ]
    for a, b in manual_combos:
        if a in cat and b in cat:
            name = f'COMBO_{a}__{b}'
            cat[name] = hash2_codes(cat[a], cat[b])
            cat_cols.append(name)

    # 2-way PAIR + 3-way TRIO hashed combos over the first 10 base cats.
    combo_bases = [
        'spectral_type', 'galaxy_population', 'spectral_x_pop',
        'alpha_floor_cat', 'delta_floor_cat', 'u_floor_cat', 'z_floor_cat',
        'alpha_q100_cat', 'delta_q100_cat', 'u_q100_cat', 'z_q100_cat',
        'redshift_q64_cat', 'u_r_q64_cat', 'g_r_q64_cat', 'mag_range_q64_cat',
    ]
    combo_bases = [c for c in combo_bases if c in cat]
    bases = combo_bases[:10]
    for a, b in combinations(bases, 2):
        name = f'PAIR_{a}__{b}'
        cat[name] = hash2_codes(cat[a], cat[b])
        cat_cols.append(name)
    for trio in [bases[:3], bases[3:6], bases[6:9]]:
        if len(trio) == 3:
            name = 'TRIO_' + '__'.join(trio)
            cat[name] = hash3_codes(cat[trio[0]], cat[trio[1]], cat[trio[2]])
            cat_cols.append(name)

    # Assemble the final frame. Categorical cols: int32 (fillna -1 already baked
    # into the codes). Numeric cols: float32. Preserve column order: raw numeric
    # cols (already in df), then feat (float), then cat -- but the source produced
    # column order = [raw cols, color cols + abs, mag/redshift/... feats, cat cols]
    # interleaved as built. To preserve the EXACT source order we rebuild df by
    # the construction order: start from raw numeric cols, then walk an ordered
    # build log. Simpler + equivalent for CatBoost (order-independent given
    # cat_features by NAME): drop ID, keep raw cols, add feat then cat.
    out = df[RAW_NUM_COLS].copy()
    out = out.astype('float32')
    # float features
    for name, arr in feat.items():
        out[name] = np.asarray(arr, dtype=np.float32)
    del feat
    gc.collect()
    # categorical features (int32, codes already include -1 for missing)
    cat_cols = [c for c in dict.fromkeys(cat_cols) if c in cat]
    for name in cat_cols:
        arr = cat[name]
        # mirror cudf: fillna(-1).astype('int32'); codes are already int with no
        # NaN (NaNs were encoded to INT32_MIN/sentinel by cat_from_arr), so the
        # fillna is a no-op here, but enforce int32 dtype.
        out[name] = np.asarray(arr, dtype=np.int32)
    del cat
    gc.collect()

    X = out.iloc[:n_train].reset_index(drop=True)
    X_test = out.iloc[n_train:n_train + n_test].reset_index(drop=True)
    X_original = out.iloc[n_train + n_test:].reset_index(drop=True)

    del out, df, mags, color_mat, fluxes, fv, color, np_raw, qsrc, rsrc
    gc.collect()
    return X, X_test, X_original, cat_cols


log('building features (pandas/numpy)')
X, X_test, X_original, CAT_COLS = build_features(train, test, original)
FEATURES = list(X.columns)

log(f'features built: X={X.shape}, X_test={X_test.shape}, X_original={X_original.shape}, categorical={len(CAT_COLS)}')
print('first features:', FEATURES[:20])
print('first categorical features:', CAT_COLS[:20])

del train, test, original
gc.collect()

# CatBoost needs the categorical columns as int/str, not float. Our cat cols are
# already int32 (built via cat_from_arr / .astype('int32')), so passing
# cat_features=CAT_COLS (by name) on the pandas DataFrame is valid.

# -----------------------------------------------------------------------------
# Train CatBoost (params verbatim; only task_type / devices adapted by GPU guard)
# -----------------------------------------------------------------------------
def make_cat_params(seed):
    params = {
        'loss_function': 'MultiClass',
        'eval_metric': 'TotalF1:average=Macro',
        'iterations': ITERATIONS,
        'depth': 8,
        'learning_rate': 0.042,
        'l2_leaf_reg': 8.0,
        'random_strength': 1.2,
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 0.2,
        'one_hot_max_size': 16,
        'max_ctr_complexity': 3,
        'class_weights': [1.0, 3.25, 5.0],
        'border_count': 254,
        'random_seed': seed,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
        'thread_count': 4,
        'allow_writing_files': False,
        'verbose': 250,
    }
    if USE_GPU:
        params.update({
            'task_type': 'GPU',
            # Kaggle competition GPU is a single P100 (device 0). Source used a
            # 2x T4 layout ('0:1'); use a single device id here, NEVER '0:1'.
            'devices': '0' if N_GPU == 1 else f'0-{N_GPU - 1}',
            'gpu_ram_part': 0.85,
            'gpu_cat_features_storage': 'CpuPinnedMemory',
        })
    else:
        params['task_type'] = 'CPU'
    return params


def predict_proba_batched(model, X_data, cat_cols, batch_size=PREDICT_BATCH_SIZE):
    parts = []
    for start in range(0, len(X_data), batch_size):
        end = min(start + batch_size, len(X_data))
        pool = Pool(X_data.iloc[start:end], cat_features=cat_cols)
        parts.append(model.predict_proba(pool).astype('float32'))
        del pool
        gc.collect()
    return np.vstack(parts).astype('float32')


def make_train_pool(X_fold, y_fold, X_orig, y_orig):
    X_fit = pd.concat([X_fold, X_orig], axis=0, ignore_index=True)
    y_fit = np.concatenate([y_fold, y_orig]).astype('int8')
    weights = np.ones(len(y_fit), dtype='float32')
    weights[len(y_fold):] = np.float32(ORIGINAL_WEIGHT)
    pool = Pool(X_fit, y_fit, cat_features=CAT_COLS, weight=weights)
    return pool, X_fit, y_fit, weights


y_cpu = y.astype('int8')
oof = np.zeros((len(X), len(CLASSES)), dtype='float32')
oof_fold = np.full(len(X), -1, dtype='int8')
test_pred_sum = np.zeros((len(X_test), len(CLASSES)), dtype='float32')
fold_rows = []

# CONTRACT split: StratifiedKFold(5, shuffle=True, random_state=42) on competition
# rows, integer labels in train-CSV order. (Source already does exactly this.)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_cpu), dtype=np.int8), y_cpu), start=1):
    fold_seed = SEED + fold
    log(f'===== Fold {fold}/{N_SPLITS} | seed={fold_seed} =====')

    X_tr = X.iloc[tr_idx]
    X_va = X.iloc[va_idx]
    y_tr = y_cpu[tr_idx]
    y_va = y_cpu[va_idx]

    train_pool, X_fit, y_fit, weights = make_train_pool(X_tr, y_tr, X_original, y_original)
    valid_pool = Pool(X_va, y_va, cat_features=CAT_COLS)

    model = CatBoostClassifier(**make_cat_params(fold_seed))
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    va_probs = model.predict_proba(valid_pool).astype('float32')
    te_probs = predict_proba_batched(model, X_test, CAT_COLS)

    oof[va_idx] = va_probs
    oof_fold[va_idx] = fold
    test_pred_sum += te_probs / N_SPLITS

    fold_score = balanced_accuracy_score(y_va, np.argmax(va_probs, axis=1))
    best_iter = model.get_best_iteration()
    fold_rows.append({
        'fold': fold,
        'balanced_accuracy': float(fold_score),
        'best_iteration': int(best_iter) if best_iter is not None else None,
        'n_train': int(len(y_fit)),
        'n_valid': int(len(va_idx)),
    })
    log(f'fold {fold} balanced accuracy: {fold_score:.8f} | best_iteration={best_iter}')

    del model, train_pool, valid_pool, X_fit, y_fit, weights, X_tr, X_va, y_tr, y_va, va_probs, te_probs
    gc.collect()

# -----------------------------------------------------------------------------
# Results: OOF balanced accuracy + per-class recall
# -----------------------------------------------------------------------------
oof_pred = np.argmax(oof, axis=1)
cv_score = balanced_accuracy_score(y_cpu, oof_pred)
cm = confusion_matrix(y_cpu, oof_pred, labels=np.arange(len(CLASSES)))
per_class_recall = recall_score(y_cpu, oof_pred, labels=np.arange(len(CLASSES)), average=None)

log('')
log('Fold scores:')
for row in fold_rows:
    log(f"  fold {row['fold']}: balanced_accuracy={row['balanced_accuracy']:.8f} "
        f"best_iter={row['best_iteration']} n_train={row['n_train']} n_valid={row['n_valid']}")
mean_fold = float(np.mean([r['balanced_accuracy'] for r in fold_rows]))
log(f'Mean fold balanced accuracy: {mean_fold:.8f}')
log(f'OOF balanced accuracy: {cv_score:.8f}')
log('Per-class recall:')
for ci, cname in enumerate(CLASSES):
    log(f'  {cname}: {per_class_recall[ci]:.8f}')
log('Confusion matrix (rows=true, cols=pred) order [GALAXY,QSO,STAR]:')
for ci, cname in enumerate(CLASSES):
    log(f'  {cname}: {cm[ci].tolist()}')

# -----------------------------------------------------------------------------
# Save artifacts: row-normalize + clip away from 0, in train-CSV / sample order
# -----------------------------------------------------------------------------
def normalize_probs(p):
    p = np.asarray(p, dtype='float32')
    p = np.clip(p, 1e-7, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p.astype('float32')


oof_preds = normalize_probs(oof)
test_preds = normalize_probs(test_pred_sum)

np.save(OOF_OUT, oof_preds)
np.save(TEST_OUT, test_preds)

pred_labels = [INT_TO_CLASS[i] for i in np.argmax(test_preds, axis=1)]
if sample is not None and ID_COL in sample.columns:
    submission = sample.copy()
    submission[TARGET] = pred_labels
else:
    submission = pd.DataFrame({ID_COL: test_ids, TARGET: pred_labels})
submission.to_csv(SUB_OUT, index=False)

log(f'Saved: {OOF_OUT} {oof_preds.shape}')
log(f'Saved: {TEST_OUT} {test_preds.shape}')
log(f'Saved: {SUB_OUT} {submission.shape}')
log('DONE')

_results_fh.close()
