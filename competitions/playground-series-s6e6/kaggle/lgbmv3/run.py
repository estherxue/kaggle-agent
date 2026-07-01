"""s6e6 lgbmv3: faithful Kaggle-script port of cdeotte's "LGBM-3 for S6E6" notebook
(local worker model `lgbm-c03`).

A single level-1 LightGBM multiclass model. It does NOT stack, blend, or post-process.
Features = June competition features + astronomy color features + denoised categorical
bins + train/test count features + original-dataset class priors + class-prototype
distance features built from star_classification.csv (SDSS17 original dataset).

This is a STATIC port — it is pushed to Kaggle to train; it never runs locally.

What was changed vs. the source notebook (everything else is verbatim):
  * Data-path discovery: try competition mount paths + recursive glob fallback.
  * Original-dataset path: my mirror dataset (recursive glob for star_classification.csv);
    the third-party fedesoriano slug is NOT used (it caused push errors).
  * Original-dataset sentinel filter: SDSS17 has rows with -9999 band magnitudes that
    would poison the color-based priors/prototypes. Drop rows outside a sane (-100,100)
    band window BEFORE feature engineering. (Required deviation for using the mirror.)
  * Fold split forced to the project contract:
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split over the
        577347 competition rows only, integer labels in original train-CSV order.
    (The source already used exactly this split, so this is a no-op confirmation.)
  * Output paths/names: oof_lgbmv3.npy / test_lgbmv3.npy / submission.csv under
    /kaggle/working, plus results.txt. Removed all plotting / IPython display.
  * Kept all seeds, all hyper-parameters, all feature engineering verbatim.

Contract (so the new OOF stacks with existing artifacts):
  * Labels: GALAXY=0, QSO=1, STAR=2 (alphabetical).
  * OOF rows in train-CSV order; test rows in sample_submission order.
  * Original rows are used ONLY for prior/prototype features (never as validation rows);
    OOF is computed only on competition train rows. This matches the source design.

Outputs to /kaggle/working/:
  oof_lgbmv3.npy   (577347, 3) float32, columns [GALAXY,QSO,STAR], train-CSV row order
  test_lgbmv3.npy  (247435, 3) float32, columns [GALAXY,QSO,STAR], sample_submission order
  submission.csv   argmax -> label (nice-to-have)
  results.txt      per-fold BA + overall OOF BA + per-class recall (GALAXY/QSO/STAR)
"""

from __future__ import annotations

import os

# The source worker used 16 CPU threads for this LightGBM run. Override BLAS helper
# threads down so the env defaults don't oversubscribe; LightGBM gets num_threads itself.
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(key, '4')

import gc
import glob
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)

T0 = time.perf_counter()
_RESULT_LINES: list[str] = []


def log(msg):
    print(f'[{time.perf_counter() - T0:8.1f}s] {msg}', flush=True)


def emit(msg):
    """Print AND record for results.txt."""
    print(msg, flush=True)
    _RESULT_LINES.append(str(msg))


# ----------------------------------------------------------------------------
# Config  (verbatim from source, except output paths)
# ----------------------------------------------------------------------------
MODEL_ID = 'lgbmv3'
SOURCE_MODEL = 'lgbm-c03'
SEED = 42
N_FOLDS = 5
N_CLASSES = 3
TARGET = 'class'
ID_COL = 'id'

CLASSES = ['GALAXY', 'QSO', 'STAR']
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
INV_MAP = {i: c for c, i in LABEL_MAP.items()}

BASE_NUMS = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
BANDS = ['u', 'g', 'r', 'i', 'z']
RAW_CATS = ['spectral_type', 'galaxy_population']
EPS = 1e-6
NUM_THREADS = 16

N_TRAIN_EXPECT = 577347
N_TEST_EXPECT = 247435

WORK = Path('/kaggle/working')
WORK.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = WORK / 'results.txt'
OOF_PATH = WORK / f'oof_{MODEL_ID}.npy'
PRED_PATH = WORK / f'test_{MODEL_ID}.npy'
SUB_PATH = WORK / 'submission.csv'

CLIP = 1e-15


def flush_results():
    RESULTS_PATH.write_text('\n'.join(_RESULT_LINES) + '\n')


# ----------------------------------------------------------------------------
# Data discovery  (changed: competition mounts + recursive glob; my mirror for orig)
# ----------------------------------------------------------------------------
def find_data_dir():
    candidates = [
        Path('/kaggle/input/competitions/playground-series-s6e6'),
        Path('/kaggle/input/playground-series-s6e6'),
    ]
    candidates += [Path(p).parent for p in glob.glob('/kaggle/input/**/train.csv', recursive=True)]
    seen = []
    for c in candidates:
        if c not in seen:
            seen.append(c)
    for p in seen:
        if (p / 'train.csv').exists() and (p / 'test.csv').exists() and (p / 'sample_submission.csv').exists():
            return p
    raise FileNotFoundError('Could not find train.csv, test.csv, and sample_submission.csv')


def find_original_file():
    # My mirror dataset cindyxue1122/s6e6-original-sdss17; recursive glob finds
    # star_classification.csv wherever Kaggle mounts it (nested or flat).
    candidates = [Path(p) for p in glob.glob('/kaggle/input/**/star_classification.csv', recursive=True)]
    seen = []
    for c in candidates:
        if c not in seen:
            seen.append(c)
    for p in seen:
        if p.exists():
            return p
    raise FileNotFoundError('This notebook requires star_classification.csv for original-dataset priors/prototypes')


# ----------------------------------------------------------------------------
# Feature engineering  (VERBATIM from source notebook)
# ----------------------------------------------------------------------------
def rebuild_original_cats(df):
    out = df.copy()
    out['spectral_type'] = pd.cut(
        out['r'] - out['g'],
        [-np.inf, -1.0, -0.5, 0.0, np.inf],
        labels=['M', 'G/K', 'A/F', 'O/B'],
    ).astype(str)
    out['galaxy_population'] = pd.cut(
        out['u'] - out['r'],
        [-np.inf, 2.2, np.inf],
        labels=['Blue_Cloud', 'Red_Sequence'],
    ).astype(str)
    return out


def add_science_features(df):
    out = pd.DataFrame(index=df.index)
    for c in BASE_NUMS:
        out[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')

    band = out[BANDS].to_numpy(dtype='float32')
    color_pairs = [
        ('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'z'),
        ('u', 'r'), ('u', 'i'), ('u', 'z'), ('g', 'i'), ('g', 'z'), ('r', 'z'),
    ]
    for a, b in color_pairs:
        vals = (out[a] - out[b]).astype('float32')
        out[f'{a}_{b}'] = vals
        out[f'{a}_{b}_abs'] = np.abs(vals).astype('float32')

    out['mag_mean'] = np.nanmean(band, axis=1).astype('float32')
    out['mag_std'] = np.nanstd(band, axis=1).astype('float32')
    out['mag_min'] = np.nanmin(band, axis=1).astype('float32')
    out['mag_max'] = np.nanmax(band, axis=1).astype('float32')
    out['mag_range'] = (out['mag_max'] - out['mag_min']).astype('float32')
    out['mag_argmin'] = np.nanargmin(band, axis=1).astype('float32')
    out['mag_argmax'] = np.nanargmax(band, axis=1).astype('float32')
    out['blue_slope'] = ((out['g'] - out['u']) + (out['r'] - out['g'])).astype('float32')
    out['red_slope'] = ((out['i'] - out['r']) + (out['z'] - out['i'])).astype('float32')
    out['curv_ugr'] = (out['u'] - 2.0 * out['g'] + out['r']).astype('float32')
    out['curv_gri'] = (out['g'] - 2.0 * out['r'] + out['i']).astype('float32')
    out['curv_riz'] = (out['r'] - 2.0 * out['i'] + out['z']).astype('float32')

    rz = out['redshift'].astype('float32')
    rz_abs = np.abs(rz).astype('float32')
    out['redshift_abs'] = rz_abs
    out['redshift_log1p_abs'] = np.log1p(rz_abs).astype('float32')
    out['redshift_sq'] = (rz * rz).astype('float32')
    out['redshift_neg'] = (rz < 0).astype('float32')
    out['redshift_low'] = (rz < 0.02).astype('float32')
    out['redshift_mid'] = ((rz >= 0.02) & (rz < 0.7)).astype('float32')
    out['redshift_high'] = (rz >= 0.7).astype('float32')

    denom = rz_abs + 0.01
    for c in BANDS + ['u_g', 'g_r', 'r_i', 'i_z', 'u_r', 'g_i', 'r_z', 'mag_mean']:
        out[f'redshift_x_{c}'] = (rz * out[c]).astype('float32')
        out[f'{c}_over_rzabs'] = (out[c] / denom).astype('float32')

    alpha_rad = np.deg2rad(out['alpha'].to_numpy(dtype='float32'))
    delta_rad = np.deg2rad(out['delta'].to_numpy(dtype='float32'))
    cos_delta = np.cos(delta_rad)
    out['alpha_sin'] = np.sin(alpha_rad).astype('float32')
    out['alpha_cos'] = np.cos(alpha_rad).astype('float32')
    out['delta_sin'] = np.sin(delta_rad).astype('float32')
    out['delta_cos'] = cos_delta.astype('float32')
    out['sky_x'] = (cos_delta * np.cos(alpha_rad)).astype('float32')
    out['sky_y'] = (cos_delta * np.sin(alpha_rad)).astype('float32')
    out['sky_z'] = np.sin(delta_rad).astype('float32')

    out['uv_excess'] = (out['u_g'] - (0.75 * out['g_r'] + 0.18)).astype('float32')
    out['red_sequence_score'] = (out['u_r'] - 2.2).astype('float32')
    out['qso_color_plane'] = (out['u_g'] - 0.5 * out['g_r'] - 0.3 * rz).astype('float32')
    out['star_redshift_penalty'] = np.log1p(rz_abs * 100.0).astype('float32')
    return out


def qbin_series(values, q):
    ranks = pd.Series(values).rank(method='first')
    return pd.qcut(ranks, q=q, labels=False, duplicates='drop').astype('int32').astype(str)


def build_category_frame(all_raw, all_num):
    cat = pd.DataFrame(index=all_raw.index)
    for c in RAW_CATS:
        cat[c] = all_raw[c].astype(str).fillna('__NA__')

    cat['spectral_type_calc'] = pd.cut(
        all_raw['r'] - all_raw['g'],
        [-np.inf, -1.0, -0.5, 0.0, np.inf],
        labels=['M', 'G/K', 'A/F', 'O/B'],
    ).astype(str)
    cat['galaxy_population_calc'] = pd.cut(
        all_raw['u'] - all_raw['r'],
        [-np.inf, 2.2, np.inf],
        labels=['Blue_Cloud', 'Red_Sequence'],
    ).astype(str)
    cat['spectral_x_pop'] = cat['spectral_type'] + '|' + cat['galaxy_population']
    cat['spectral_calc_x_pop_calc'] = cat['spectral_type_calc'] + '|' + cat['galaxy_population_calc']

    qbin_cols = [
        'alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift',
        'u_g', 'g_r', 'r_i', 'i_z', 'u_r', 'g_i', 'r_z',
        'mag_mean', 'mag_std', 'mag_range', 'uv_excess',
    ]
    for c in qbin_cols:
        for q in [16, 32, 64, 128]:
            cat[f'{c}_q{q}'] = qbin_series(all_num[c].to_numpy(), q)

    round_cols = ['redshift', 'u_g', 'g_r', 'r_i', 'i_z', 'u_r', 'g_i', 'r_z', 'mag_mean', 'mag_std']
    for c in round_cols:
        cat[f'{c}_r1'] = np.round(all_num[c].to_numpy(dtype='float32'), 1).astype(str)
        cat[f'{c}_r2'] = np.round(all_num[c].to_numpy(dtype='float32'), 2).astype(str)

    cat['redshift_q32_x_spectral'] = cat['redshift_q32'] + '|' + cat['spectral_type']
    cat['redshift_q64_x_spectral'] = cat['redshift_q64'] + '|' + cat['spectral_type']
    cat['redshift_q64_x_pop'] = cat['redshift_q64'] + '|' + cat['galaxy_population']
    cat['redshift_q128_x_spectral_calc'] = cat['redshift_q128'] + '|' + cat['spectral_type_calc']
    cat['u_g_q32_x_g_r_q32'] = cat['u_g_q32'] + '|' + cat['g_r_q32']
    cat['u_g_q64_x_g_r_q64'] = cat['u_g_q64'] + '|' + cat['g_r_q64']
    cat['g_r_q64_x_r_i_q64'] = cat['g_r_q64'] + '|' + cat['r_i_q64']
    cat['r_i_q64_x_i_z_q64'] = cat['r_i_q64'] + '|' + cat['i_z_q64']
    cat['u_r_q64_x_redshift_q64'] = cat['u_r_q64'] + '|' + cat['redshift_q64']
    cat['alpha_q16_x_delta_q16'] = cat['alpha_q16'] + '|' + cat['delta_q16']
    cat['alpha_q32_x_delta_q32'] = cat['alpha_q32'] + '|' + cat['delta_q32']
    return cat


# ----------------------------------------------------------------------------
# Column profiles  (VERBATIM from source notebook)
# ----------------------------------------------------------------------------
BASE_CAT_COLS = [
    'spectral_type',
    'galaxy_population',
    'spectral_type_calc',
    'galaxy_population_calc',
    'spectral_x_pop',
    'spectral_calc_x_pop_calc',
]

COLOR_CAT_COLS = [
    'redshift_q32', 'redshift_q64', 'redshift_q128',
    'u_g_q64', 'g_r_q64', 'r_i_q64', 'i_z_q64', 'u_r_q64', 'g_i_q64', 'r_z_q64',
    'mag_mean_q64', 'mag_std_q64', 'mag_range_q64', 'uv_excess_q64',
    'redshift_r1', 'redshift_r2', 'u_g_r1', 'g_r_r1', 'r_i_r1', 'i_z_r1', 'u_r_r1', 'mag_mean_r1',
    'redshift_q64_x_spectral', 'redshift_q64_x_pop',
    'u_g_q64_x_g_r_q64', 'g_r_q64_x_r_i_q64', 'r_i_q64_x_i_z_q64', 'u_r_q64_x_redshift_q64',
]

SKY_CAT_COLS = [
    'alpha_q16', 'alpha_q32', 'alpha_q64',
    'delta_q16', 'delta_q32', 'delta_q64',
    'alpha_q16_x_delta_q16', 'alpha_q32_x_delta_q32',
]

FULL_CAT_COLS = BASE_CAT_COLS + SKY_CAT_COLS + COLOR_CAT_COLS + [
    'u_q64', 'g_q64', 'r_q64', 'i_q64', 'z_q64', 'redshift_q128_x_spectral_calc',
]

ROUNDED_CAT_COLS = BASE_CAT_COLS + [
    'redshift_r1', 'redshift_r2', 'u_g_r1', 'u_g_r2', 'g_r_r1', 'g_r_r2',
    'r_i_r1', 'r_i_r2', 'i_z_r1', 'i_z_r2', 'u_r_r1', 'u_r_r2',
    'g_i_r1', 'r_z_r1', 'mag_mean_r1', 'mag_mean_r2', 'mag_std_r1', 'mag_std_r2',
] + COLOR_CAT_COLS[:12]


def cat_columns_for(profile):
    if profile == 'numeric_only':
        cols = []
    elif profile in {'compact', 'minimal'}:
        cols = BASE_CAT_COLS + COLOR_CAT_COLS[:18] + SKY_CAT_COLS[:4]
    elif profile == 'color':
        cols = BASE_CAT_COLS + COLOR_CAT_COLS
    elif profile == 'sky':
        cols = BASE_CAT_COLS + SKY_CAT_COLS + ['redshift_q64', 'redshift_q128', 'redshift_q64_x_spectral']
    elif profile == 'rounded':
        cols = ROUNDED_CAT_COLS
    else:
        cols = FULL_CAT_COLS
    return list(dict.fromkeys(cols))


def numeric_columns_for(all_num, profile):
    if profile in {'numeric', 'numeric_only'}:
        return list(all_num.columns)
    if profile == 'color':
        keep = []
        tokens = ['u', 'g', 'r', 'i', 'z', 'redshift', 'mag_', 'curv_', 'slope', 'uv_', 'qso_', 'red_sequence']
        for c in all_num.columns:
            if any(t in c for t in tokens):
                keep.append(c)
        return list(dict.fromkeys(keep))
    if profile == 'sky':
        keep = BASE_NUMS + ['alpha_sin', 'alpha_cos', 'delta_sin', 'delta_cos', 'sky_x', 'sky_y', 'sky_z']
        keep += ['redshift_abs', 'redshift_log1p_abs', 'redshift_sq', 'u_g', 'g_r', 'r_i', 'i_z', 'u_r', 'mag_mean', 'mag_std']
        return [c for c in keep if c in all_num.columns]
    if profile == 'minimal':
        keep = BASE_NUMS + [
            'u_g', 'g_r', 'r_i', 'i_z', 'u_r', 'g_i', 'r_z',
            'mag_mean', 'mag_std', 'mag_range', 'redshift_abs', 'redshift_log1p_abs',
            'uv_excess', 'red_sequence_score', 'qso_color_plane',
        ]
        return [c for c in keep if c in all_num.columns]
    return list(all_num.columns)


def factorize_selected(cat, cols):
    codes = pd.DataFrame(index=cat.index)
    for c in cols:
        code, _ = pd.factorize(cat[c].astype(str).fillna('__NA__'), sort=True)
        codes[c] = code.astype('int32')
    return codes


def make_count_features(codes_tt, selected_cols):
    out = pd.DataFrame(index=codes_tt.index)
    n_rows = len(codes_tt)
    for c in selected_cols:
        codes = codes_tt[c].to_numpy(dtype='int32')
        cnt = np.bincount(codes, minlength=int(codes.max()) + 1).astype('float32')
        vals = cnt[codes]
        out[f'ce_{c}'] = np.log1p(vals).astype('float32')
        out[f'freq_{c}'] = (vals / n_rows).astype('float32')
    return out


def make_original_prior_features(codes_all, n_train, n_test, y_orig, selected_cols, smoothing=30.0):
    start_orig = n_train + n_test
    global_probs = np.bincount(y_orig, minlength=N_CLASSES).astype('float32')
    global_probs /= global_probs.sum()
    codes_tt = codes_all.iloc[: n_train + n_test]
    codes_orig = codes_all.iloc[start_orig:]
    out = pd.DataFrame(index=np.arange(n_train + n_test))
    for c in selected_cols:
        co = codes_orig[c].to_numpy(dtype='int32')
        ca = codes_tt[c].to_numpy(dtype='int32')
        max_code = max(int(codes_all[c].max()), 0)
        cnt = np.bincount(co, minlength=max_code + 1).astype('float32')
        denom = cnt + smoothing
        for k, cls in enumerate(CLASSES):
            cls_cnt = np.bincount(co, weights=(y_orig == k).astype('float32'), minlength=max_code + 1).astype('float32')
            prob = (cls_cnt + smoothing * global_probs[k]) / np.maximum(denom, EPS)
            out[f'orig_prior_{c}_{cls}'] = prob[ca].astype('float32')
        out[f'orig_count_{c}'] = np.log1p(cnt[ca]).astype('float32')
    return out


def make_prototype_features(all_num, n_train, n_test, y_orig):
    start_orig = n_train + n_test
    proto_cols = [
        'alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift',
        'u_g', 'g_r', 'r_i', 'i_z', 'u_r', 'g_i', 'r_z',
        'mag_mean', 'mag_std', 'mag_range', 'uv_excess', 'red_sequence_score',
        'sky_x', 'sky_y', 'sky_z',
    ]
    proto_cols = [c for c in proto_cols if c in all_num.columns]
    vals = all_num[proto_cols].to_numpy(dtype='float32')
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = StandardScaler().fit_transform(vals).astype('float32')
    tt = scaled[: n_train + n_test]
    orig_scaled = scaled[start_orig:]
    out = pd.DataFrame(index=np.arange(n_train + n_test))
    euclid_all = []
    diag_all = []
    for k, cls in enumerate(CLASSES):
        block = orig_scaled[y_orig == k]
        mu = block.mean(axis=0).astype('float32')
        sigma = block.std(axis=0).astype('float32') + 0.05
        diff = tt - mu
        euclid = np.sqrt(np.mean(diff * diff, axis=1)).astype('float32')
        diag = np.sqrt(np.mean((diff / sigma) ** 2, axis=1)).astype('float32')
        out[f'proto_euclid_{cls}'] = euclid
        out[f'proto_diag_{cls}'] = diag
        out[f'proto_absmean_{cls}'] = np.mean(np.abs(diff / sigma), axis=1).astype('float32')
        euclid_all.append(euclid)
        diag_all.append(diag)

    eu = np.vstack(euclid_all).T
    dg = np.vstack(diag_all).T
    eu_sort = np.sort(eu, axis=1)
    dg_sort = np.sort(dg, axis=1)
    out['proto_euclid_min'] = eu_sort[:, 0].astype('float32')
    out['proto_euclid_gap12'] = (eu_sort[:, 1] - eu_sort[:, 0]).astype('float32')
    out['proto_diag_min'] = dg_sort[:, 0].astype('float32')
    out['proto_diag_gap12'] = (dg_sort[:, 1] - dg_sort[:, 0]).astype('float32')
    out['proto_star_minus_qso'] = (eu[:, LABEL_MAP['STAR']] - eu[:, LABEL_MAP['QSO']]).astype('float32')
    out['proto_galaxy_minus_qso'] = (eu[:, LABEL_MAP['GALAXY']] - eu[:, LABEL_MAP['QSO']]).astype('float32')
    return out


# ----------------------------------------------------------------------------
# Build the model matrix  (VERBATIM, takes train/test/orig frames as args)
# ----------------------------------------------------------------------------
def build_features(train, test, orig, profile='full', native_categories=True):
    y = train[TARGET].map(LABEL_MAP).to_numpy(dtype='int32')
    orig_local = rebuild_original_cats(orig)
    y_orig = orig_local[TARGET].map(LABEL_MAP).to_numpy(dtype='int32')
    n_train, n_test = len(train), len(test)

    all_raw = pd.concat(
        [train.drop(columns=[TARGET]), test, orig_local[[*BASE_NUMS, *RAW_CATS]]],
        ignore_index=True,
    )
    all_num = add_science_features(all_raw)
    cat_all = build_category_frame(all_raw, all_num)

    cat_cols = cat_columns_for(profile)
    if profile == 'priorheavy':
        cat_cols = FULL_CAT_COLS + ROUNDED_CAT_COLS
    cat_cols = list(dict.fromkeys([c for c in cat_cols if c in cat_all.columns]))

    code_cols = sorted(set(cat_cols + FULL_CAT_COLS + ROUNDED_CAT_COLS))
    code_cols = [c for c in code_cols if c in cat_all.columns]
    codes_all = factorize_selected(cat_all, code_cols)

    parts = [all_num.iloc[: n_train + n_test][numeric_columns_for(all_num, profile)].reset_index(drop=True)]

    # Train/test count features. These use no target labels.
    parts.append(make_count_features(codes_all.iloc[: n_train + n_test].reset_index(drop=True), cat_cols))

    # Supervised priors from only the labeled original dataset.
    parts.append(make_original_prior_features(codes_all, n_train, n_test, y_orig, cat_cols))

    # Original-dataset class prototype distances in standardized science-feature space.
    parts.append(make_prototype_features(all_num, n_train, n_test, y_orig))

    native_cat_cols = cat_cols if native_categories else []
    if native_categories and cat_cols:
        cat_df = cat_all.iloc[: n_train + n_test][cat_cols].reset_index(drop=True).copy()
        for c in cat_cols:
            cat_df[c] = cat_df[c].astype(str).fillna('__NA__')
        parts.append(cat_df)
    else:
        code_df = codes_all.iloc[: n_train + n_test][cat_cols].reset_index(drop=True).astype('float32')
        code_df.columns = [f'code_{c}' for c in code_df.columns]
        parts.append(code_df)

    feature_df = pd.concat(parts, axis=1)
    num_cols = [c for c in feature_df.columns if c not in native_cat_cols]
    feature_df[num_cols] = feature_df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')

    X = feature_df.iloc[:n_train].reset_index(drop=True)
    X_test = feature_df.iloc[n_train : n_train + n_test].reset_index(drop=True)
    feature_info = pd.DataFrame({'feature': list(X.columns), 'is_cat': [c in native_cat_cols for c in X.columns]})

    log(f'Feature profile={profile}, train={X.shape}, test={X_test.shape}, native_cats={len(native_cat_cols)}')

    del orig_local, all_raw, all_num, cat_all, codes_all, feature_df
    gc.collect()
    return X, X_test, y, native_cat_cols, feature_info


# ----------------------------------------------------------------------------
# Model  (VERBATIM hyper-parameters from source notebook)
# ----------------------------------------------------------------------------
def class_weights_for_indices(y, idx):
    classes = np.arange(N_CLASSES)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y[idx]).astype('float32')
    return cw[y[idx]]


def lgbm_params(seed, threads=NUM_THREADS):
    return {
        'objective': 'multiclass',
        'num_class': N_CLASSES,
        'metric': 'multi_logloss',
        'learning_rate': 0.025,
        'n_estimators': 6000,
        'num_leaves': 80,
        'max_depth': -1,
        'min_child_samples': 80,
        'subsample': 0.82,
        'subsample_freq': 1,
        'colsample_bytree': 0.72,
        'reg_alpha': 0.05,
        'reg_lambda': 10.0,
        'random_state': seed,
        'num_threads': threads,
        'verbosity': -1,
    }


# ----------------------------------------------------------------------------
# Reporting helpers  (added for the contract results.txt)
# ----------------------------------------------------------------------------
def per_class_recall(y_true, y_pred):
    rec = {}
    for i, name in enumerate(CLASSES):
        mask = y_true == i
        rec[name] = float((y_pred[mask] == i).mean()) if mask.sum() else 0.0
    return rec


def normalize_proba(p):
    p = np.clip(p, CLIP, 1.0 - CLIP)
    p = p / p.sum(axis=1, keepdims=True)
    return p.astype('float32')


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    emit(f'# s6e6 lgbmv3 : faithful port of cdeotte LGBM-3 (source worker {SOURCE_MODEL})')
    emit(f'# seed={SEED} folds={N_FOLDS} profile=full native_categories=True')

    DATA_DIR = find_data_dir()
    ORIG_PATH = find_original_file()
    log(f'DATA_DIR={DATA_DIR}')
    log(f'ORIG_PATH={ORIG_PATH}')

    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    sample_submission = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    orig = pd.read_csv(ORIG_PATH)
    log(f'train={train.shape}, test={test.shape}, original={orig.shape}')

    assert len(train) == N_TRAIN_EXPECT, f'train rows {len(train)} != {N_TRAIN_EXPECT}'
    assert len(test) == N_TEST_EXPECT, f'test rows {len(test)} != {N_TEST_EXPECT}'

    # CONTRACT: test rows must be in sample_submission order so test preds align.
    if ID_COL in test.columns and ID_COL in sample_submission.columns:
        test = test.set_index(ID_COL).loc[sample_submission[ID_COL]].reset_index()
        log('test reordered to sample_submission id order')

    # Original-dataset normalization. The mirror LACKS spectral_type/galaxy_population;
    # rebuild_original_cats (called inside build_features) reconstructs them from colors.
    # First: uppercase class labels, keep only GALAXY/QSO/STAR, drop SDSS17 sentinel
    # band rows (e.g. -9999) so the color-based priors/prototypes are not poisoned.
    orig[TARGET] = orig[TARGET].astype(str).str.upper()
    orig = orig[orig[TARGET].isin(CLASSES)].reset_index(drop=True)
    n_before = len(orig)
    band_vals = orig[BANDS].apply(pd.to_numeric, errors='coerce')
    sane_mask = ((band_vals > -100.0) & (band_vals < 100.0)).all(axis=1)
    orig = orig[sane_mask].reset_index(drop=True)
    log(f'original sentinel/out-of-range band rows dropped: {n_before - len(orig)} '
        f'(orig now {orig.shape})')

    # Build the full feature matrix (profile="full", native pandas categoricals).
    X, X_test, y, CAT_COLS_NATIVE, feature_info = build_features(
        train, test, orig, profile='full', native_categories=True
    )
    n_cat = int(feature_info['is_cat'].sum())
    emit(f'# n_features={X.shape[1]} (native_categoricals={n_cat}) '
         f'n_orig_rows={len(orig)}')

    # Free the raw frames; keep only the model matrices.
    del train, test, orig
    gc.collect()

    # LightGBM native categoricals: mark the selected cols as pandas "category" dtype.
    for c in CAT_COLS_NATIVE:
        X[c] = X[c].astype('category')
        X_test[c] = X_test[c].astype('category')

    # CONTRACT fold split: StratifiedKFold(5, shuffle=True, random_state=42) over the
    # competition train rows in original CSV order. (Identical to the source notebook.)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), N_CLASSES), dtype='float32')
    test_preds = np.zeros((len(X_test), N_CLASSES), dtype='float32')
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y), dtype=np.int8), y), start=1):
        t0 = time.perf_counter()
        fold_seed = SEED + fold * 100
        log(f'===== Fold {fold}/{N_FOLDS} seed={fold_seed} =====')

        model = lgb.LGBMClassifier(**lgbm_params(fold_seed))
        callbacks = [
            lgb.log_evaluation(period=250),
            lgb.early_stopping(stopping_rounds=250, verbose=True),
        ]
        model.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            sample_weight=class_weights_for_indices(y, tr_idx),
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_sample_weight=[class_weights_for_indices(y, va_idx)],
            categorical_feature=CAT_COLS_NATIVE,
            callbacks=callbacks,
        )

        va_pred = model.predict_proba(X.iloc[va_idx]).astype('float32')
        te_pred = model.predict_proba(X_test).astype('float32')
        oof[va_idx] = va_pred
        test_preds += te_pred / N_FOLDS

        fold_score = balanced_accuracy_score(y[va_idx], np.argmax(va_pred, axis=1))
        best_iter = getattr(model, 'best_iteration_', None)
        elapsed = time.perf_counter() - t0
        fold_rows.append({'fold': fold, 'balanced_accuracy': float(fold_score),
                          'best_iteration': best_iter, 'elapsed_sec': elapsed})
        emit(f'Fold {fold} balanced accuracy={fold_score:.8f}, best_iteration={best_iter}, '
             f'elapsed={elapsed:.1f}s')
        flush_results()

        del model, va_pred, te_pred
        gc.collect()

    fold_scores = [r['balanced_accuracy'] for r in fold_rows]
    overall_score = balanced_accuracy_score(y, np.argmax(oof, axis=1))
    rec = per_class_recall(y, np.argmax(oof, axis=1))
    cm = confusion_matrix(y, np.argmax(oof, axis=1), labels=list(range(N_CLASSES)))

    emit('')
    emit('===== lgbmv3 (LightGBM, profile=full) =====')
    emit(f'  per-fold BA: {[round(float(s), 8) for s in fold_scores]}')
    emit(f'  mean fold BA: {np.mean(fold_scores):.8f}')
    emit(f'  OVERALL OOF balanced accuracy: {overall_score:.8f}')
    emit(f'  per-class recall: GALAXY={rec["GALAXY"]:.4f} QSO={rec["QSO"]:.4f} STAR={rec["STAR"]:.4f}')
    emit('  confusion (rows=true GALAXY/QSO/STAR, cols=pred):')
    for i, name in enumerate(CLASSES):
        emit(f'    {name:7s} {cm[i].tolist()}')
    flush_results()

    # Save artifacts: row-normalize + clip away from 0 so they stack cleanly.
    oof_out = normalize_proba(oof)
    test_out = normalize_proba(test_preds)
    np.save(OOF_PATH, oof_out.astype('float32'))
    np.save(PRED_PATH, test_out.astype('float32'))

    submission = sample_submission.copy()
    submission[TARGET] = np.argmax(test_out, axis=1)
    submission[TARGET] = submission[TARGET].map(INV_MAP)
    submission.to_csv(SUB_PATH, index=False)

    log(f'Saved OOF: {OOF_PATH} shape={oof_out.shape}')
    log(f'Saved test predictions: {PRED_PATH} shape={test_out.shape}')
    log(f'Saved submission: {SUB_PATH} shape={submission.shape}')

    emit('')
    emit('DONE')
    flush_results()
    log('ALL DONE')


if __name__ == '__main__':
    main()
