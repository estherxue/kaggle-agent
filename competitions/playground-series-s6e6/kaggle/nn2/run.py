"""s6e6 nn2 - DCN PyTorch NN (port of cdeotte-style nn-v2 / NN3-032).

Single level-1 PyTorch Deep & Cross Network per fold. No stacking/blending here.
Ported VERBATIM (model + config + FE) from the nn-v2 notebook, with:
  - StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on integer labels in CSV order.
  - Label mapping GALAXY=0, QSO=1, STAR=2 (alphabetical), matching existing artifacts.
  - Outputs renamed to oof_nn2.npy / test_nn2.npy in [GALAXY,QSO,STAR] order, train-CSV / sample_submission order.
  - results.txt written to /kaggle/working with per-fold BAC, overall OOF BAC, per-class recall.
  - All plotting / display removed. Self-contained (no project imports).

Runs training ONLY on Kaggle (GPU). Do not execute locally.
"""

import os

# Kaggle exposes its GPU as device 0. Override locally before running if desired.
_ = os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
# Helps reduce fragmentation on memory-constrained T4 sessions. Must be set before importing torch.
_ = os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import gc
import glob
import json
import random
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# --- Pascal-compatible torch ---
# Kaggle's stock torch 2.10+cu128 dropped sm_60, but this batch kernel may land on a P100.
# Install a cu121 build (supports sm_60 P100 AND sm_75 T4) BEFORE importing torch.
import sys as _sys, subprocess as _sp
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "torch==2.4.1",
         "--extra-index-url", "https://download.pytorch.org/whl/cu121"], check=False)

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
pd.set_option('display.max_columns', 220)
pd.set_option('display.width', 220)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

T0 = time.perf_counter()


def log(msg):
    print(f'[{time.perf_counter() - T0:8.1f}s] {msg}', flush=True)


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    GPU_TOTAL_GB = props.total_memory / 1024 ** 3
    print(f'GPU: {props.name} | memory={GPU_TOTAL_GB:.2f} GiB')
else:
    GPU_TOTAL_GB = 0.0

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MODEL_ID = 'nn2b'                       # output artifact stem: oof_nn2.npy / test_nn2.npy
SOURCE_EXPERIMENT = 'NN3-032'
SOURCE_EXPECTED_CV = 0.96643599
SEED = 42                              # fold-split seed (CRITICAL: must be 42 for alignment)
MODEL_SEED = 770411
N_FOLDS = 5
TARGET = 'class'
ID = 'id'

CLASSES = np.array(['GALAXY', 'QSO', 'STAR'])   # GALAXY=0, QSO=1, STAR=2
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}

RAW_NUMS = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
BANDS = ['u', 'g', 'r', 'i', 'z']
BASE_CATS = ['spectral_type', 'galaxy_population']
EPS = np.float32(1e-6)
MISSING = '__MISSING__'

WORK = Path('/kaggle/working')
WORK.mkdir(parents=True, exist_ok=True)

OOF_PATH = WORK / f'oof_{MODEL_ID}.npy'
PRED_PATH = WORK / f'test_{MODEL_ID}.npy'
SUB_PATH = WORK / 'submission.csv'
LABEL_PATH = WORK / f'{MODEL_ID}_oof_labels.npy'
FOLD_PATH = WORK / f'{MODEL_ID}_oof_fold.npy'
SCORES_PATH = WORK / f'{MODEL_ID}_fold_scores.csv'
RESULT_PATH = WORK / f'{MODEL_ID}_result.json'
RESULTS_TXT = WORK / 'results.txt'

CFG = SimpleNamespace(
    name=MODEL_ID,
    source_experiment=SOURCE_EXPERIMENT,
    seed=MODEL_SEED,
    threads=12,
    epochs=84,
    patience=16,
    # NN3-032 used batch_size=4096 directly. This keeps the same effective batch
    # with T4-safe 1024-row micro-batches and 4-step accumulation.
    # If a fragmented T4 still OOMs, use micro_batch_size=512 and grad_accum_steps=8.
    batch_size=4096,
    micro_batch_size=1024,
    grad_accum_steps=4,
    eval_batch_size=8192,
    hidden=768,
    blocks=4,
    cross_layers=7,
    dropout=0.10,
    emb_dropout=0.05,
    max_emb_dim=32,
    lr=0.0006,
    weight_decay=1.0e-4,
    label_smoothing=0.002,
    ema_decay=0.995,
    class_weight_power=1.0,
    focal_gamma=0.65,
    grad_clip=5.0,
    qbins=32,
    qbin_levels=[16, 32, 64],
    qbin_cols=28,
    rare_min_count=10,
    max_num=0,
    max_cat=0,
    append_original=True,
    original_weight=0.12,
    artifact_lowfreq=True,
    artifact_counts=True,
    artifact_count_scope='all',
    balanced_sampler=False,
    amp=torch.cuda.is_available(),
    pin_memory=torch.cuda.is_available(),
    report_every=5,
)

seed_everything(SEED)
torch.set_num_threads(CFG.threads)
try:
    torch.set_num_interop_threads(max(1, min(4, CFG.threads // 4)))
except RuntimeError:
    pass

print(CFG)

# ----------------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------------
def find_competition_root():
    candidates = [
        Path('/kaggle/input/competitions/playground-series-s6e6'),
        Path('/kaggle/input/playground-series-s6e6'),
    ]
    candidates += [Path(p).parent for p in glob.glob('/kaggle/input/*/train.csv')]
    candidates += [Path(p).parent for p in glob.glob('/kaggle/input/**/train.csv', recursive=True)]
    for root in candidates:
        if (root / 'train.csv').exists() and (root / 'test.csv').exists() and (root / 'sample_submission.csv').exists():
            return root
    raise FileNotFoundError('Could not find train.csv, test.csv, and sample_submission.csv.')


def find_original_path(data_root):
    candidates = [
        data_root / 'star_classification.csv',
        Path('/kaggle/input/datasets/fedesoriano/stellar-classification-dataset-sdss17/star_classification.csv'),
    ]
    candidates += [Path(p) for p in glob.glob('/kaggle/input/**/star_classification.csv', recursive=True)]
    candidates += [Path(p) for p in glob.glob('/kaggle/input/**/stellar_classification.csv', recursive=True)]
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    return None


DATA_ROOT = find_competition_root()
ORIGINAL_PATH = find_original_path(DATA_ROOT)
print('Competition data root:', DATA_ROOT)
print('Original data file:   ', ORIGINAL_PATH)

train_raw = pd.read_csv(DATA_ROOT / 'train.csv')
test_raw = pd.read_csv(DATA_ROOT / 'test.csv')
sample_submission = pd.read_csv(DATA_ROOT / 'sample_submission.csv')

# The NN3-032 recipe REQUIRES the original SDSS17 rows (append_original=True,
# original_weight=0.12). Dropping them silently trains a weaker, non-reproducing model,
# so fail loudly if the dataset is not attached. The dataset is mounted via
# kernel-metadata.json dataset_sources=["fedesoriano/stellar-classification-dataset-sdss17"].
if CFG.append_original and ORIGINAL_PATH is None:
    raise FileNotFoundError(
        'CFG.append_original=True but the original SDSS star_classification.csv was not found '
        'under /kaggle/input. Attach the dataset "fedesoriano/stellar-classification-dataset-sdss17" '
        'via kernel-metadata.json dataset_sources, or set CFG.append_original=False to train without it '
        '(this will NOT reproduce the NN3-032 reference CV).'
    )

if ORIGINAL_PATH is not None:
    original_raw = pd.read_csv(ORIGINAL_PATH)
    original_raw = original_raw[original_raw[TARGET].isin(CLASSES)].copy()
else:
    # Only reached when CFG.append_original is already False.
    print('NOTE: original SDSS star_classification.csv not attached; '
          'proceeding WITHOUT appended original rows (append_original=False).')
    original_raw = pd.DataFrame(columns=list(train_raw.columns))

# Align test rows to sample_submission order (sample_submission defines test row order).
if ID in test_raw.columns and ID in sample_submission.columns:
    test_raw = sample_submission[[ID]].merge(test_raw, on=ID, how='left').reset_index(drop=True)

print('train:', train_raw.shape)
print('test :', test_raw.shape)
print('orig :', original_raw.shape)

# ----------------------------------------------------------------------------
# Feature Engineering (verbatim from notebook)
# ----------------------------------------------------------------------------
def spectral_type_from_gr(g, r):
    return pd.cut(
        r - g,
        [-np.inf, -1.0, -0.5, 0.0, np.inf],
        labels=['M', 'G/K', 'A/F', 'O/B'],
    ).astype(str)


def galaxy_population_from_ur(u, r):
    return pd.cut(
        u - r,
        [-np.inf, 2.2, np.inf],
        labels=['Blue_Cloud', 'Red_Sequence'],
    ).astype(str)


def add_series(out, name, values, dtype='float32'):
    out[name] = np.asarray(values, dtype=dtype)


def fixed_bin(values, edges, labels):
    return pd.cut(values, [-np.inf] + list(edges) + [np.inf], labels=labels).astype(str)


def prepare_raw_tables(train, test, original):
    for df in (train, test, original):
        for c in RAW_NUMS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    if len(original) > 0:
        if 'spectral_type' not in original.columns:
            original['spectral_type'] = spectral_type_from_gr(original['g'], original['r'])
        if 'galaxy_population' not in original.columns:
            original['galaxy_population'] = galaxy_population_from_ur(original['u'], original['r'])
    for df in (train, test, original):
        for c in BASE_CATS:
            if c in df.columns:
                df[c] = df[c].astype('string').fillna(MISSING).astype(str)
    y = train[TARGET].map(CLASS_TO_INT).astype('int64').to_numpy()
    if len(original) > 0:
        y_original = original[TARGET].map(CLASS_TO_INT).astype('int64').to_numpy()
    else:
        y_original = np.zeros(0, dtype='int64')
    return train, test, original, y, y_original


def build_features(df):
    out = pd.DataFrame(index=df.index)
    for c in RAW_NUMS:
        out[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')

    for c in BASE_CATS:
        if c in df.columns:
            out[c] = df[c].astype('string').fillna(MISSING).astype(str)

    # SDSS colors. Magnitude differences are log flux ratios.
    for a in BANDS:
        for b in BANDS:
            if BANDS.index(a) < BANDS.index(b):
                out[f'{a}_{b}'] = (out[a] - out[b]).astype('float32')
                out[f'{a}_{b}_abs'] = out[f'{a}_{b}'].abs().astype('float32')

    mags = out[BANDS].to_numpy(dtype='float32')
    add_series(out, 'mag_mean', np.nanmean(mags, axis=1))
    add_series(out, 'mag_std', np.nanstd(mags, axis=1))
    add_series(out, 'mag_min', np.nanmin(mags, axis=1))
    add_series(out, 'mag_max', np.nanmax(mags, axis=1))
    out['mag_range'] = (out['mag_max'] - out['mag_min']).astype('float32')

    red = out['redshift'].to_numpy(dtype='float32')
    add_series(out, 'redshift_abs', np.abs(red))
    add_series(out, 'redshift_log1p_abs', np.log1p(np.abs(red)))
    add_series(out, 'redshift_sq', np.square(red))

    gr = out['g_r'].astype('float32')
    ri = out['r_i'].astype('float32')
    ug = out['u_g'].astype('float32')
    ur = out['u_r'].astype('float32')

    out['uvx_u_g_minus_0p6'] = (ug - 0.6).astype('float32')
    out['uvx_plane'] = (ug - 0.75 * gr - 0.18).astype('float32')
    out['lrg_cperp'] = (ri - gr / 4.0 - 0.18).astype('float32')
    out['lrg_cparallel'] = (0.7 * gr + 1.2 * (ri - 0.18)).astype('float32')
    out['qso_color_plane'] = (ug - 0.8 * gr + 0.35 * ri - 0.2 * out['i_z']).astype('float32')
    out['u_r_minus_2p2'] = (ur - 2.2).astype('float32')

    z_clip = np.clip(out['redshift'].to_numpy(dtype='float32'), -0.05, 5.0)
    out['star_z_peak'] = np.exp(-np.abs(z_clip) / np.float32(0.003)).astype('float32')
    out['galaxy_z_low_peak'] = np.exp(-np.square((z_clip - np.float32(0.1)) / np.float32(0.25))).astype('float32')
    out['qso_z_high_sigmoid'] = (1.0 / (1.0 + np.exp(-4.0 * (z_clip - np.float32(0.7))))).astype('float32')
    out['qso_z_veryhigh_sigmoid'] = (1.0 / (1.0 + np.exp(-3.0 * (z_clip - np.float32(2.2))))).astype('float32')

    out['spectral_type_calc'] = spectral_type_from_gr(out['g'], out['r'])
    out['galaxy_population_calc'] = galaxy_population_from_ur(out['u'], out['r'])
    out['spectral_calc_x_pop_calc'] = out['spectral_type_calc'].astype(str) + '__' + out['galaxy_population_calc'].astype(str)

    out['redshift_regime'] = fixed_bin(out['redshift'], [0.003, 0.02, 0.1, 0.7, 2.2], ['near0', 'tiny', 'low', 'mid', 'high', 'qsohigh'])
    out['uvx_bin'] = fixed_bin(out['u_g'], [0.6, 1.5], ['uvx', 'normal', 'red'])
    out['g_r_spec_bin'] = fixed_bin(out['g_r'], [0.0, 0.5, 1.0], ['blue', 'af', 'gk', 'm'])
    out['u_r_pop_bin'] = fixed_bin(out['u_r'], [2.2], ['bluecloud', 'redseq'])
    out['redshift_x_uvx'] = out['redshift_regime'].astype(str) + '__' + out['uvx_bin'].astype(str)
    out['redshift_x_pop'] = out['redshift_regime'].astype(str) + '__' + out['u_r_pop_bin'].astype(str)

    keep_num = RAW_NUMS + [
        'u_g', 'g_r', 'r_i', 'i_z', 'u_r', 'u_z', 'g_z',
        'mag_mean', 'mag_std', 'mag_range',
        'redshift_abs', 'redshift_log1p_abs', 'redshift_sq',
        'star_z_peak', 'galaxy_z_low_peak', 'qso_z_high_sigmoid', 'qso_z_veryhigh_sigmoid',
        'uvx_u_g_minus_0p6', 'u_r_minus_2p2',
        'lrg_cperp', 'lrg_cparallel', 'qso_color_plane',
    ]
    keep_cat = [
        'spectral_type_calc', 'galaxy_population_calc', 'spectral_calc_x_pop_calc',
        'redshift_regime', 'uvx_bin', 'g_r_spec_bin', 'redshift_x_uvx', 'redshift_x_pop',
    ]

    out = out[keep_num + keep_cat].copy()
    for c in keep_num:
        out[c] = pd.to_numeric(out[c], errors='coerce').replace([np.inf, -np.inf], np.nan).astype('float32')
    for c in keep_cat:
        out[c] = out[c].astype('string').fillna(MISSING).astype(str)
    return out


def _fit_rank_bins(series_list, n_bins):
    values = pd.concat([pd.to_numeric(s, errors='coerce') for s in series_list], ignore_index=True).astype('float32')
    values = values[np.isfinite(values)]
    edges = np.unique(np.nanquantile(values.to_numpy(dtype='float32'), np.linspace(0, 1, n_bins + 1, dtype='float32')[1:-1])).astype('float32')
    return edges


def _transform_rank_bins(series, edges):
    vals = pd.to_numeric(series, errors='coerce').astype('float32')
    arr = vals.to_numpy(dtype='float32')
    ids = np.zeros(len(arr), dtype='int32')
    good = np.isfinite(arr)
    ids[good] = np.searchsorted(edges, arr[good], side='right').astype('int32')
    return pd.Series(ids, index=series.index).astype('string').fillna(MISSING).astype(str)


def _floor_cat(series):
    return np.floor(pd.to_numeric(series, errors='coerce')).astype('Int64').astype('string').fillna(MISSING).astype(str)


def add_artifact_features(train_feat, test_feat, orig_feat):
    if not (CFG.artifact_lowfreq or CFG.artifact_counts):
        return train_feat, test_feat, orig_feat

    frames = [train_feat.copy(), test_feat.copy(), orig_feat.copy()]
    artifact_cat_cols = []

    def add_col(name, values):
        for df, val in zip(frames, values):
            df[name] = val.astype('string').fillna(MISSING).astype(str)
        artifact_cat_cols.append(name)

    if CFG.artifact_lowfreq:
        for c in RAW_NUMS:
            if all(c in df.columns for df in frames):
                add_col(f'art_{c}_floor', [_floor_cat(df[c]) for df in frames])

        if all(c in frames[0].columns for c in ['alpha', 'delta']):
            add_col(
                'art_alpha_floor_x_delta_floor',
                [(_floor_cat(df['alpha']) + '__' + _floor_cat(df['delta'])) for df in frames],
            )
        if all(c in frames[0].columns for c in ['u', 'z']):
            add_col(
                'art_u_floor_x_z_floor',
                [(_floor_cat(df['u']) + '__' + _floor_cat(df['z'])) for df in frames],
            )

        for n_bins in [100, 500]:
            if all('delta' in df.columns for df in frames):
                edges = _fit_rank_bins([df['delta'] for df in frames], n_bins)
                add_col(f'art_delta_q{n_bins}', [_transform_rank_bins(df['delta'], edges) for df in frames])

    if CFG.artifact_counts:
        count_cols = list(dict.fromkeys(artifact_cat_cols))
        if CFG.artifact_count_scope == 'pair_delta':
            count_cols = [c for c in count_cols if '_x_' in c or 'delta_q' in c]
        elif CFG.artifact_count_scope == 'pairs_only':
            count_cols = [c for c in count_cols if '_x_' in c]
        for c in count_cols:
            tt = pd.concat([frames[0][c], frames[1][c]], ignore_index=True).astype(str)
            tt_counts = tt.value_counts()
            total = np.float32(len(tt))
            for df in frames:
                cnt = df[c].astype(str).map(tt_counts).fillna(0).astype('float32').to_numpy()
                df[f'art_count_log_{c}'] = np.log1p(cnt).astype('float32')
                df[f'art_freq_{c}'] = (cnt / total).astype('float32')

    return frames[0], frames[1], frames[2]


def split_columns(df):
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' or str(df[c].dtype).startswith('string')]
    for c in ['mag_argmin', 'mag_argmax', 'spectral_match', 'pop_match']:
        if c in df.columns and c not in cat_cols:
            cat_cols.append(c)
    num_cols = [c for c in df.columns if c not in cat_cols]
    return num_cols, cat_cols


train_raw, test_raw, original_raw, y, y_original = prepare_raw_tables(train_raw, test_raw, original_raw)
train_feat = build_features(train_raw.drop(columns=[TARGET]))
test_feat = build_features(test_raw)
if len(original_raw) > 0:
    original_feat = build_features(original_raw.drop(columns=[TARGET]))
else:
    original_feat = pd.DataFrame(columns=train_feat.columns)
train_feat, test_feat, original_feat = add_artifact_features(train_feat, test_feat, original_feat)
num_cols, cat_cols = split_columns(train_feat)

log(
    f'features source={SOURCE_EXPERIMENT} num={len(num_cols)} cat={len(cat_cols)} '
    f'first_num={num_cols[:18]} first_cat={cat_cols[:18]}'
)

# ----------------------------------------------------------------------------
# Encoding Helpers (verbatim)
# ----------------------------------------------------------------------------
def choose_qbin_cols(num_cols):
    preferred = [
        'redshift', 'redshift_abs', 'redshift_log1p_abs', 'u_g', 'g_r', 'r_i', 'i_z', 'u_r', 'u_z', 'g_z',
        'uvx_u_g_minus_0p6', 'uvx_plane', 'lrg_cperp', 'lrg_cparallel', 'qso_color_plane',
        'u_r_minus_2p2', 'mag_std', 'mag_range', 'flux_std', 'f_opt_eboss',
    ]
    out = [c for c in preferred if c in num_cols]
    out += [c for c in num_cols if c not in out]
    return out


def fit_cat_maps(df, cat_cols, min_count):
    maps = []
    cards = []
    for c in cat_cols:
        s = df[c].astype('string').fillna(MISSING).astype(str)
        vc = s.value_counts()
        keep = vc[vc >= min_count].index
        mapping = {v: i + 1 for i, v in enumerate(keep)}
        maps.append(mapping)
        cards.append(len(mapping) + 1)
    return maps, cards


def encode_cats(df, cat_cols, maps):
    if not cat_cols:
        return np.zeros((len(df), 0), dtype='int32')
    out = np.zeros((len(df), len(cat_cols)), dtype='int32')
    for j, (c, mapping) in enumerate(zip(cat_cols, maps)):
        out[:, j] = df[c].astype('string').fillna(MISSING).astype(str).map(mapping).fillna(0).astype('int32').to_numpy()
    return out


def fit_qbins(df, qcols, n_bins):
    qs = np.linspace(0, 1, n_bins + 1, dtype='float32')[1:-1]
    edges = []
    cards = []
    for c in qcols:
        v = pd.to_numeric(df[c], errors='coerce').to_numpy(dtype='float32')
        v = v[np.isfinite(v)]
        e = np.unique(np.nanquantile(v, qs).astype('float32')) if len(v) else np.array([], dtype='float32')
        edges.append(e)
        cards.append(len(e) + 2)
    return edges, cards


def transform_qbins(df, qcols, edges):
    if not qcols:
        return np.zeros((len(df), 0), dtype='int32')
    out = np.zeros((len(df), len(qcols)), dtype='int32')
    for j, (c, e) in enumerate(zip(qcols, edges)):
        v = pd.to_numeric(df[c], errors='coerce').to_numpy(dtype='float32')
        good = np.isfinite(v)
        ids = np.zeros(len(v), dtype='int32')
        ids[good] = np.searchsorted(e, v[good], side='right').astype('int32') + 1
        out[:, j] = ids
    return out


def make_fold_arrays(tr_idx, va_idx):
    num_cols, cat_cols = split_columns(train_feat)
    if CFG.max_num and len(num_cols) > CFG.max_num:
        q_pref = choose_qbin_cols(num_cols)
        num_cols = list(dict.fromkeys(q_pref + num_cols))[:CFG.max_num]
    if CFG.max_cat and len(cat_cols) > CFG.max_cat:
        cat_cols = cat_cols[:CFG.max_cat]

    tr_df = train_feat.iloc[tr_idx].reset_index(drop=True)
    va_df = train_feat.iloc[va_idx].reset_index(drop=True)
    te_df = test_feat.reset_index(drop=True)
    fit_df = tr_df
    y_tr = y[tr_idx]
    sample_weight = None

    if CFG.append_original and len(original_feat) > 0:
        fit_df = pd.concat([tr_df, original_feat.reset_index(drop=True)], axis=0, ignore_index=True)
        y_tr = np.concatenate([y_tr, y_original]).astype('int64')
        sample_weight = np.concatenate([
            np.ones(len(tr_df), dtype='float32'),
            np.full(len(original_feat), CFG.original_weight, dtype='float32'),
        ])

    scaler = StandardScaler()
    xtr_num = scaler.fit_transform(fit_df[num_cols].fillna(0).to_numpy(dtype='float32')).astype('float32')
    xva_num = scaler.transform(va_df[num_cols].fillna(0).to_numpy(dtype='float32')).astype('float32')
    xte_num = scaler.transform(te_df[num_cols].fillna(0).to_numpy(dtype='float32')).astype('float32')

    maps, cards = fit_cat_maps(fit_df, cat_cols, CFG.rare_min_count)
    xtr_cat = encode_cats(fit_df, cat_cols, maps)
    xva_cat = encode_cats(va_df, cat_cols, maps)
    xte_cat = encode_cats(te_df, cat_cols, maps)

    qcols = choose_qbin_cols(num_cols)[:CFG.qbin_cols]
    levels = CFG.qbin_levels if CFG.qbin_levels else [CFG.qbins]
    q_cards = []
    qtr_parts = []
    qva_parts = []
    qte_parts = []
    for n_bins in levels:
        edges, level_cards = fit_qbins(fit_df, qcols, n_bins)
        q_cards.extend(level_cards)
        qtr_parts.append(transform_qbins(fit_df, qcols, edges))
        qva_parts.append(transform_qbins(va_df, qcols, edges))
        qte_parts.append(transform_qbins(te_df, qcols, edges))

    if qtr_parts:
        xtr_cat = np.hstack([xtr_cat] + qtr_parts).astype('int32', copy=False)
        xva_cat = np.hstack([xva_cat] + qva_parts).astype('int32', copy=False)
        xte_cat = np.hstack([xte_cat] + qte_parts).astype('int32', copy=False)
    cards = cards + q_cards
    return xtr_num, xva_num, xte_num, xtr_cat, xva_cat, xte_cat, y_tr, y[va_idx], sample_weight, cards, num_cols, cat_cols, qcols


# ----------------------------------------------------------------------------
# Model (verbatim)
# ----------------------------------------------------------------------------
class TabDataset(Dataset):
    def __init__(self, x_num, x_cat, y=None, sample_weight=None):
        self.x_num = torch.from_numpy(x_num.astype('float32', copy=False))
        self.x_cat = torch.from_numpy(x_cat.astype('int64', copy=False))
        self.y = torch.from_numpy(y.astype('int64', copy=False)) if y is not None else None
        self.sample_weight = torch.from_numpy(sample_weight.astype('float32', copy=False)) if sample_weight is not None else None

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x_num[idx], self.x_cat[idx]
        if self.sample_weight is None:
            return self.x_num[idx], self.x_cat[idx], self.y[idx]
        return self.x_num[idx], self.x_cat[idx], self.y[idx], self.sample_weight[idx]


def emb_dim(card, max_dim):
    return min(max_dim, max(2, int(round(card ** 0.25 * 3.5))))


class DenseInput(nn.Module):
    def __init__(self, n_num, cards, max_emb_dim, emb_dropout):
        super().__init__()
        self.embs = nn.ModuleList()
        emb_out = 0
        for card in cards:
            dim = emb_dim(card, max_emb_dim)
            emb = nn.Embedding(card, dim)
            nn.init.normal_(emb.weight, 0, 0.02)
            self.embs.append(emb)
            emb_out += dim
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.out_dim = n_num + emb_out

    def forward(self, x_num, x_cat):
        if not self.embs:
            return x_num
        emb = torch.cat([e(x_cat[:, j]) for j, e in enumerate(self.embs)], dim=1)
        return torch.cat([x_num, self.emb_dropout(emb)], dim=1)


class CrossNet(nn.Module):
    def __init__(self, dim, layers):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(dim) * 0.01) for _ in range(layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(dim)) for _ in range(layers)])

    def forward(self, x0):
        x = x0
        for w, b in zip(self.weights, self.biases):
            xw = torch.sum(x * w, dim=1, keepdim=True)
            x = x0 * xw + b + x
        return x


class DCN(nn.Module):
    def __init__(self, n_num, cards, hidden, blocks, cross_layers, dropout, emb_dropout, max_emb_dim, n_classes=3):
        super().__init__()
        self.input = DenseInput(n_num, cards, max_emb_dim, emb_dropout)
        self.norm = nn.LayerNorm(self.input.out_dim)
        self.cross = CrossNet(self.input.out_dim, cross_layers)
        layers = [nn.Linear(self.input.out_dim, hidden), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(blocks - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout)]
        self.deep = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.LayerNorm(self.input.out_dim + hidden), nn.Linear(self.input.out_dim + hidden, n_classes))

    def forward(self, x_num, x_cat):
        x0 = self.norm(self.input(x_num, x_cat))
        return self.head(torch.cat([self.cross(x0), self.deep(x0)], dim=1))


def build_model(n_num, cards):
    return DCN(
        n_num=n_num,
        cards=cards,
        hidden=CFG.hidden,
        blocks=CFG.blocks,
        cross_layers=CFG.cross_layers,
        dropout=CFG.dropout,
        emb_dropout=CFG.emb_dropout,
        max_emb_dim=CFG.max_emb_dim,
        n_classes=len(CLASSES),
    )


def compute_loss(logits, yb, class_weights, sample_weight=None):
    # Class-mean cross entropy (targets balanced accuracy better than class-frequency weighting).
    loss = F.cross_entropy(logits, yb, label_smoothing=CFG.label_smoothing, reduction='none')
    parts = []
    for cls in range(len(CLASSES)):
        mask = yb == cls
        if torch.any(mask):
            cls_loss = loss[mask]
            if sample_weight is not None:
                cls_w = sample_weight[mask]
                parts.append((cls_loss * cls_w).sum() / cls_w.sum().clamp_min(1e-6))
            else:
                parts.append(cls_loss.mean())
    return torch.stack(parts).mean()


@torch.no_grad()
def predict(model, loader):
    model.eval()
    chunks = []
    amp_enabled = CFG.amp and device.type == 'cuda'
    for xb_num, xb_cat in loader:
        xb_num = xb_num.to(device, non_blocking=True)
        xb_cat = xb_cat.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
            logits = model(xb_num, xb_cat)
        chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.vstack(chunks).astype('float32')


def update_ema_model(model, ema_model, decay):
    model_state = model.state_dict()
    ema_state = ema_model.state_dict()
    with torch.no_grad():
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)


# ----------------------------------------------------------------------------
# Train 5-Fold CV (verbatim)
# ----------------------------------------------------------------------------
def train_fold(fold, arrays, class_weights):
    xtr_num, xva_num, xte_num, xtr_cat, xva_cat, xte_cat, y_tr, y_va, sample_weight, cards, *_ = arrays
    seed_everything(CFG.seed + fold)

    ds_tr = TabDataset(xtr_num, xtr_cat, y_tr, sample_weight)
    ds_va = TabDataset(xva_num, xva_cat)
    ds_te = TabDataset(xte_num, xte_cat)

    sampler = None
    shuffle = True
    if CFG.balanced_sampler:
        counts = np.bincount(y_tr, minlength=len(CLASSES)).astype('float32')
        weights = 1.0 / np.maximum(counts[y_tr], 1.0)
        if sample_weight is not None:
            weights = weights * sample_weight
        sampler = WeightedRandomSampler(torch.from_numpy(weights.astype('float32')), num_samples=len(y_tr), replacement=True)
        shuffle = False

    dl_tr = DataLoader(
        ds_tr,
        batch_size=CFG.micro_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=True,
        num_workers=0,
        pin_memory=CFG.pin_memory,
    )
    dl_va = DataLoader(ds_va, batch_size=CFG.eval_batch_size, shuffle=False, num_workers=0, pin_memory=CFG.pin_memory)
    dl_te = DataLoader(ds_te, batch_size=CFG.eval_batch_size, shuffle=False, num_workers=0, pin_memory=CFG.pin_memory)

    model = build_model(xtr_num.shape[1], cards).to(device)
    ema_model = build_model(xtr_num.shape[1], cards).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=CFG.lr * 0.02)
    scaler = torch.amp.GradScaler('cuda', enabled=CFG.amp and device.type == 'cuda')

    best_bac = -1.0
    best_loss = np.inf
    best_epoch = 0
    best_state = None
    stale = 0
    grad_accum_steps = max(1, int(CFG.grad_accum_steps))
    amp_enabled = CFG.amp and device.type == 'cuda'

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        n_batches = len(dl_tr)

        for step, batch in enumerate(dl_tr, start=1):
            xb_num = batch[0].to(device, non_blocking=True)
            xb_cat = batch[1].to(device, non_blocking=True)
            yb = batch[2].to(device, non_blocking=True)
            wb = batch[3].to(device, non_blocking=True) if len(batch) == 4 else None

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
                loss = compute_loss(model(xb_num, xb_cat), yb, class_weights, wb)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if step % grad_accum_steps == 0 or step == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_ema_model(model, ema_model, CFG.ema_decay)

        scheduler.step()
        va_prob = predict(ema_model, dl_va)
        bac = balanced_accuracy_score(y_va, va_prob.argmax(axis=1))
        ll = log_loss(y_va, va_prob, labels=list(range(len(CLASSES))))
        improved = (bac > best_bac + 1e-7) or (abs(bac - best_bac) <= 1e-7 and ll < best_loss)
        if improved:
            best_bac = float(bac)
            best_loss = float(ll)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= CFG.patience:
                log(f'{MODEL_ID} fold={fold} early stop at epoch={epoch}, best_BAC={best_bac:.8f}')
                break

        if epoch == 1 or epoch % CFG.report_every == 0:
            log(f'{MODEL_ID} fold={fold} epoch={epoch:03d} BAC={bac:.8f} logloss={ll:.6f} best={best_bac:.8f}')

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)
    va_prob = predict(model, dl_va)
    te_prob = predict(model, dl_te)
    fold_bac = balanced_accuracy_score(y_va, va_prob.argmax(axis=1))
    fold_loss = log_loss(y_va, va_prob, labels=list(range(len(CLASSES))))

    del model, ema_model, optimizer, scheduler, scaler, ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te, best_state
    cleanup_cuda()
    return va_prob, te_prob, fold_bac, fold_loss, best_epoch


counts = np.bincount(y, minlength=len(CLASSES)).astype('float32')
class_weight_values = counts.sum() / (len(CLASSES) * counts)
class_weight_values = np.power(class_weight_values, CFG.class_weight_power).astype('float32')
class_weights = torch.tensor(class_weight_values, dtype=torch.float32, device=device)
print('Class counts:', dict(zip(CLASSES, counts.astype(int))))
print('Class weights:', dict(zip(CLASSES, class_weight_values)))

oof_preds = np.zeros((len(train_raw), len(CLASSES)), dtype='float32')
test_preds = np.zeros((len(test_raw), len(CLASSES)), dtype='float32')
oof_fold = np.zeros(len(train_raw), dtype='int16')
fold_rows = []

# CRITICAL: identical y + n_splits + random_state => identical folds across all artifacts.
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y), dtype=np.int8), y), start=1):
    t0 = time.perf_counter()
    arrays = make_fold_arrays(tr_idx, va_idx)
    log(
        f'{MODEL_ID} fold={fold} rows_tr={len(arrays[6])} rows_va={len(va_idx)} '
        f'num={arrays[0].shape[1]} cat={arrays[3].shape[1]} qbin={len(arrays[-1])} cards_first={arrays[9][:12]}'
    )

    va_prob, te_prob, fold_bac, fold_loss, best_epoch = train_fold(fold, arrays, class_weights)
    oof_preds[va_idx] = va_prob
    test_preds += te_prob / N_FOLDS
    oof_fold[va_idx] = fold

    row = {
        'fold': fold,
        'bac': float(fold_bac),
        'log_loss': float(fold_loss),
        'best_epoch': int(best_epoch),
        'seconds': time.perf_counter() - t0,
    }
    fold_rows.append(row)
    print(f'Fold {fold} balanced accuracy: {fold_bac:.8f} | logloss={fold_loss:.6f} | best_epoch={best_epoch}', flush=True)

    del arrays, va_prob, te_prob
    cleanup_cuda()

fold_scores = pd.DataFrame(fold_rows)
overall_score = balanced_accuracy_score(y, oof_preds.argmax(axis=1))
cm = confusion_matrix(y, oof_preds.argmax(axis=1), labels=np.arange(len(CLASSES)))
per_class_recall = recall_score(y, oof_preds.argmax(axis=1), labels=np.arange(len(CLASSES)), average=None)

print('=' * 80)
print(fold_scores)
print(f'Overall OOF balanced accuracy: {overall_score:.8f}')
print(f'NN3-032 local reference:       {SOURCE_EXPECTED_CV:.8f}')
print('=' * 80)
print(pd.DataFrame(cm, index=[f'True {c}' for c in CLASSES], columns=[f'Pred {c}' for c in CLASSES]))

# ----------------------------------------------------------------------------
# Normalize + clip probabilities, then save artifacts
# ----------------------------------------------------------------------------
def normalize_clip(p):
    p = np.clip(p.astype('float32'), 1e-15, None)
    p = p / p.sum(axis=1, keepdims=True)
    return p.astype('float32')


oof_preds = normalize_clip(oof_preds)
test_preds = normalize_clip(test_preds)

np.save(OOF_PATH, oof_preds.astype('float32'))
np.save(PRED_PATH, test_preds.astype('float32'))
np.save(LABEL_PATH, y.astype('int64'))
np.save(FOLD_PATH, oof_fold.astype('int16'))
fold_scores.to_csv(SCORES_PATH, index=False)

submission = sample_submission.copy()
submission[TARGET] = CLASSES[test_preds.argmax(axis=1)]
submission.to_csv(SUB_PATH, index=False)

result = {
    'model_id': MODEL_ID,
    'source_experiment': SOURCE_EXPERIMENT,
    'source_expected_cv': SOURCE_EXPECTED_CV,
    'overall_bac': float(overall_score),
    'mean_fold_bac': float(fold_scores['bac'].mean()),
    'std_fold_bac': float(fold_scores['bac'].std(ddof=0)),
    'per_class_recall': {CLASSES[i]: float(per_class_recall[i]) for i in range(len(CLASSES))},
    'folds': fold_scores.to_dict(orient='records'),
    'confusion_matrix': cm.tolist(),
    'config': {k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v)) for k, v in vars(CFG).items()},
}
RESULT_PATH.write_text(json.dumps(result, indent=2))

# ----------------------------------------------------------------------------
# results.txt (file output is reliably pulled; console log may not be)
# ----------------------------------------------------------------------------
lines = []
lines.append(f'model_id={MODEL_ID} source={SOURCE_EXPERIMENT}')
lines.append(f'oof shape={oof_preds.shape} test shape={test_preds.shape} columns=[GALAXY,QSO,STAR]')
lines.append(f'label mapping: GALAXY=0, QSO=1, STAR=2')
lines.append(f'fold split: StratifiedKFold(n_splits={N_FOLDS}, shuffle=True, random_state={SEED})')
lines.append('')
lines.append('Per-fold balanced accuracy:')
for r in fold_rows:
    lines.append(f"  fold {r['fold']}: BAC={r['bac']:.8f} logloss={r['log_loss']:.6f} "
                 f"best_epoch={r['best_epoch']} seconds={r['seconds']:.1f}")
lines.append('')
lines.append(f'Overall OOF balanced accuracy: {overall_score:.8f}')
lines.append(f'Mean fold BAC: {fold_scores["bac"].mean():.8f}  std: {fold_scores["bac"].std(ddof=0):.8f}')
lines.append(f'NN3-032 local reference:       {SOURCE_EXPECTED_CV:.8f}')
lines.append('')
lines.append('Per-class OOF recall:')
for i in range(len(CLASSES)):
    lines.append(f'  {CLASSES[i]}: {per_class_recall[i]:.8f}')
lines.append('')
lines.append('Confusion matrix (rows=true, cols=pred) order [GALAXY,QSO,STAR]:')
lines.append(str(cm.tolist()))
lines.append('')
lines.append(f'FINAL SUMMARY: nn2 OOF balanced_accuracy={overall_score:.8f} '
             f'(GALAXY/QSO/STAR recall = '
             f'{per_class_recall[0]:.6f}/{per_class_recall[1]:.6f}/{per_class_recall[2]:.6f})')

RESULTS_TXT.write_text('\n'.join(lines) + '\n')

print('Saved:')
for path in [OOF_PATH, PRED_PATH, LABEL_PATH, FOLD_PATH, SCORES_PATH, RESULT_PATH, SUB_PATH, RESULTS_TXT]:
    print(' ', path)

print('\n'.join(lines), flush=True)
