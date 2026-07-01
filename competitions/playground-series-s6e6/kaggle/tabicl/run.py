"""TabICL (in-context-learning transformer) for S6E6 — a genuinely different inductive bias.
Port of cdeotte/tabicl-v2: raw features, balanced 30k context/fold, 5-fold seed-42 (aligns with all
artifacts). P100-safe torch (cu121). Saves oof_tabicl.npy / test_tabicl.npy for stacking."""
import os, gc, glob, time, sys, subprocess
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('WANDB_DISABLED', 'true')

# Install tabicl (+ its deps), THEN force a Pascal-compatible cu121 torch (stock 2.10+cu128 dropped sm_60,
# and this kernel may land on a P100). tabicl uses standard torch ops, so 2.4.1 is fine.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tabicl'], check=False)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch==2.4.1',
                '--extra-index-url', 'https://download.pytorch.org/whl/cu121'], check=False)

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from tabicl import TabICLClassifier

print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available(),
      '| dev:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', flush=True)

SEED, N_FOLDS, N_CLASSES = 42, 5, 3
CLASSES = ['GALAXY', 'QSO', 'STAR']
TMAP = {c: i for i, c in enumerate(CLASSES)}
SAMPLES_PER_CLASS = 10000
WORK = '/kaggle/working'


def find_root():
    for r in ['/kaggle/input/competitions/playground-series-s6e6',
              '/kaggle/input/playground-series-s6e6']:
        if os.path.exists(os.path.join(r, 'train.csv')):
            return r
    for p in glob.glob('/kaggle/input/**/train.csv', recursive=True):
        return os.path.dirname(p)
    raise FileNotFoundError('no train.csv')


ROOT = find_root()
train = pd.read_csv(os.path.join(ROOT, 'train.csv'))
test = pd.read_csv(os.path.join(ROOT, 'test.csv'))
sample = pd.read_csv(os.path.join(ROOT, 'sample_submission.csv'))
# align test to sample_submission id order
test = test.set_index('id').loc[sample['id']].reset_index()

y = train['class'].map(TMAP).astype('int64').to_numpy()
assert not np.any(np.isnan(y.astype(float)))

FEATURES = [c for c in train.columns if c not in ('id', 'class')]
# Label-encode the two binned-color categoricals to int codes (safe for TabICL); numerics stay float32.
Xtr = train[FEATURES].copy()
Xte = test[FEATURES].copy()
for c in FEATURES:
    if Xtr[c].dtype == object:
        le = LabelEncoder()
        le.fit(pd.concat([Xtr[c], Xte[c]], axis=0).astype(str))
        Xtr[c] = le.transform(Xtr[c].astype(str))
        Xte[c] = le.transform(Xte[c].astype(str))
    else:
        Xtr[c] = Xtr[c].astype('float32'); Xte[c] = Xte[c].astype('float32')
print(f'train {Xtr.shape} test {Xte.shape} feats={FEATURES}', flush=True)

# device MUST be indexed ('cuda:0', not 'cuda') — torch 2.4.1's mem_get_info rejects a bare 'cuda'.
PARAMS = dict(n_estimators=8, batch_size=1, device='cuda:0' if torch.cuda.is_available() else 'cpu',
              use_amp='auto', random_state=SEED, verbose=False)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros((len(y), N_CLASSES), dtype='float32')
test_sum = np.zeros((len(Xte), N_CLASSES), dtype='float32')
fold_ba = []
for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y), start=1):
    t0 = time.time()
    ctx = []
    for c in range(N_CLASSES):
        ci = tr_idx[y[tr_idx] == c]
        np.random.seed(SEED + fold + c)
        ctx.extend(np.random.choice(ci, size=min(len(ci), SAMPLES_PER_CLASS), replace=False))
    Xc, yc = Xtr.iloc[ctx], y[ctx]
    clf = TabICLClassifier(**PARAMS)
    clf.fit(Xc, yc)
    oof[va_idx] = clf.predict_proba(Xtr.iloc[va_idx]).astype('float32')
    test_sum += clf.predict_proba(Xte).astype('float32') / N_FOLDS
    ba = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
    fold_ba.append(ba)
    print(f'Fold {fold} BA={ba:.6f} ctx={len(ctx)} {time.time()-t0:.0f}s', flush=True)
    del clf; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

oof = oof / np.clip(oof.sum(1, keepdims=True), 1e-9, None)
test_sum = test_sum / np.clip(test_sum.sum(1, keepdims=True), 1e-9, None)
overall = balanced_accuracy_score(y, oof.argmax(1))
rec = {CLASSES[c]: float((oof.argmax(1)[y == c] == c).mean()) for c in range(N_CLASSES)}
np.save(os.path.join(WORK, 'oof_tabicl.npy'), oof.astype('float32'))
np.save(os.path.join(WORK, 'test_tabicl.npy'), test_sum.astype('float32'))
sub = sample.copy(); sub['class'] = [CLASSES[i] for i in test_sum.argmax(1)]
sub.to_csv(os.path.join(WORK, 'submission.csv'), index=False)
with open(os.path.join(WORK, 'results.txt'), 'w') as f:
    f.write(f'tabicl OOF BA={overall:.6f}\nper-fold={fold_ba}\nrecalls={rec}\n'
            f'oof{oof.shape} test{test_sum.shape}\n')
print(f'TabICL OOF BA={overall:.6f} recalls={rec}', flush=True)
print('saved oof_tabicl.npy / test_tabicl.npy', flush=True)
