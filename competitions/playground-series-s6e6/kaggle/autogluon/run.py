"""AutoGluon Tabular for S6E6 — a strong, decorrelated AutoML ensemble as a new stacking base.
Bagged (num_bag_folds=5) so predict_proba_oof() gives a valid out-of-fold prediction for stacking
(its internal folds need NOT match seed-42 — OOF is still out-of-fold, which is all stacking requires).
eval_metric=balanced_accuracy (the comp metric). CPU (avoids the P100 torch/cuDF walls; AutoGluon runs
its NN/GBDT zoo fine on CPU). Saves oof_autogluon.npy / test_autogluon.npy in [GALAXY,QSO,STAR] order."""
import os, sys, glob, time, subprocess
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'autogluon.tabular[all]'], check=False)

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from autogluon.tabular import TabularPredictor

LABELS = ['GALAXY', 'QSO', 'STAR']
LMAP = {c: i for i, c in enumerate(LABELS)}
WORK = '/kaggle/working'
T0 = time.time()


def find_root():
    for r in ['/kaggle/input/competitions/playground-series-s6e6', '/kaggle/input/playground-series-s6e6']:
        if os.path.exists(os.path.join(r, 'train.csv')):
            return r
    for p in glob.glob('/kaggle/input/**/train.csv', recursive=True):
        return os.path.dirname(p)
    raise FileNotFoundError('train.csv not found')


ROOT = find_root()
train = pd.read_csv(os.path.join(ROOT, 'train.csv'))
test = pd.read_csv(os.path.join(ROOT, 'test.csv'))
sample = pd.read_csv(os.path.join(ROOT, 'sample_submission.csv'))
test = test.set_index('id').loc[sample['id']].reset_index()   # align to sample_submission order
assert len(train) == 577347 and len(test) == 247435, (len(train), len(test))

y = train['class'].map(LMAP).to_numpy()
train_data = train.drop(columns=['id'])            # keep raw features + 'class' label (AutoGluon handles cats)
test_data = test.drop(columns=['id'])
print('train', train_data.shape, 'test', test_data.shape, flush=True)

predictor = TabularPredictor(
    label='class', eval_metric='balanced_accuracy', path=os.path.join(WORK, 'ag_model'), verbosity=2,
).fit(
    train_data,
    presets='good_quality',
    num_bag_folds=5,        # bagging -> valid OOF for stacking
    num_stack_levels=0,     # keep it 1-level (clean OOF, bounded time)
    time_limit=18000,       # 5h cap (safe under the 12h CPU wall; + install/predict overhead)
)

# OOF (out-of-fold) predictions aligned to train_data row order (= CSV order = y order)
oof_df = predictor.predict_proba_oof()
oof = oof_df[LABELS].to_numpy().astype('float32')
test_proba = predictor.predict_proba(test_data)[LABELS].to_numpy().astype('float32')
oof = oof / np.clip(oof.sum(1, keepdims=True), 1e-9, None)
test_proba = test_proba / np.clip(test_proba.sum(1, keepdims=True), 1e-9, None)
assert oof.shape == (577347, 3) and test_proba.shape == (247435, 3), (oof.shape, test_proba.shape)

np.save(os.path.join(WORK, 'oof_autogluon.npy'), oof)
np.save(os.path.join(WORK, 'test_autogluon.npy'), test_proba)
ba = balanced_accuracy_score(y, oof.argmax(1))
rec = {LABELS[c]: round(float((oof.argmax(1)[y == c] == c).mean()), 4) for c in range(3)}
sub = sample.copy(); sub['class'] = [LABELS[i] for i in test_proba.argmax(1)]
sub.to_csv(os.path.join(WORK, 'submission.csv'), index=False)
with open(os.path.join(WORK, 'results.txt'), 'w') as f:
    f.write(f'autogluon OOF balanced_accuracy={ba:.6f}\nrecalls={rec}\n'
            f'oof{oof.shape} test{test_proba.shape}  elapsed={time.time()-T0:.0f}s\n')
    try:
        f.write('\nleaderboard:\n' + predictor.leaderboard(silent=True).to_string())
    except Exception as e:
        f.write(f'\n(leaderboard unavailable: {e!r})')
print(f'AutoGluon OOF BA={ba:.5f} recalls={rec}', flush=True)
print('saved oof_autogluon.npy / test_autogluon.npy', flush=True)
