"""Pick-2 candidate: LightGBM (non-linear) meta-learner over the FULL 48-model broad pool.
Honest nested-CV of this meta on the 48-pool = 0.97040 (vs 24-base GBDT-meta 0.97022) — the broad GM
pool lifts it. Methodologically distinct from the level-2 meta-of-metas (pick-1): a single non-linear
meta. Multi-seed averaged; additive-bias calibrated on inner-CV meta-OOF."""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from config import ARTIFACTS, CLASS_ORDER, N_SPLITS, DATA, ID_COL, TARGET_COL, SUBMISSIONS
from features import load_data
from stack import _logp, search_bias

MODELS = ("lgb_multi,xgb_multi,cat_multi,hgb_multi,logreg_multi,mlp_multi,specialist,knn_multi,realmlp,"
          "realmlp5,realmlp5b,realmlp5c,nn2,nn2b,tabm,lgb_orig,xgb_orig,cat_orig,catv3,xgbv5,lgbmv3,"
          "ovrxgb,ovrcat,ovrcatb,cdtabm1,cdxgb6,cdlgbm5,cdrealmlp0,cdrealmlp2,cdtabm0,tabicl,"
          "cdx_cat0,cdx_cat3,cdx_lgbm3,cdx_logreg1,cdx_nn1,cdx_nn2,cdx_realmlp1,cdx_realmlp5,cdx_tabicl2,"
          "cdx_xgb0,cdx_xgb1,cdx_xgb3,cdx_xgb5,autogluon,massfe_lgb,pseudolgb,pseudoxgb").split(",")
NC = len(CLASS_ORDER)
SEEDS = [42, 43, 44, 45, 46]

y = np.load(ARTIFACTS / "y_train.npy")
Xtr = np.concatenate([_logp(np.load(ARTIFACTS / f"oof_{m}.npy")) for m in MODELS], 1).astype(np.float32)
Xte = np.concatenate([_logp(np.load(ARTIFACTS / f"test_{m}.npy")) for m in MODELS], 1).astype(np.float32)
print(f"GBDT-meta pool48: {len(MODELS)} models, {Xtr.shape[1]} feats, train {Xtr.shape}, test {Xte.shape}")

PARAMS = dict(objective="multiclass", num_class=NC, n_estimators=400, learning_rate=0.03, num_leaves=8,
              min_child_samples=1000, subsample=0.6, subsample_freq=1, colsample_bytree=0.5,
              reg_lambda=20.0, reg_alpha=1.0, n_jobs=-1, verbose=-1)

meta_oof = np.zeros((len(y), NC))
test_proba = np.zeros((len(Xte), NC))
for s in SEEDS:
    oof_s = np.zeros((len(y), NC))
    for tr, va in StratifiedKFold(N_SPLITS, shuffle=True, random_state=s).split(Xtr, y):
        m = lgb.LGBMClassifier(random_state=s, **PARAMS)
        m.fit(Xtr[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        oof_s[va] = m.predict_proba(Xtr[va])
    meta_oof += oof_s / len(SEEDS)
    mf = lgb.LGBMClassifier(random_state=s, **PARAMS)
    mf.fit(Xtr, y, sample_weight=compute_sample_weight("balanced", y))
    test_proba += mf.predict_proba(Xte) / len(SEEDS)
    print(f"  seed {s} done", flush=True)

ba0 = balanced_accuracy_score(y, meta_oof.argmax(1))
b = search_bias(y, meta_oof)
pred_oof = (_logp(meta_oof) + b).argmax(1)
ba1 = balanced_accuracy_score(y, pred_oof)
rec = {CLASS_ORDER[c]: round(float((pred_oof[y == c] == c).mean()), 4) for c in range(NC)}
print(f"GBDT-meta pool48 OOF BA before calib: {ba0:.5f}")
print(f"GBDT-meta pool48 OOF BA after  calib: {ba1:.5f}  recalls={rec}")

_, test, _ = load_data(DATA)
pred = (_logp(test_proba) + b).argmax(1)
out = SUBMISSIONS / "stack_gbdtmeta_pool48.csv"
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: [CLASS_ORDER[i] for i in pred]}).to_csv(out, index=False)
print(f"Saved {out}")
