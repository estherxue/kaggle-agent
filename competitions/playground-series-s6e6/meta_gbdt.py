"""Non-linear (LightGBM) meta-learner over the 24 base models' OOF log-probs.
The linear meta saturates at ~0.9704 while the oracle ceiling is ~0.974+ — a GBDT meta can model
cross-model interactions a linear meta can't. Honest: inner-CV OOF + nested-CV check (no LB/test peeking).
Reports OOF BA (bias-calibrated) for several regularizations, and a nested-CV estimate for the best."""
from __future__ import annotations
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from config import ARTIFACTS, CLASS_ORDER, N_SPLITS
from stack import _logp, search_bias

MODELS = ("lgb_multi,xgb_multi,cat_multi,hgb_multi,logreg_multi,mlp_multi,specialist,knn_multi,realmlp,"
          "realmlp5,realmlp5b,realmlp5c,nn2,nn2b,tabm,lgb_orig,xgb_orig,cat_orig,catv3,xgbv5,lgbmv3,"
          "ovrxgb,ovrcat,ovrcatb").split(",")
y = np.load(ARTIFACTS / "y_train.npy")
X = np.concatenate([_logp(np.load(ARTIFACTS / f"oof_{m}.npy")) for m in MODELS], axis=1).astype(np.float32)
NC = len(CLASS_ORDER)
print(f"GBDT-meta on {len(MODELS)} models -> {X.shape[1]} features")


def gbdt_params(cfg):
    return dict(objective="multiclass", num_class=NC, n_estimators=cfg["n_est"], learning_rate=cfg["lr"],
                num_leaves=cfg["leaves"], min_child_samples=cfg["mcs"], subsample=cfg["ss"],
                subsample_freq=1, colsample_bytree=cfg["cs"], reg_lambda=cfg["l2"], reg_alpha=cfg.get("l1", 0.0),
                random_state=42, n_jobs=-1, verbose=-1)


def meta_oof(cfg, seed=42):
    oof = np.zeros((len(y), NC))
    for tr, va in StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(X, y):
        sw = compute_sample_weight("balanced", y[tr])
        m = lgb.LGBMClassifier(**gbdt_params(cfg))
        m.fit(X[tr], y[tr], sample_weight=sw)
        oof[va] = m.predict_proba(X[va])
    return oof


def score(oof):
    ba0 = balanced_accuracy_score(y, oof.argmax(1))
    b = search_bias(y, oof)
    ba1 = balanced_accuracy_score(y, (_logp(oof) + b).argmax(1))
    return ba0, ba1


def nested(cfg, seed=42):
    """Honest: outer fold's meta trained only on outer-train; bias fit on inner-OOF of outer-train."""
    pred = np.zeros(len(y), dtype=int)
    for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(X, y):
        # inner OOF on outer-train to fit bias
        inner_oof = np.zeros((len(tr), NC))
        Xtr, ytr = X[tr], y[tr]
        for itr, iva in StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed + 1).split(Xtr, ytr):
            sw = compute_sample_weight("balanced", ytr[itr])
            m = lgb.LGBMClassifier(**gbdt_params(cfg)); m.fit(Xtr[itr], ytr[itr], sample_weight=sw)
            inner_oof[iva] = m.predict_proba(Xtr[iva])
        b = search_bias(ytr, inner_oof)
        sw = compute_sample_weight("balanced", ytr)
        mm = lgb.LGBMClassifier(**gbdt_params(cfg)); mm.fit(Xtr, ytr, sample_weight=sw)
        pred[te] = (_logp(mm.predict_proba(X[te])) + b).argmax(1)
    return balanced_accuracy_score(y, pred)


CFGS = {
    "strongreg": dict(n_est=400, lr=0.03, leaves=8, mcs=1000, ss=0.6, cs=0.5, l2=20.0, l1=1.0),
    "medreg":    dict(n_est=500, lr=0.02, leaves=15, mcs=400, ss=0.7, cs=0.5, l2=8.0),
    "lightreg":  dict(n_est=700, lr=0.02, leaves=31, mcs=150, ss=0.8, cs=0.6, l2=3.0),
}

if __name__ == "__main__":
    print(f"{'cfg':10s} {'OOF(pre)':>9s} {'OOF(bias)':>10s}")
    best = (None, 0.0, None)
    for name, cfg in CFGS.items():
        oof = meta_oof(cfg)
        ba0, ba1 = score(oof)
        print(f"{name:10s} {ba0:9.5f} {ba1:10.5f}", flush=True)
        if ba1 > best[1]:
            best = (name, ba1, cfg)
    print(f"\nlinear-meta reference: OOF(bias) 0.97040, nested-CV 0.96954")
    print(f"best GBDT-meta: {best[0]} OOF(bias)={best[1]:.5f}")
    ncv = nested(best[2])
    print(f"nested-CV (honest) of best GBDT-meta = {ncv:.5f}  (linear nested-CV was 0.96954)")
