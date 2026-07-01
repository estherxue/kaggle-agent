"""Final model-subset selection for the honest 0.97 push.

Loads base-model OOF probabilities, builds logit features, and evaluates several
candidate subsets with a class-weighted multinomial-LR meta-learner (inner 5-fold,
optional multi-seed). Reports OOF balanced accuracy before/after additive-bias calib.
Stacking only (pure numpy/sklearn) — fine to run locally.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from config import ARTIFACTS, CLASS_ORDER, N_SPLITS
from stack import _logp, prob_to_logit, search_bias

y = np.load(ARTIFACTS / "y_train.npy")
NC = len(CLASS_ORDER)


def present(models):
    return [m for m in models if (ARTIFACTS / f"oof_{m}.npy").exists()]


def feats(models):
    return np.concatenate([prob_to_logit(np.load(ARTIFACTS / f"oof_{m}.npy")) for m in models], axis=1)


def eval_set(models, seeds=(42,), max_iter=1000):
    X = feats(models)
    oof = np.zeros((len(y), NC))
    for s in seeds:
        o = np.zeros((len(y), NC))
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=s)
        for tr, va in skf.split(X, y):
            m = LogisticRegression(C=1.0, max_iter=max_iter, class_weight="balanced",
                                   n_jobs=1, random_state=s)
            m.fit(X[tr], y[tr])
            o[va] = m.predict_proba(X[va])
        oof += o / len(seeds)
    ba0 = balanced_accuracy_score(y, oof.argmax(1))
    b = search_bias(y, oof)
    pred = (_logp(oof) + b).argmax(1)
    ba1 = balanced_accuracy_score(y, pred)
    rec = {c: round(float((pred[y == i] == i).mean()), 4) for i, c in enumerate(CLASS_ORDER)}
    return ba0, ba1, rec


REALMLP = ["realmlp5", "realmlp5b", "realmlp5c"]   # seed ensemble of the strongest model
DEEP = ["nn2", "nn2b", "tabm"]
ORIG = ["lgb_orig", "xgb_orig", "cat_orig"]
STRONG_GBDT = ["catv3", "xgbv5", "lgbmv3"]          # cdeotte single-model ports
OLD_GBDT = ["lgb_multi", "xgb_multi", "cat_multi", "hgb_multi"]
OLD_DIV = ["logreg_multi", "mlp_multi", "specialist", "knn_multi"]

CANDS = {
    "B_prev_best(14)": OLD_GBDT + OLD_DIV + ["realmlp5"] + DEEP + ORIG,
    "G_all_in": OLD_GBDT + OLD_DIV + REALMLP + DEEP + ORIG + STRONG_GBDT,
    "H_drop_oldgbdt": OLD_DIV + REALMLP + DEEP + ORIG + STRONG_GBDT,
    "I_drop_origGBDT(use strong)": OLD_GBDT + OLD_DIV + REALMLP + DEEP + STRONG_GBDT,
    "J_strong_core": OLD_DIV + REALMLP + DEEP + STRONG_GBDT,
    "K_no_old_at_all": REALMLP + DEEP + ORIG + STRONG_GBDT,
    "L_everything+oldrealmlp": OLD_GBDT + OLD_DIV + ["realmlp"] + REALMLP + DEEP + ORIG + STRONG_GBDT,
}

if __name__ == "__main__":
    print(f"{'set':30s} {'n':>3s} {'OOF(pre)':>9s} {'OOF(bias)':>10s}  recalls")
    results = {}
    for name, ms in CANDS.items():
        ms = present(ms)
        if not ms:
            continue
        ba0, ba1, rec = eval_set(ms)
        results[name] = (ba1, ms)
        print(f"{name:30s} {len(ms):3d} {ba0:9.5f} {ba1:10.5f}  {rec}", flush=True)
    best = max(results, key=lambda k: results[k][0])
    print(f"\nBEST by OOF(bias): {best} = {results[best][0]:.5f}")
    print("models:", ",".join(results[best][1]))
