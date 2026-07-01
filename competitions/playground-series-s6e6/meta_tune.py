"""Meta-learner tuning on the 21-model logit stack: does scaling / C / non-linear meta help?
Stacking only (local). Reports OOF balanced accuracy after additive-bias calibration."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from config import ARTIFACTS, CLASS_ORDER, N_SPLITS
from stack import prob_to_logit, search_bias, _logp

MODELS = ("lgb_multi,xgb_multi,cat_multi,hgb_multi,logreg_multi,mlp_multi,specialist,knn_multi,"
          "realmlp,realmlp5,realmlp5b,realmlp5c,nn2,nn2b,tabm,lgb_orig,xgb_orig,cat_orig,"
          "catv3,xgbv5,lgbmv3").split(",")
y = np.load(ARTIFACTS / "y_train.npy")
X = np.concatenate([prob_to_logit(np.load(ARTIFACTS / f"oof_{m}.npy")) for m in MODELS], axis=1)
NC = len(CLASS_ORDER)


def build(kind, scale, C, seed):
    if kind == "logreg":
        est = LogisticRegression(C=C, max_iter=4000, class_weight="balanced", n_jobs=1, random_state=seed)
    elif kind == "mlp":
        est = MLPClassifier(hidden_layer_sizes=(64,), alpha=C, max_iter=400, early_stopping=True,
                            n_iter_no_change=15, random_state=seed)
    return make_pipeline(StandardScaler(), est) if scale else est


def evalcfg(kind, scale, C, seeds=(42,)):
    oof = np.zeros((len(y), NC))
    for s in seeds:
        o = np.zeros((len(y), NC))
        for tr, va in StratifiedKFold(N_SPLITS, shuffle=True, random_state=s).split(X, y):
            m = build(kind, scale, C, s)
            m.fit(X[tr], y[tr]); o[va] = m.predict_proba(X[va])
        oof += o / len(seeds)
    ba0 = balanced_accuracy_score(y, oof.argmax(1))
    b = search_bias(y, oof); ba1 = balanced_accuracy_score(y, (_logp(oof) + b).argmax(1))
    return ba0, ba1


if __name__ == "__main__":
    cfgs = [("logreg", False, 1.0), ("logreg", True, 1.0), ("logreg", True, 3.0),
            ("logreg", True, 0.5), ("logreg", False, 3.0), ("mlp", True, 1e-3)]
    print(f"{'meta':8s} {'scale':5s} {'C':>6s} {'OOF(pre)':>9s} {'OOF(bias)':>10s}")
    best = (None, 0.0)
    for kind, scale, C in cfgs:
        ba0, ba1 = evalcfg(kind, scale, C)
        print(f"{kind:8s} {str(scale):5s} {C:6.3g} {ba0:9.5f} {ba1:10.5f}", flush=True)
        if ba1 > best[1]:
            best = ((kind, scale, C), ba1)
    print(f"\nBEST meta: {best[0]} = {best[1]:.5f}  (baseline logreg/noscale/C1 reference)")
