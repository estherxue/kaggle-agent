"""Train non-GBDT base models (different family) to add ensemble coverage.

The 4 GBDTs (lgb/xgb/cat/hgb) are highly correlated (mean error-Jaccard ~0.72, oracle
ceiling stuck ~0.974 — see analyze_correlation.py). A different model *family* can correct
samples all the trees miss, raising the oracle ceiling the blend draws from.

Writes ``artifacts/oof_<name>.npy`` and ``artifacts/test_<name>.npy`` using the SAME
StratifiedKFold(seed) split as train_oof.py, so rows align with the existing OOF arrays.
Add the name to ``config.MODELS`` and re-run blend.py to fold it into the ensemble.

Usage:
    python train_diverse.py --model logreg
    python train_diverse.py --model mlp
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import ARTIFACTS, CLASS_ORDER, DATA, N_SPLITS, RANDOM_STATE
from features import load_data, prepare_lgb_xgb
from metrics import clip_proba
from train_oof import encode_target

_TREE_MODELS = {"rf", "et"}  # tree-based; don't need scaling but won't be harmed by it


def build(model_name: str, seed: int):
    if model_name == "logreg":
        clf = LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=seed
        )
    elif model_name == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            alpha=1e-4,
            batch_size=4096,
            learning_rate_init=1e-3,
            early_stopping=True,
            n_iter_no_change=10,
            max_iter=100,
            random_state=seed,
        )
    elif model_name == "knn":
        # Distance-weighted KNN: very different decision boundary from all tree models.
        # k=15 balances bias/variance; distance weights reduce quantisation noise.
        clf = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)
    elif model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=seed
        )
    elif model_name == "et":
        # Extra-Trees: more randomised splits than RF → lower correlation with boosted trees.
        clf = ExtraTreesClassifier(
            n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=seed
        )
    else:
        raise ValueError(model_name)
    # StandardScaler is necessary for logreg/mlp/knn; harmless for rf/et.
    return make_pipeline(StandardScaler(), clf)


def run_diverse(
    model_name: str,
    seed: int = RANDOM_STATE,
    feature_set: str = "all_v2",
    out_name: str | None = None,
) -> float:
    """Train and save a diverse base model; return OOF balanced accuracy."""
    if out_name is None:
        out_name = model_name
    train, test, _ = load_data(DATA)
    y, _ = encode_target(train)
    n_classes = len(CLASS_ORDER)

    train_f, test_f, cols = prepare_lgb_xgb(train, test, feature_set=feature_set)
    X = np.nan_to_num(train_f[cols].to_numpy(dtype=np.float64), posinf=0.0, neginf=0.0)
    Xt = np.nan_to_num(test_f[cols].to_numpy(dtype=np.float64), posinf=0.0, neginf=0.0)

    oof = np.zeros((len(y), n_classes))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        model = build(model_name, seed)
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])
        ba = balanced_accuracy_score(y[va], oof[va].argmax(1))
        print(f"  fold {fold}/{N_SPLITS}  BA={ba:.5f}")

    oof = clip_proba(oof)
    oof_ba = balanced_accuracy_score(y, oof.argmax(1))
    recalls = {CLASS_ORDER[c]: float((oof.argmax(1)[y == c] == c).mean()) for c in range(n_classes)}
    print(f"{out_name}: OOF BA={oof_ba:.5f} recalls={ {k: round(v, 4) for k, v in recalls.items()} }")

    print("Refitting on full train for test predictions...")
    full = build(model_name, seed)
    full.fit(X, y)
    test_pred = clip_proba(full.predict_proba(Xt))

    np.save(ARTIFACTS / f"oof_{out_name}.npy", oof)
    np.save(ARTIFACTS / f"test_{out_name}.npy", test_pred)
    print(f"Saved artifacts/oof_{out_name}.npy and artifacts/test_{out_name}.npy")
    return oof_ba


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a non-GBDT base model for the S6E6 stack")
    p.add_argument("--model", choices=["logreg", "mlp", "knn", "rf", "et"], required=True)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--feature-set", default="all_v2")
    p.add_argument("--name", default=None, help="Output artifact name (default: same as --model)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_diverse(args.model, seed=args.seed, feature_set=args.feature_set, out_name=args.name)


if __name__ == "__main__":
    main()
