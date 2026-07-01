"""GALAXY-vs-STAR specialist: LightGBM with heavily upweighted GALAXY/STAR.

The dominant stack error is GALAXY→STAR confusion (~14k misclassifications).
This specialist downweights QSO in the sample weights so the model focuses its
capacity on the GALAXY/STAR decision boundary instead of the easy QSO separation.
Used as an additional base model for the stacking meta-learner alongside the 4 GBDTs.

Saves oof_<name>.npy and test_<name>.npy.

Usage:
    python specialist.py [--seed 42] [--feature-set all_v2] [--name specialist]
"""

from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from config import ARTIFACTS, CLASS_ORDER, DATA, EARLY_STOPPING_ROUNDS, N_SPLITS, RANDOM_STATE
from features import load_data, prepare_lgb_xgb
from metrics import clip_proba, evaluate_proba
from train_oof import TrainConfig, build_lgbm, encode_target

# GALAXY=0, QSO=1, STAR=2  (CLASS_ORDER order)
# Upweight the confused pair, shrink QSO so the model stops wasting capacity there.
_SPECIALIST_FACTOR = {0: 2.0, 1: 0.25, 2: 2.0}


def _specialist_weights(y: np.ndarray) -> np.ndarray:
    balanced = compute_sample_weight("balanced", y)
    factor = np.array([_SPECIALIST_FACTOR[c] for c in y])
    return balanced * factor


def run_specialist(
    seed: int = RANDOM_STATE,
    feature_set: str = "all_v2",
    out_name: str = "specialist",
) -> float:
    train, test, _ = load_data(DATA)
    y, _ = encode_target(train)
    n_classes = len(CLASS_ORDER)

    cfg = TrainConfig(feature_set=feature_set, seed=seed)
    train_f, test_f, cols = prepare_lgb_xgb(train, test, feature_set=feature_set)
    X = train_f[cols].values
    Xt = test_f[cols].values

    oof = np.zeros((len(y), n_classes))
    best_iters: list[int] = []

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        y_tr, y_va = y[tr_idx], y[va_idx]
        sw = _specialist_weights(y_tr)

        model = build_lgbm(cfg, n_classes, cfg.n_estimators)
        model.fit(
            X[tr_idx],
            y_tr,
            sample_weight=sw,
            eval_set=[(X[va_idx], y_va)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(0)],
        )
        best = model.best_iteration_ or cfg.n_estimators
        best_iters.append(int(best))
        oof[va_idx] = clip_proba(model.predict_proba(X[va_idx]))
        r = evaluate_proba(y_va, oof[va_idx])
        print(
            f"  fold {fold}/{N_SPLITS}  BA={r.oof_balanced_accuracy:.5f}"
            f"  GALAXY={r.per_class_recall['GALAXY']:.4f}"
            f"  STAR={r.per_class_recall['STAR']:.4f}"
            f"  iters={best}"
        )

    oof = clip_proba(oof)
    oof_ba = balanced_accuracy_score(y, oof.argmax(1))
    recalls = {
        CLASS_ORDER[c]: round(float((oof.argmax(1)[y == c] == c).mean()), 4)
        for c in range(n_classes)
    }
    print(f"\n{out_name}: OOF BA={oof_ba:.5f} recalls={recalls}")

    n_est = min(int(np.mean(best_iters) * 1.1), cfg.n_estimators)
    full = build_lgbm(cfg, n_classes, n_est)
    full.fit(X, y, sample_weight=_specialist_weights(y))
    test_pred = clip_proba(full.predict_proba(Xt))

    np.save(ARTIFACTS / f"oof_{out_name}.npy", oof)
    np.save(ARTIFACTS / f"test_{out_name}.npy", test_pred)
    print(f"Saved oof_{out_name}.npy and test_{out_name}.npy (refit iters={n_est})")
    return oof_ba


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GALAXY-vs-STAR specialist LGBM for S6E6 stack")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--feature-set", default="all_v2")
    p.add_argument("--name", default="specialist")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_specialist(seed=args.seed, feature_set=args.feature_set, out_name=args.name)


if __name__ == "__main__":
    main()
