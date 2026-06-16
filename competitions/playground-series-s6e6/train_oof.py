"""5-fold OOF training for LGBM, XGBoost, CatBoost."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from config import (
    ARTIFACTS,
    CLASS_ORDER,
    DATA,
    N_ESTIMATORS,
    N_SPLITS,
    RANDOM_STATE,
    TARGET_COL,
)
from experiment_log import log_experiment
from features import FEATURE_SETS, load_data, prepare_catboost, prepare_lgb_xgb
from metrics import clip_proba, evaluate_proba

ALL_MODELS = ("lgb", "xgb", "cat")


@dataclass
class TrainConfig:
    feature_set: str = "all"
    models: tuple[str, ...] = ALL_MODELS
    seed: int = RANDOM_STATE
    use_coords: bool = True
    cat_native: bool = False
    cat_iterations: int = N_ESTIMATORS
    cat_depth: int = 8
    cat_learning_rate: float = 0.05
    n_estimators: int = N_ESTIMATORS
    learning_rate: float = 0.05


def encode_target(train: pd.DataFrame) -> tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    y = le.transform(train[TARGET_COL].astype(str))
    return y, le


def train_lgbm(X_tr, y_tr, X_va, n_classes: int, sample_weight, cfg: TrainConfig):
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=cfg.seed,
        verbose=-1,
    )
    model.fit(X_tr, y_tr, sample_weight=sample_weight)
    return clip_proba(model.predict_proba(X_va))


def train_xgb(X_tr, y_tr, X_va, n_classes: int, sample_weight, cfg: TrainConfig):
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=cfg.seed,
        verbosity=0,
    )
    model.fit(X_tr, y_tr, sample_weight=sample_weight)
    return clip_proba(model.predict_proba(X_va))


def train_cat(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    y_tr,
    y_va,
    va_idx,
    tr_idx,
    n_classes: int,
    cfg: TrainConfig,
):
    from catboost import CatBoostClassifier, Pool

    cat_idx = [feature_cols.index(c) for c in cat_cols] if cat_cols else []
    sw = compute_sample_weight(class_weight="balanced", y=y_tr)

    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=cfg.cat_iterations,
        learning_rate=cfg.cat_learning_rate,
        depth=cfg.cat_depth,
        random_seed=cfg.seed,
        verbose=0,
        auto_class_weights="Balanced",
    )
    train_pool = Pool(
        train_df.iloc[tr_idx][feature_cols],
        y_tr,
        cat_features=cat_idx or None,
        weight=sw,
    )
    model.fit(train_pool)
    va_proba = clip_proba(model.predict_proba(train_df.iloc[va_idx][feature_cols]))
    return model, va_proba


def _load_previous_oof(name: str, n_samples: int, n_classes: int) -> np.ndarray | None:
    path = ARTIFACTS / f"oof_{name}.npy"
    if path.exists():
        arr = np.load(path)
        if arr.shape == (n_samples, n_classes):
            return arr
    return None


def run_oof(cfg: TrainConfig) -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    train, test, _ = load_data(DATA)
    y, _ = encode_target(train)
    n_classes = len(CLASS_ORDER)
    models = set(cfg.models)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=cfg.seed)

    oof_lgb = np.zeros((len(y), n_classes))
    oof_xgb = np.zeros((len(y), n_classes))
    oof_cat = np.zeros((len(y), n_classes))

    train_lx, test_lx, lx_cols = prepare_lgb_xgb(
        train, test, feature_set=cfg.feature_set, use_coords=cfg.use_coords
    )
    X_lx = train_lx[lx_cols].values
    X_test_lx = test_lx[lx_cols].values

    train_cb, test_cb, cb_cols, cb_cat_names = prepare_catboost(
        train,
        test,
        feature_set=cfg.feature_set,
        use_coords=cfg.use_coords,
        cat_native=cfg.cat_native,
    )

    test_lgb = np.zeros((len(test), n_classes))
    test_xgb = np.zeros((len(test), n_classes))
    test_cat = np.zeros((len(test), n_classes))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_lx, y), start=1):
        print(f"\n--- Fold {fold}/{N_SPLITS} (seed={cfg.seed}, feature_set={cfg.feature_set}) ---")
        y_tr, y_va = y[tr_idx], y[va_idx]
        sw = compute_sample_weight(class_weight="balanced", y=y_tr)

        if "lgb" in models:
            p_lgb = train_lgbm(X_lx[tr_idx], y_tr, X_lx[va_idx], n_classes, sw, cfg)
            oof_lgb[va_idx] = p_lgb
            r_lgb = evaluate_proba(y_va, p_lgb)
            print(f"LGB  BA={r_lgb.oof_balanced_accuracy:.5f} STAR_r={r_lgb.per_class_recall['STAR']:.4f}")

        if "xgb" in models:
            p_xgb = train_xgb(X_lx[tr_idx], y_tr, X_lx[va_idx], n_classes, sw, cfg)
            oof_xgb[va_idx] = p_xgb
            r_xgb = evaluate_proba(y_va, p_xgb)
            print(f"XGB  BA={r_xgb.oof_balanced_accuracy:.5f} STAR_r={r_xgb.per_class_recall['STAR']:.4f}")

        if "cat" in models:
            _, p_cat = train_cat(
                train_cb, cb_cols, cb_cat_names, y_tr, y_va, va_idx, tr_idx, n_classes, cfg
            )
            oof_cat[va_idx] = p_cat
            r_cat = evaluate_proba(y_va, p_cat)
            print(f"CAT  BA={r_cat.oof_balanced_accuracy:.5f} STAR_r={r_cat.per_class_recall['STAR']:.4f}")

    if "lgb" not in models:
        prev = _load_previous_oof("lgb", len(y), n_classes)
        if prev is not None:
            oof_lgb = prev
    if "xgb" not in models:
        prev = _load_previous_oof("xgb", len(y), n_classes)
        if prev is not None:
            oof_xgb = prev
    if "cat" not in models:
        prev = _load_previous_oof("cat", len(y), n_classes)
        if prev is not None:
            oof_cat = prev

    print("\n--- Full fit for test ---")
    sw_full = compute_sample_weight(class_weight="balanced", y=y)

    if "lgb" in models:
        lgb_full = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=cfg.seed,
            verbose=-1,
        )
        lgb_full.fit(X_lx, y, sample_weight=sw_full)
        test_lgb = clip_proba(lgb_full.predict_proba(X_test_lx))
    else:
        prev = ARTIFACTS / "test_lgb.npy"
        if prev.exists():
            test_lgb = np.load(prev)

    if "xgb" in models:
        xgb_full = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=cfg.seed,
            verbosity=0,
        )
        xgb_full.fit(X_lx, y, sample_weight=sw_full)
        test_xgb = clip_proba(xgb_full.predict_proba(X_test_lx))
    else:
        prev = ARTIFACTS / "test_xgb.npy"
        if prev.exists():
            test_xgb = np.load(prev)

    if "cat" in models:
        from catboost import CatBoostClassifier, Pool

        cat_idx = [cb_cols.index(c) for c in cb_cat_names] if cb_cat_names else []
        cat_full = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=cfg.cat_iterations,
            learning_rate=cfg.cat_learning_rate,
            depth=cfg.cat_depth,
            random_seed=cfg.seed,
            verbose=0,
            auto_class_weights="Balanced",
        )
        cat_full.fit(
            Pool(train_cb[cb_cols], y, cat_features=cat_idx or None, weight=sw_full)
        )
        test_cat = clip_proba(cat_full.predict_proba(test_cb[cb_cols]))
    else:
        prev = ARTIFACTS / "test_cat.npy"
        if prev.exists():
            test_cat = np.load(prev)

    np.save(ARTIFACTS / "oof_lgb.npy", oof_lgb)
    np.save(ARTIFACTS / "oof_xgb.npy", oof_xgb)
    np.save(ARTIFACTS / "oof_cat.npy", oof_cat)
    np.save(ARTIFACTS / "test_lgb.npy", test_lgb)
    np.save(ARTIFACTS / "test_xgb.npy", test_xgb)
    np.save(ARTIFACTS / "test_cat.npy", test_cat)
    np.save(ARTIFACTS / "y_train.npy", y)
    (ARTIFACTS / "classes.json").write_text(json.dumps(CLASS_ORDER))

    print("\n=== Single-model OOF (argmax) ===")
    model_scores: dict[str, float] = {}
    for name, oof in [("lgb", oof_lgb), ("xgb", oof_xgb), ("cat", oof_cat)]:
        r = evaluate_proba(y, oof)
        model_scores[name] = r.oof_balanced_accuracy
        print(f"{name}: BA={r.oof_balanced_accuracy:.5f} recalls={r.per_class_recall}")

    meta = {
        "feature_set": cfg.feature_set,
        "seed": cfg.seed,
        "models": list(cfg.models),
        "use_coords": cfg.use_coords,
        "cat_native": cfg.cat_native,
        "n_features_lgb": len(lx_cols),
        "n_features_cat": len(cb_cols),
        "model_oof_ba": model_scores,
    }
    (ARTIFACTS / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nArtifacts saved to {ARTIFACTS}")
    return meta


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="5-fold OOF training for S6E6")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="all")
    parser.add_argument("--models", default="lgb,xgb,cat", help="Comma-separated: lgb,xgb,cat")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--no-coords", action="store_true")
    parser.add_argument("--model", choices=ALL_MODELS, help="Train only this model (alias)")
    parser.add_argument("--cat-native", action="store_true")
    parser.add_argument("--iterations", type=int, default=N_ESTIMATORS)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    parser.add_argument("--tune", choices=["balanced_accuracy"], help="Reserved for hyperparam search")
    args = parser.parse_args()

    if args.model:
        models = (args.model,)
    else:
        models = tuple(m.strip() for m in args.models.split(",") if m.strip())
        for m in models:
            if m not in ALL_MODELS:
                parser.error(f"Unknown model {m!r}")

    return TrainConfig(
        feature_set=args.feature_set,
        models=models,
        seed=args.seed,
        use_coords=not args.no_coords,
        cat_native=args.cat_native,
        cat_iterations=args.iterations,
        cat_depth=args.depth,
        cat_learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
    )


def main() -> None:
    cfg = parse_args()
    meta = run_oof(cfg)
    log_experiment(
        {
            "experiment": "train_oof",
            "seed": cfg.seed,
            "feature_set": cfg.feature_set,
            "models": ",".join(cfg.models),
            "lgb_oof_ba": meta["model_oof_ba"].get("lgb"),
            "xgb_oof_ba": meta["model_oof_ba"].get("xgb"),
            "cat_oof_ba": meta["model_oof_ba"].get("cat"),
            "notes": json.dumps({"n_features_lgb": meta["n_features_lgb"], "cat_native": cfg.cat_native}),
        }
    )


if __name__ == "__main__":
    main()
