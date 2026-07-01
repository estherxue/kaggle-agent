"""5-fold OOF training for LGBM, XGBoost, CatBoost, HistGradientBoosting.

Each fold uses early stopping on its validation fold; the full-data refit (used
for test predictions) reuses the mean best-iteration across folds since it has no
held-out set. Tuned hyper-parameters, if present in
``artifacts/best_params_<model>.json``, are layered on top of the defaults.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from config import (
    ARTIFACTS,
    CLASS_ORDER,
    DATA,
    EARLY_STOPPING_ROUNDS,
    MODELS,
    N_ESTIMATORS,
    N_SPLITS,
    RANDOM_STATE,
    TARGET_COL,
)
from experiment_log import log_experiment
from features import FEATURE_SETS, load_data, prepare_catboost, prepare_lgb_xgb
from metrics import clip_proba, evaluate_proba

ALL_MODELS = tuple(MODELS)


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


def _tuned_params(model: str) -> dict:
    """Load tuned hyper-parameters for a model if tune.py has produced them."""
    path = ARTIFACTS / f"best_params_{model}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError):
            return {}
    return {}


def encode_target(train: pd.DataFrame) -> tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    y = le.transform(train[TARGET_COL].astype(str))
    return y, le


# --- model builders (shared between fold training and full-data refit) ---


def build_lgbm(cfg: TrainConfig, n_classes: int, n_estimators: int) -> lgb.LGBMClassifier:
    params = dict(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=cfg.seed,
        verbose=-1,
    )
    params.update(_tuned_params("lgb"))
    params["n_estimators"] = n_estimators  # cap always wins; early stopping decides
    return lgb.LGBMClassifier(**params)


def build_xgb(
    cfg: TrainConfig, n_classes: int, n_estimators: int, early_stop: bool = False
) -> xgb.XGBClassifier:
    params = dict(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=cfg.seed,
        verbosity=0,
        eval_metric="mlogloss",
    )
    params.update(_tuned_params("xgb"))
    params["n_estimators"] = n_estimators
    if early_stop:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    return xgb.XGBClassifier(**params)


def build_hgb(
    cfg: TrainConfig, max_iter: int, early_stop: bool = True
) -> HistGradientBoostingClassifier:
    params = dict(
        learning_rate=cfg.learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=63,
        l2_regularization=0.0,
        random_state=cfg.seed,
    )
    params.update(_tuned_params("hgb"))
    params["max_iter"] = max_iter
    # HGB uses an internal validation split for early stopping (no eval_set arg).
    params["early_stopping"] = early_stop
    if early_stop:
        params.setdefault("validation_fraction", 0.1)
        params.setdefault("n_iter_no_change", 50)
    return HistGradientBoostingClassifier(**params)


# --- per-model fold trainers: return (val_proba, n_iterations_used) ---


def train_lgbm(X_tr, y_tr, X_va, y_va, n_classes, sample_weight, cfg):
    model = build_lgbm(cfg, n_classes, cfg.n_estimators)
    model.fit(
        X_tr,
        y_tr,
        sample_weight=sample_weight,
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(0)],
    )
    best = model.best_iteration_ or cfg.n_estimators
    return clip_proba(model.predict_proba(X_va)), int(best)


def train_xgb(X_tr, y_tr, X_va, y_va, n_classes, sample_weight, cfg):
    model = build_xgb(cfg, n_classes, cfg.n_estimators, early_stop=True)
    model.fit(X_tr, y_tr, sample_weight=sample_weight, eval_set=[(X_va, y_va)], verbose=False)
    best = model.best_iteration if model.best_iteration is not None else cfg.n_estimators - 1
    return clip_proba(model.predict_proba(X_va)), int(best) + 1


def train_hgb(X_tr, y_tr, X_va, y_va, n_classes, sample_weight, cfg):
    model = build_hgb(cfg, cfg.n_estimators, early_stop=True)
    model.fit(X_tr, y_tr, sample_weight=sample_weight)
    return clip_proba(model.predict_proba(X_va)), int(model.n_iter_)


def train_cat(train_df, feature_cols, cat_cols, y_tr, y_va, va_idx, tr_idx, n_classes, cfg):
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
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    train_pool = Pool(
        train_df.iloc[tr_idx][feature_cols], y_tr, cat_features=cat_idx or None, weight=sw
    )
    eval_pool = Pool(
        train_df.iloc[va_idx][feature_cols], y_va, cat_features=cat_idx or None
    )
    model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
    va_proba = clip_proba(model.predict_proba(train_df.iloc[va_idx][feature_cols]))
    best = model.get_best_iteration()
    best = (best + 1) if best is not None else cfg.cat_iterations
    return va_proba, int(best)


def _refit_iters(best_list: list[int], cap: int, default: int) -> int:
    if best_list:
        return min(int(np.mean(best_list) * 1.1), cap)
    return default


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
    n_train = len(y)
    models = set(cfg.models)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=cfg.seed)

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

    # OOF / test buffers only for the models we actually train this run.
    oof = {m: np.zeros((n_train, n_classes)) for m in models}
    test_pred: dict[str, np.ndarray] = {}
    best_iters: dict[str, list[int]] = {m: [] for m in models}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_lx, y), start=1):
        print(f"\n--- Fold {fold}/{N_SPLITS} (seed={cfg.seed}, feature_set={cfg.feature_set}) ---")
        y_tr, y_va = y[tr_idx], y[va_idx]
        sw = compute_sample_weight(class_weight="balanced", y=y_tr)

        if "lgb" in models:
            p, n_it = train_lgbm(X_lx[tr_idx], y_tr, X_lx[va_idx], y_va, n_classes, sw, cfg)
            oof["lgb"][va_idx] = p
            best_iters["lgb"].append(n_it)
            r = evaluate_proba(y_va, p)
            print(f"LGB  BA={r.oof_balanced_accuracy:.5f} STAR_r={r.per_class_recall['STAR']:.4f} iters={n_it}")

        if "xgb" in models:
            p, n_it = train_xgb(X_lx[tr_idx], y_tr, X_lx[va_idx], y_va, n_classes, sw, cfg)
            oof["xgb"][va_idx] = p
            best_iters["xgb"].append(n_it)
            r = evaluate_proba(y_va, p)
            print(f"XGB  BA={r.oof_balanced_accuracy:.5f} STAR_r={r.per_class_recall['STAR']:.4f} iters={n_it}")

        if "hgb" in models:
            p, n_it = train_hgb(X_lx[tr_idx], y_tr, X_lx[va_idx], y_va, n_classes, sw, cfg)
            oof["hgb"][va_idx] = p
            best_iters["hgb"].append(n_it)
            r = evaluate_proba(y_va, p)
            print(f"HGB  BA={r.oof_balanced_accuracy:.5f} STAR_r={r.per_class_recall['STAR']:.4f} iters={n_it}")

        if "cat" in models:
            p, n_it = train_cat(
                train_cb, cb_cols, cb_cat_names, y_tr, y_va, va_idx, tr_idx, n_classes, cfg
            )
            oof["cat"][va_idx] = p
            best_iters["cat"].append(n_it)
            r = evaluate_proba(y_va, p)
            print(f"CAT  BA={r.oof_balanced_accuracy:.5f} STAR_r={r.per_class_recall['STAR']:.4f} iters={n_it}")

    # For models not trained this run, reuse previous OOF so the blend stays complete.
    for m in ALL_MODELS:
        if m not in models:
            prev = _load_previous_oof(m, n_train, n_classes)
            if prev is not None:
                oof[m] = prev

    print("\n--- Full fit for test ---")
    sw_full = compute_sample_weight(class_weight="balanced", y=y)

    if "lgb" in models:
        n_est = _refit_iters(best_iters["lgb"], cfg.n_estimators, cfg.n_estimators)
        lgb_full = build_lgbm(cfg, n_classes, n_est)
        lgb_full.fit(X_lx, y, sample_weight=sw_full)
        test_pred["lgb"] = clip_proba(lgb_full.predict_proba(X_test_lx))
        print(f"LGB full refit iters={n_est}")

    if "xgb" in models:
        n_est = _refit_iters(best_iters["xgb"], cfg.n_estimators, cfg.n_estimators)
        xgb_full = build_xgb(cfg, n_classes, n_est, early_stop=False)
        xgb_full.fit(X_lx, y, sample_weight=sw_full)
        test_pred["xgb"] = clip_proba(xgb_full.predict_proba(X_test_lx))
        print(f"XGB full refit iters={n_est}")

    if "hgb" in models:
        n_est = _refit_iters(best_iters["hgb"], cfg.n_estimators, cfg.n_estimators)
        hgb_full = build_hgb(cfg, n_est, early_stop=False)
        hgb_full.fit(X_lx, y, sample_weight=sw_full)
        test_pred["hgb"] = clip_proba(hgb_full.predict_proba(X_test_lx))
        print(f"HGB full refit iters={n_est}")

    if "cat" in models:
        from catboost import CatBoostClassifier, Pool

        n_est = _refit_iters(best_iters["cat"], cfg.cat_iterations, cfg.cat_iterations)
        cat_idx = [cb_cols.index(c) for c in cb_cat_names] if cb_cat_names else []
        cat_full = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=n_est,
            learning_rate=cfg.cat_learning_rate,
            depth=cfg.cat_depth,
            random_seed=cfg.seed,
            verbose=0,
            auto_class_weights="Balanced",
        )
        cat_full.fit(Pool(train_cb[cb_cols], y, cat_features=cat_idx or None, weight=sw_full))
        test_pred["cat"] = clip_proba(cat_full.predict_proba(test_cb[cb_cols]))
        print(f"CAT full refit iters={n_est}")

    # Reuse previous test predictions for models not trained this run.
    for m in ALL_MODELS:
        if m not in models:
            prev = ARTIFACTS / f"test_{m}.npy"
            if prev.exists():
                test_pred[m] = np.load(prev)

    for m, arr in oof.items():
        np.save(ARTIFACTS / f"oof_{m}.npy", arr)
    for m, arr in test_pred.items():
        np.save(ARTIFACTS / f"test_{m}.npy", arr)
    np.save(ARTIFACTS / "y_train.npy", y)
    (ARTIFACTS / "classes.json").write_text(json.dumps(CLASS_ORDER))

    print("\n=== Single-model OOF (argmax) ===")
    model_scores: dict[str, float] = {}
    for m in ALL_MODELS:
        if m in oof and not np.allclose(oof[m], 0):
            r = evaluate_proba(y, oof[m])
            model_scores[m] = r.oof_balanced_accuracy
            print(f"{m}: BA={r.oof_balanced_accuracy:.5f} recalls={r.per_class_recall}")

    meta = {
        "feature_set": cfg.feature_set,
        "seed": cfg.seed,
        "models": list(cfg.models),
        "use_coords": cfg.use_coords,
        "cat_native": cfg.cat_native,
        "n_features_lgb": len(lx_cols),
        "n_features_cat": len(cb_cols),
        "best_iterations": {m: best_iters[m] for m in models},
        "model_oof_ba": model_scores,
    }
    (ARTIFACTS / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nArtifacts saved to {ARTIFACTS}")
    return meta


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="5-fold OOF training for S6E6")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="all")
    parser.add_argument("--models", default=",".join(ALL_MODELS), help="Comma-separated subset")
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
            "notes": json.dumps(
                {
                    "n_features_lgb": meta["n_features_lgb"],
                    "cat_native": cfg.cat_native,
                    "hgb_oof_ba": meta["model_oof_ba"].get("hgb"),
                }
            ),
        }
    )


if __name__ == "__main__":
    main()
