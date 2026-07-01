"""Lightweight hyper-parameter tuning for S6E6.

Uses Optuna if installed, otherwise falls back to random search. Tunes *structural*
params only (NOT n_estimators — early stopping decides that) on a stratified subsample
with short CV, then writes ``artifacts/best_params_<model>.json``. train_oof.py layers
those on top of its defaults automatically, so no other code needs editing.

Usage:
    python tune.py --model lgb --trials 25 --subsample 150000 --feature-set all_v2
"""

from __future__ import annotations

import argparse
import json
import random

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from config import ARTIFACTS, CLASS_ORDER, DATA, EARLY_STOPPING_ROUNDS, RANDOM_STATE
from features import FEATURE_SETS, load_data, prepare_lgb_xgb
from train_oof import encode_target

TUNE_FOLDS = 3
TUNE_CAP = 1500  # boosting-round cap during tuning; early stopping cuts it short
TUNABLE = ("lgb", "xgb", "hgb")


class TrialSampler:
    """Unifies optuna trial suggestions and plain random sampling."""

    def __init__(self, trial=None, rng: random.Random | None = None):
        self.trial = trial
        self.rng = rng

    def int(self, name: str, lo: int, hi: int) -> int:
        if self.trial is not None:
            return self.trial.suggest_int(name, lo, hi)
        return self.rng.randint(lo, hi)

    def float(self, name: str, lo: float, hi: float, log: bool = False) -> float:
        if self.trial is not None:
            return self.trial.suggest_float(name, lo, hi, log=log)
        if log:
            return float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))
        return self.rng.uniform(lo, hi)


def lgb_params(s: TrialSampler) -> dict:
    return dict(
        num_leaves=s.int("num_leaves", 31, 255),
        max_depth=s.int("max_depth", 4, 12),
        learning_rate=s.float("learning_rate", 0.02, 0.1, log=True),
        min_child_samples=s.int("min_child_samples", 10, 200),
        subsample=s.float("subsample", 0.6, 1.0),
        colsample_bytree=s.float("colsample_bytree", 0.6, 1.0),
        reg_alpha=s.float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=s.float("reg_lambda", 1e-3, 10.0, log=True),
    )


def xgb_params(s: TrialSampler) -> dict:
    return dict(
        max_depth=s.int("max_depth", 4, 12),
        learning_rate=s.float("learning_rate", 0.02, 0.1, log=True),
        min_child_weight=s.float("min_child_weight", 1.0, 20.0),
        subsample=s.float("subsample", 0.6, 1.0),
        colsample_bytree=s.float("colsample_bytree", 0.6, 1.0),
        reg_alpha=s.float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=s.float("reg_lambda", 1e-3, 10.0, log=True),
    )


def hgb_params(s: TrialSampler) -> dict:
    return dict(
        learning_rate=s.float("learning_rate", 0.02, 0.2, log=True),
        max_leaf_nodes=s.int("max_leaf_nodes", 31, 255),
        min_samples_leaf=s.int("min_samples_leaf", 20, 200),
        l2_regularization=s.float("l2_regularization", 1e-3, 10.0, log=True),
    )


PARAM_FN = {"lgb": lgb_params, "xgb": xgb_params, "hgb": hgb_params}


def make_model(model_name: str, params: dict, n_classes: int, seed: int):
    if model_name == "lgb":
        return lgb.LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            n_estimators=TUNE_CAP,
            random_state=seed,
            verbose=-1,
            **params,
        )
    if model_name == "xgb":
        return xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=TUNE_CAP,
            tree_method="hist",
            random_state=seed,
            verbosity=0,
            eval_metric="mlogloss",
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **params,
        )
    return HistGradientBoostingClassifier(
        max_iter=TUNE_CAP,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=seed,
        **params,
    )


def cv_score(model_name: str, params: dict, X, y, n_classes: int, seed: int) -> float:
    skf = StratifiedKFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), n_classes))
    for tr, va in skf.split(X, y):
        sw = compute_sample_weight(class_weight="balanced", y=y[tr])
        model = make_model(model_name, params, n_classes, seed)
        if model_name == "lgb":
            model.fit(
                X[tr], y[tr], sample_weight=sw, eval_set=[(X[va], y[va])],
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(0)],
            )
        elif model_name == "xgb":
            model.fit(X[tr], y[tr], sample_weight=sw, eval_set=[(X[va], y[va])], verbose=False)
        else:
            model.fit(X[tr], y[tr], sample_weight=sw)
        oof[va] = model.predict_proba(X[va])
    return float(balanced_accuracy_score(y, oof.argmax(axis=1)))


def optimize(model_name: str, X, y, n_classes: int, n_trials: int, seed: int) -> tuple[dict, float]:
    param_fn = PARAM_FN[model_name]
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            return cv_score(model_name, param_fn(TrialSampler(trial=trial)), X, y, n_classes, seed)

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        print(f"[optuna] best BA={study.best_value:.5f}")
        return study.best_params, study.best_value
    except ImportError:
        print("optuna not installed — using random search fallback")
        rng = random.Random(seed)
        best_params, best_score = None, -1.0
        for i in range(n_trials):
            params = param_fn(TrialSampler(rng=rng))
            score = cv_score(model_name, params, X, y, n_classes, seed)
            if score > best_score:
                best_score, best_params = score, params
            print(f"trial {i + 1}/{n_trials} BA={score:.5f} best={best_score:.5f}")
        return best_params, best_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lightweight HPO for S6E6 (writes best_params_<model>.json)")
    p.add_argument("--model", choices=TUNABLE, required=True)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--subsample", type=int, default=150000, help="Stratified subsample size (0 = full)")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--feature-set", choices=FEATURE_SETS, default="all_v2")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train, test, _ = load_data(DATA)
    y, _ = encode_target(train)
    n_classes = len(CLASS_ORDER)

    train_f, _, cols = prepare_lgb_xgb(train, test, feature_set=args.feature_set, use_coords=True)
    X = train_f[cols].values

    if args.subsample and args.subsample < len(y):
        X, _, y, _ = train_test_split(
            X, y, train_size=args.subsample, stratify=y, random_state=args.seed
        )
        print(f"Tuning on stratified subsample of {len(y)} rows")

    best_params, best_score = optimize(args.model, X, y, n_classes, args.trials, args.seed)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / f"best_params_{args.model}.json"
    out.write_text(json.dumps(best_params, indent=2))
    print(f"\nBest {args.model} params (OOF BA={best_score:.5f}) saved to {out}:")
    print(json.dumps(best_params, indent=2))


if __name__ == "__main__":
    main()
