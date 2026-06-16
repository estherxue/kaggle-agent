"""Blend OOF probabilities: ensemble weights + class-bias for balanced accuracy."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from sklearn.metrics import balanced_accuracy_score

from config import ARTIFACTS, CLASS_ORDER, DATA, ID_COL, SUBMISSIONS, TARGET_COL
from experiment_log import log_experiment
from features import load_data
from metrics import clip_proba, evaluate_proba, predict_labels


def load_oof_stack() -> tuple[np.ndarray, np.ndarray, list[str]]:
    y = np.load(ARTIFACTS / "y_train.npy")
    oof_lgb = np.load(ARTIFACTS / "oof_lgb.npy")
    oof_xgb = np.load(ARTIFACTS / "oof_xgb.npy")
    oof_cat = np.load(ARTIFACTS / "oof_cat.npy")
    stack = np.stack([oof_lgb, oof_xgb, oof_cat], axis=0)
    test_lgb = np.load(ARTIFACTS / "test_lgb.npy")
    test_xgb = np.load(ARTIFACTS / "test_xgb.npy")
    test_cat = np.load(ARTIFACTS / "test_cat.npy")
    test_stack = np.stack([test_lgb, test_xgb, test_cat], axis=0)
    return y, stack, test_stack


def blend_stack(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = weights / weights.sum()
    return clip_proba(np.tensordot(w, stack, axes=(0, 0)))


def search_weights(y: np.ndarray, stack: np.ndarray) -> np.ndarray:
    n_models = stack.shape[0]

    def objective(w_raw: np.ndarray) -> float:
        w = np.abs(w_raw)
        if w.sum() < 1e-9:
            return 1.0
        proba = blend_stack(stack, w)
        pred = proba.argmax(axis=1)
        return -balanced_accuracy_score(y, pred)

    x0 = np.ones(n_models) / n_models
    res = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 500, "xatol": 1e-4})
    w = np.abs(res.x)
    return w / w.sum()


def search_class_bias(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    eps = 1e-15
    logp = np.log(np.clip(proba, eps, 1 - eps))
    n_classes = proba.shape[1]

    def objective(bias: np.ndarray) -> float:
        pred = np.argmax(logp + bias, axis=1)
        return -balanced_accuracy_score(y, pred)

    res = differential_evolution(
        objective,
        bounds=[(-1.0, 1.0)] * n_classes,
        seed=42,
        tol=1e-6,
        maxiter=100,
        polish=True,
    )
    return res.x


def parse_weights(weights_str: str) -> np.ndarray:
    parts = [float(x.strip()) for x in weights_str.split(",")]
    if len(parts) != 3:
        raise ValueError("--weights must have exactly 3 comma-separated values (lgb,xgb,cat)")
    return np.array(parts, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend OOF models for S6E6")
    parser.add_argument("--weights", help="Fixed blend weights: lgb,xgb,cat (e.g. 0.59,0.41,0.00)")
    parser.add_argument("--no-bias", action="store_true", help="Skip class-bias search; use zero bias")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output submission path (default: submissions/blend_v1.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, test, _ = load_data(DATA)
    y, oof_stack, test_stack = load_oof_stack()

    meta_path = ARTIFACTS / "train_meta.json"
    train_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    if args.weights:
        weights = parse_weights(args.weights)
        print(f"Using fixed weights LGB/XGB/CAT: {weights}")
    else:
        print("Searching ensemble weights (OOF balanced accuracy)...")
        weights = search_weights(y, oof_stack)
        print(f"Weights LGB/XGB/CAT: {weights}")

    oof_blend = blend_stack(oof_stack, weights)
    r_before = evaluate_proba(y, oof_blend)
    print(f"Blended OOF before bias: BA={r_before.oof_balanced_accuracy:.5f} recalls={r_before.per_class_recall}")

    if args.no_bias:
        bias = np.zeros(len(CLASS_ORDER))
        print("Skipping class bias (--no-bias)")
    else:
        print("Searching class bias...")
        bias = search_class_bias(y, oof_blend)
        print(f"Class bias (GALAXY, QSO, STAR): {bias}")

    r_after = evaluate_proba(y, oof_blend, bias=None if args.no_bias else bias)
    label = "after bias" if not args.no_bias else "no bias"
    print(f"Blended OOF {label}:  BA={r_after.oof_balanced_accuracy:.5f} recalls={r_after.per_class_recall}")
    print(f"  (reference log_loss={r_after.log_loss:.5f} macro_f1={r_after.macro_f1:.5f})")

    test_blend = blend_stack(test_stack, weights)
    pred_idx = predict_labels(test_blend, bias=None if args.no_bias else bias)
    pred_labels = [CLASS_ORDER[i] for i in pred_idx]

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    out_path = args.output or (SUBMISSIONS / "blend_v1.csv")
    if not out_path.is_absolute():
        out_path = (SUBMISSIONS.parent / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: pred_labels}).to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    blend_cfg = {
        "experiment": "blend",
        "weights": {"lgb": float(weights[0]), "xgb": float(weights[1]), "cat": float(weights[2])},
        "class_bias": {c: float(b) for c, b in zip(CLASS_ORDER, bias)},
        "no_bias": args.no_bias,
        "oof_balanced_accuracy_before_bias": r_before.oof_balanced_accuracy,
        "oof_balanced_accuracy": r_after.oof_balanced_accuracy,
        "per_class_recall": r_after.per_class_recall,
        "log_loss": r_after.log_loss,
        "submission": str(out_path.name),
        "seed": train_meta.get("seed"),
        "feature_set": train_meta.get("feature_set"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (ARTIFACTS / "blend_config.json").write_text(json.dumps(blend_cfg, indent=2))
    log_experiment(blend_cfg)


if __name__ == "__main__":
    main()
