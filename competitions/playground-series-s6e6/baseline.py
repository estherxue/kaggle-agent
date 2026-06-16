"""Baseline for playground-series-s6e6: weighted multiclass GBDT ensemble."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SUBMISSIONS = ROOT / "submissions"
RANDOM_STATE = 42
N_SPLITS = 5
N_ESTIMATORS = 300
CLIP = 1e-15

TARGET_COL = "class"
ID_COL = "id"
CAT_COLS = ["spectral_type", "galaxy_population"]
CLASS_ORDER = ["GALAXY", "QSO", "STAR"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    return train, test, sample


def add_features(df: pd.DataFrame, use_coords: bool = True) -> pd.DataFrame:
    out = df.copy()

    # alpha/delta are RA/Dec in degrees
    if use_coords and {"alpha", "delta"}.issubset(out.columns):
        ra = out["alpha"].astype(float)
        dec = out["delta"].astype(float)
        out["ra_dec"] = ra * dec
        out["sin_ra"] = np.sin(np.radians(ra))
        out["cos_ra"] = np.cos(np.radians(ra))
        out["sin_dec"] = np.sin(np.radians(dec))
        out["cos_dec"] = np.cos(np.radians(dec))

    bands = [c for c in ["u", "g", "r", "i", "z"] if c in out.columns]
    for i in range(len(bands)):
        for j in range(i + 1, len(bands)):
            a, b = bands[i], bands[j]
            out[f"ratio_{a}_{b}"] = out[a] / (out[b].abs() + 1e-6)

    if "redshift" in out.columns:
        out["redshift_bin"] = pd.qcut(
            out["redshift"], q=10, duplicates="drop", labels=False
        )

    return out


def encode_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_f = add_features(train)
    test_f = add_features(test)

    for col in CAT_COLS:
        if col in train_f.columns:
            le = LabelEncoder()
            combined = pd.concat([train_f[col], test_f[col]], axis=0).astype(str)
            le.fit(combined)
            train_f[col] = le.transform(train_f[col].astype(str))
            test_f[col] = le.transform(test_f[col].astype(str))

    drop_cols = {ID_COL, TARGET_COL}
    feature_cols = [c for c in train_f.columns if c not in drop_cols and c in test_f.columns]
    return train_f, test_f, feature_cols


def prepare_xy(
    train_f: pd.DataFrame,
    test_f: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    y = le.transform(train_f[TARGET_COL].astype(str))
    X = train_f[feature_cols].values
    X_test = test_f[feature_cols].values
    return X, y, X_test, le


def clip_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(proba, CLIP, 1 - CLIP)
    return proba / proba.sum(axis=1, keepdims=True)


def cv_ensemble(X: np.ndarray, y: np.ndarray, n_classes: int) -> dict[str, float]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((len(y), n_classes))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        sw = compute_sample_weight(class_weight="balanced", y=y_tr)

        lgbm = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            n_estimators=N_ESTIMATORS,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        lgbm.fit(X_tr, y_tr, sample_weight=sw)

        xgb_clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=N_ESTIMATORS,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        xgb_clf.fit(X_tr, y_tr, sample_weight=sw)

        p_va = clip_proba((lgbm.predict_proba(X_va) + xgb_clf.predict_proba(X_va)) / 2)
        oof[va_idx] = p_va

        y_pred = p_va.argmax(axis=1)
        print(
            f"Fold {fold}:",
            {
                "log_loss": log_loss(y_va, p_va, labels=list(range(n_classes))),
                "balanced_accuracy": balanced_accuracy_score(y_va, y_pred),
                "macro_f1": f1_score(y_va, y_pred, average="macro", zero_division=0),
                "macro_recall": recall_score(y_va, y_pred, average="macro", zero_division=0),
            },
        )
        print("Confusion matrix:\n", confusion_matrix(y_va, y_pred))

    y_hat = oof.argmax(axis=1)
    overall = {
        "log_loss": log_loss(y, oof, labels=list(range(n_classes))),
        "balanced_accuracy": balanced_accuracy_score(y, y_hat),
        "macro_f1": f1_score(y, y_hat, average="macro", zero_division=0),
        "macro_recall": recall_score(y, y_hat, average="macro", zero_division=0),
    }
    print("\nOOF overall:", overall)
    print(classification_report(y, y_hat, target_names=CLASS_ORDER))
    return overall


def train_full_and_predict(
    X: np.ndarray, y: np.ndarray, X_test: np.ndarray, n_classes: int
) -> np.ndarray:
    sw = compute_sample_weight(class_weight="balanced", y=y)

    lgbm = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=N_ESTIMATORS,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    lgbm.fit(X, y, sample_weight=sw)

    xgb_clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=N_ESTIMATORS,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    xgb_clf.fit(X, y, sample_weight=sw)

    return clip_proba((lgbm.predict_proba(X_test) + xgb_clf.predict_proba(X_test)) / 2)


def run_ablation() -> None:
    train, test, sample = load_data()

    print("Train shape:", train.shape)
    print("Class distribution:\n", train[TARGET_COL].value_counts(normalize=True))

    results: dict[str, dict[str, float]] = {}
    for use_coords in (False, True):
        label = "with_coords" if use_coords else "no_coords"
        train_f = add_features(train, use_coords=use_coords)
        test_f = add_features(test, use_coords=use_coords)
        for col in CAT_COLS:
            if col in train_f.columns:
                le = LabelEncoder()
                combined = pd.concat([train_f[col], test_f[col]], axis=0).astype(str)
                le.fit(combined)
                train_f[col] = le.transform(train_f[col].astype(str))
                test_f[col] = le.transform(test_f[col].astype(str))
        drop_cols = {ID_COL, TARGET_COL}
        feature_cols = [c for c in train_f.columns if c not in drop_cols and c in test_f.columns]

        print(f"\n=== CV {label} ({len(feature_cols)} features) ===")
        X, y, X_test, le = prepare_xy(train_f, test_f, feature_cols)
        results[label] = cv_ensemble(X, y, n_classes=len(CLASS_ORDER))

    print("\n=== Ablation summary ===")
    print(json.dumps(results, indent=2))

    best = min(results.keys(), key=lambda k: (results[k]["log_loss"], -results[k]["macro_f1"]))
    use_coords = best == "with_coords"
    print(f"\nTraining full model ({best})...")

    train_f, test_f, feature_cols = encode_features(train, test)
    if not use_coords:
        coord_cols = {"ra_dec", "sin_ra", "cos_ra", "sin_dec", "cos_dec"}
        feature_cols = [c for c in feature_cols if c not in coord_cols]

    X, y, X_test, le = prepare_xy(train_f, test_f, feature_cols)
    proba = train_full_and_predict(X, y, X_test, n_classes=len(CLASS_ORDER))

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    pred_labels = le.inverse_transform(proba.argmax(axis=1))
    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: pred_labels})
    out_path = SUBMISSIONS / "baseline.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved submission: {out_path}")


if __name__ == "__main__":
    run_ablation()
