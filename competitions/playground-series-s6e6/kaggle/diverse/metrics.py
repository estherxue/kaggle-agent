"""Evaluation metrics — primary: OOF balanced accuracy."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, recall_score

from config import CLASS_ORDER, CLIP


@dataclass
class EvalResult:
    oof_balanced_accuracy: float
    per_class_recall: dict[str, float]
    log_loss: float
    macro_f1: float
    macro_recall: float

    def to_dict(self) -> dict:
        return asdict(self)


def clip_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(proba, CLIP, 1 - CLIP)
    return proba / proba.sum(axis=1, keepdims=True)


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    recalls = {}
    for i, name in enumerate(CLASS_ORDER):
        mask = y_true == i
        if mask.sum() == 0:
            recalls[name] = 0.0
        else:
            recalls[name] = float((y_pred[mask] == i).mean())
    return recalls


def evaluate_proba(
    y_true: np.ndarray,
    proba: np.ndarray,
    bias: np.ndarray | None = None,
) -> EvalResult:
    proba = clip_proba(proba)
    if bias is not None:
        logp = np.log(proba)
        y_pred = np.argmax(logp + bias, axis=1)
    else:
        y_pred = proba.argmax(axis=1)

    n_classes = proba.shape[1]
    return EvalResult(
        oof_balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        per_class_recall=per_class_recall(y_true, y_pred),
        log_loss=float(log_loss(y_true, proba, labels=list(range(n_classes)))),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        macro_recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    )


def predict_labels(proba: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    proba = clip_proba(proba)
    if bias is not None:
        return np.argmax(np.log(proba) + bias, axis=1)
    return proba.argmax(axis=1)
