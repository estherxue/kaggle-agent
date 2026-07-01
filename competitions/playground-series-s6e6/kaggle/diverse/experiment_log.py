"""Append experiment results to experiments.csv and experiments.jsonl."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from config import ARTIFACTS

CSV_COLUMNS = [
    "timestamp",
    "experiment",
    "seed",
    "feature_set",
    "models",
    "oof_balanced_accuracy",
    "oof_balanced_accuracy_before_bias",
    "per_class_recall_galaxy",
    "per_class_recall_qso",
    "per_class_recall_star",
    "lgb_oof_ba",
    "xgb_oof_ba",
    "cat_oof_ba",
    "blend_weights",
    "class_bias",
    "no_bias",
    "submission",
    "public_lb",
    "notes",
]


def _flatten_record(record: dict) -> dict:
    recalls = record.get("per_class_recall") or {}
    weights = record.get("weights") or {}
    row = {
        "timestamp": record.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "experiment": record.get("experiment", ""),
        "seed": record.get("seed", ""),
        "feature_set": record.get("feature_set", ""),
        "models": record.get("models", ""),
        "oof_balanced_accuracy": record.get("oof_balanced_accuracy", ""),
        "oof_balanced_accuracy_before_bias": record.get("oof_balanced_accuracy_before_bias", ""),
        "per_class_recall_galaxy": recalls.get("GALAXY", ""),
        "per_class_recall_qso": recalls.get("QSO", ""),
        "per_class_recall_star": recalls.get("STAR", ""),
        "lgb_oof_ba": record.get("lgb_oof_ba", ""),
        "xgb_oof_ba": record.get("xgb_oof_ba", ""),
        "cat_oof_ba": record.get("cat_oof_ba", ""),
        "blend_weights": json.dumps(weights) if weights else "",
        "class_bias": json.dumps(record.get("class_bias") or {}),
        "no_bias": record.get("no_bias", ""),
        "submission": record.get("submission", ""),
        "public_lb": record.get("public_lb", ""),
        "notes": record.get("notes", ""),
    }
    return row


def log_experiment(record: dict, artifacts_dir: Path | None = None) -> None:
    artifacts = artifacts_dir or ARTIFACTS
    artifacts.mkdir(parents=True, exist_ok=True)

    if "timestamp" not in record:
        record = {**record, "timestamp": datetime.now(timezone.utc).isoformat()}

    jsonl_path = artifacts / "experiments.jsonl"
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    csv_path = artifacts / "experiments.csv"
    row = _flatten_record(record)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
