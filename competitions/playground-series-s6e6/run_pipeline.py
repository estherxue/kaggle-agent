"""Run full S6E6 pipeline: train_oof -> blend, with multi-seed support."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from blend import main as run_blend
from config import ARTIFACTS, RANDOM_STATE, SUBMISSIONS
from experiment_log import log_experiment
from features import FEATURE_SETS
from train_oof import ALL_MODELS, TrainConfig, run_oof

ALL_MODELS_STR = ",".join(ALL_MODELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S6E6 train + blend pipeline")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="all")
    parser.add_argument("--seeds", default=str(RANDOM_STATE), help="Comma-separated CV seeds")
    parser.add_argument("--models", default=ALL_MODELS_STR)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--weights", help="Fixed blend weights: lgb,xgb,cat")
    parser.add_argument("--no-coords", action="store_true")
    parser.add_argument("--cat-native", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Only blend existing artifacts")
    return parser.parse_args()


def _snapshot_artifacts(seed: int, feature_set: str) -> Path:
    snap = ARTIFACTS / f"snapshot_seed{seed}_{feature_set}"
    snap.mkdir(parents=True, exist_ok=True)
    for name in [
        "oof_lgb.npy", "oof_xgb.npy", "oof_cat.npy",
        "test_lgb.npy", "test_xgb.npy", "test_cat.npy",
        "y_train.npy", "train_meta.json", "blend_config.json",
    ]:
        src = ARTIFACTS / name
        if src.exists():
            shutil.copy2(src, snap / name)
    return snap


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())

    for seed in seeds:
        print(f"\n{'=' * 60}")
        print(f"Pipeline seed={seed} feature_set={args.feature_set}")
        print(f"{'=' * 60}")

        if not args.skip_train:
            cfg = TrainConfig(
                feature_set=args.feature_set,
                models=models,
                seed=seed,
                use_coords=not args.no_coords,
                cat_native=args.cat_native,
            )
            meta = run_oof(cfg)
            log_experiment(
                {
                    "experiment": "pipeline_train",
                    "seed": seed,
                    "feature_set": args.feature_set,
                    "models": ",".join(models),
                    "lgb_oof_ba": meta["model_oof_ba"].get("lgb"),
                    "xgb_oof_ba": meta["model_oof_ba"].get("xgb"),
                    "cat_oof_ba": meta["model_oof_ba"].get("cat"),
                }
            )

        blend_argv = ["blend.py"]
        if args.no_bias:
            blend_argv.append("--no-bias")
        if args.weights:
            blend_argv.extend(["--weights", args.weights])
        suffix = f"seed{seed}_{args.feature_set}"
        if args.no_bias:
            suffix += "_nobias"
        out = SUBMISSIONS / f"blend_{suffix}.csv"
        blend_argv.extend(["--output", str(out)])

        import sys
        old_argv = sys.argv
        try:
            sys.argv = blend_argv
            run_blend()
        finally:
            sys.argv = old_argv

        snap = _snapshot_artifacts(seed, args.feature_set)
        blend_cfg = json.loads((ARTIFACTS / "blend_config.json").read_text())
        log_experiment(
            {
                "experiment": "pipeline_blend",
                "seed": seed,
                "feature_set": args.feature_set,
                "models": ",".join(models),
                "no_bias": args.no_bias,
                "oof_balanced_accuracy": blend_cfg.get("oof_balanced_accuracy"),
                "oof_balanced_accuracy_before_bias": blend_cfg.get("oof_balanced_accuracy_before_bias"),
                "per_class_recall": blend_cfg.get("per_class_recall"),
                "weights": blend_cfg.get("weights"),
                "submission": blend_cfg.get("submission"),
                "notes": f"snapshot={snap.name}",
            }
        )


if __name__ == "__main__":
    main()
