from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dafbot.config import ensure_directories, load_config
from dafbot.data.dataset import load_processed_dataset
from dafbot.pipeline import run_preprocess_pipeline
from dafbot.train.trainer import EXPERIMENT_VARIANTS, evaluate_checkpoint, train_experiment


ABLATION_EXPERIMENTS = [
    "dynamic_graph_adaptive",
    "ablation_no_temporal",
    "ablation_no_adaptive_fusion",
    "ablation_no_quality",
    "dynamic_graph_concat",
    "dynamic_graph_only",
    "attr_only",
    "text_only",
    "static_graph_only",
    "attr_text_concat",
    "attr_text_static_graph_concat",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing, training, evaluation, and optional ablation in one command.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--experiment",
        type=str,
        default="dynamic_graph_adaptive",
        choices=sorted(EXPERIMENT_VARIANTS),
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--with-ablation", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)

    if not args.skip_preprocess:
        print("[1/4] Running preprocessing...")
        run_preprocess_pipeline(config)
        print("Preprocessing finished.")

    print("[2/4] Loading processed dataset...")
    dataset = load_processed_dataset(config)
    print("Processed dataset loaded.")

    checkpoint_path: Path | None = None
    main_metrics_frame: pd.DataFrame | None = None
    if not args.skip_train:
        print(f"[3/4] Training experiment: {args.experiment}")
        artifacts = train_experiment(config, dataset, args.experiment)
        checkpoint_path = artifacts.checkpoint_path
        main_metrics_frame = artifacts.metrics_frame
        print(f"Training finished. Checkpoint saved to {checkpoint_path}")
    elif args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = Path(config["paths"]["checkpoint_dir"]) / f"{args.experiment}.pt"

    if not args.skip_eval:
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"[4/4] Evaluating checkpoint: {checkpoint_path}")
        metrics = evaluate_checkpoint(config, dataset, checkpoint_path)
        print(metrics.to_string(index=False))

    if args.with_ablation:
        print("Running baseline and ablation experiments...")
        frames = []
        for experiment in ABLATION_EXPERIMENTS:
            if experiment == args.experiment and main_metrics_frame is not None:
                print(f"  - {experiment} (reuse existing result)")
                frames.append(main_metrics_frame)
                continue
            print(f"  - {experiment}")
            artifacts = train_experiment(config, dataset, experiment)
            frames.append(artifacts.metrics_frame)
        summary = pd.concat(frames, ignore_index=True)
        summary_path = Path(config["paths"]["table_dir"]) / "run_all_ablation_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Ablation summary saved to {summary_path}")

    print("Run all pipeline finished.")


if __name__ == "__main__":
    main()
