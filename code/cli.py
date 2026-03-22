from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import PipelineConfig
from .data import prepare_dataset
from .experiments import run_classification_experiments, run_group_detection
from .features import enrich_with_graph_features
from .visualization import generate_visualizations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Social computing experiment pipeline",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("result"))
    parser.add_argument("--max-labeled-users", type=int, default=None)
    parser.add_argument("--max-tweets-per-user", type=int, default=8)
    parser.add_argument("--tfidf-max-features", type=int, default=8000)
    parser.add_argument("--tfidf-min-df", type=int, default=3)
    parser.add_argument("--deepwalk-dimensions", type=int, default=64)
    parser.add_argument("--deepwalk-walk-length", type=int, default=20)
    parser.add_argument("--deepwalk-num-walks", type=int, default=5)
    parser.add_argument("--deepwalk-window", type=int, default=5)
    parser.add_argument("--deepwalk-epochs", type=int, default=5)
    parser.add_argument("--betweenness-sample-k", type=int, default=128)
    parser.add_argument("--harmonic-sample-sources", type=int, default=128)
    parser.add_argument("--tsne-sample-size", type=int, default=3000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Parse the dataset and build cached features")
    subparsers.add_parser("train", help="Run classification baselines")

    cluster_parser = subparsers.add_parser("cluster", help="Run suspicious group discovery")
    cluster_parser.add_argument("--method", choices=["dbscan", "spectral"], default="dbscan")
    cluster_parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    cluster_parser.add_argument("--threshold", type=float, default=0.8)
    cluster_parser.add_argument("--use-ground-truth", action="store_true")

    subparsers.add_parser("visualize", help="Generate charts and cluster network plots")

    run_all_parser = subparsers.add_parser("run-all", help="Execute prepare/train/cluster/visualize in sequence")
    run_all_parser.add_argument("--method", choices=["dbscan", "spectral"], default="dbscan")
    run_all_parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    run_all_parser.add_argument("--threshold", type=float, default=0.8)
    run_all_parser.add_argument("--use-ground-truth", action="store_true")
    return parser


def make_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_labeled_users=args.max_labeled_users,
        max_tweets_per_user=args.max_tweets_per_user,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_min_df=args.tfidf_min_df,
        deepwalk_dimensions=args.deepwalk_dimensions,
        deepwalk_walk_length=args.deepwalk_walk_length,
        deepwalk_num_walks=args.deepwalk_num_walks,
        deepwalk_window=args.deepwalk_window,
        deepwalk_epochs=args.deepwalk_epochs,
        betweenness_sample_k=args.betweenness_sample_k,
        harmonic_sample_sources=args.harmonic_sample_sources,
        tsne_sample_size=args.tsne_sample_size,
        random_state=args.random_state,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = make_config(args)
    config.ensure_dirs()

    if args.command == "prepare":
        prepared = prepare_dataset(config)
        enrich_with_graph_features(config, prepared)
    elif args.command == "train":
        run_classification_experiments(config)
    elif args.command == "cluster":
        run_group_detection(
            config,
            method=args.method,
            split=args.split,
            threshold=args.threshold,
            use_ground_truth=args.use_ground_truth,
        )
    elif args.command == "visualize":
        generate_visualizations(config)
    elif args.command == "run-all":
        prepared = prepare_dataset(config)
        enrich_with_graph_features(config, prepared)
        run_classification_experiments(config)
        run_group_detection(
            config,
            method=args.method,
            split=args.split,
            threshold=args.threshold,
            use_ground_truth=args.use_ground_truth,
        )
        generate_visualizations(config)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
