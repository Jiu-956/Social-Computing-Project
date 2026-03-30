from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import ProjectConfig
from .data import prepare_dataset
from .experiments import run_experiments
from .reporting import generate_report
from .visualization import generate_visualizations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TwiBot-20 bot detection research pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-graph-users", type=int, default=None)
    parser.add_argument("--max-tweets-per-user", type=int, default=12)

    parser.add_argument("--tfidf-max-features", type=int, default=10000)
    parser.add_argument("--tfidf-min-df", type=int, default=3)
    parser.add_argument("--dense-text-max-features", type=int, default=12000)
    parser.add_argument("--dense-text-svd-dim", type=int, default=64)

    parser.add_argument("--disable-transformer", action="store_true")
    parser.add_argument("--transformer-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--transformer-batch-size", type=int, default=32)

    parser.add_argument("--skip-node2vec", action="store_true")
    parser.add_argument("--node2vec-dimensions", type=int, default=64)
    parser.add_argument("--node2vec-walk-length", type=int, default=20)
    parser.add_argument("--node2vec-num-walks", type=int, default=6)
    parser.add_argument("--node2vec-window", type=int, default=5)
    parser.add_argument("--node2vec-epochs", type=int, default=5)
    parser.add_argument("--node2vec-return-p", type=float, default=1.0)
    parser.add_argument("--node2vec-inout-q", type=float, default=2.0)

    parser.add_argument("--skip-gnn", action="store_true")
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-epochs", type=int, default=50)
    parser.add_argument("--gnn-patience", type=int, default=10)
    parser.add_argument("--gnn-learning-rate", type=float, default=1e-2)
    parser.add_argument("--gnn-weight-decay", type=float, default=5e-2)
    parser.add_argument("--gnn-dropout", type=float, default=0.1)

    parser.add_argument("--visualization-sample-size", type=int, default=3000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")

    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "visualize", "report", "run-all"):
        subparsers.add_parser(name)
    return parser


def make_config(args: argparse.Namespace) -> ProjectConfig:
    return ProjectConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_graph_users=args.max_graph_users,
        max_tweets_per_user=args.max_tweets_per_user,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_min_df=args.tfidf_min_df,
        dense_text_max_features=args.dense_text_max_features,
        dense_text_svd_dim=args.dense_text_svd_dim,
        use_transformer=not args.disable_transformer,
        transformer_model_name=args.transformer_model_name,
        transformer_batch_size=args.transformer_batch_size,
        run_node2vec=not args.skip_node2vec,
        node2vec_dimensions=args.node2vec_dimensions,
        node2vec_walk_length=args.node2vec_walk_length,
        node2vec_num_walks=args.node2vec_num_walks,
        node2vec_window=args.node2vec_window,
        node2vec_epochs=args.node2vec_epochs,
        node2vec_return_p=args.node2vec_return_p,
        node2vec_inout_q=args.node2vec_inout_q,
        run_gnn=not args.skip_gnn,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_epochs=args.gnn_epochs,
        gnn_patience=args.gnn_patience,
        gnn_learning_rate=args.gnn_learning_rate,
        gnn_weight_decay=args.gnn_weight_decay,
        gnn_dropout=args.gnn_dropout,
        visualization_sample_size=args.visualization_sample_size,
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
    config.ensure_directories()

    if args.command == "prepare":
        prepare_dataset(config)
    elif args.command == "train":
        run_experiments(config)
    elif args.command == "visualize":
        generate_visualizations(config)
    elif args.command == "report":
        generate_report(config)
    elif args.command == "run-all":
        prepare_dataset(config)
        run_experiments(config)
        generate_visualizations(config)
        generate_report(config)
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")
