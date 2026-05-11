from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ProjectConfig
from .baselines import run_experiments
from .data import prepare_dataset
from .reporting import generate_report
from .visualization import generate_visualizations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TwiBot-20 bot detection research pipeline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing edge.csv/node.json/split.csv/label.csv, or its parent directory that contains raw/.",
    )
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
    parser.add_argument("--only-botdgt", action="store_true", help="Run only the BotDGT GNN model")
    parser.add_argument("--only-tign", action="store_true", help="Run only the TIGN GNN model")
    parser.add_argument("--only-tignv2", action="store_true", help="Run only the TIGN-v2 GNN model")
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-epochs", type=int, default=50)
    parser.add_argument("--gnn-patience", type=int, default=10)
    parser.add_argument("--gnn-learning-rate", type=float, default=1e-3)
    parser.add_argument("--gnn-weight-decay", type=float, default=5e-4)
    parser.add_argument("--gnn-dropout", type=float, default=0.1)
    parser.add_argument("--botsai-invariant-weight", type=float, default=0.05)
    parser.add_argument("--botsai-attention-heads", type=int, default=4)
    parser.add_argument("--botdgt-snapshot-count", type=int, default=8)
    parser.add_argument("--botdgt-min-keep-ratio", type=float, default=0.15)
    parser.add_argument(
        "--botdgt-temporal-module",
        type=str,
        default="attention",
        choices=("attention", "gru", "lstm"),
    )
    parser.add_argument("--botdgt-temporal-heads", type=int, default=4)
    parser.add_argument("--botdgt-temporal-smoothness-weight", type=float, default=0.05)
    parser.add_argument("--botdgt-temporal-consistency-weight", type=float, default=0.03)
    parser.add_argument("--botdgt-structural-lr", type=float, default=1e-4)
    parser.add_argument("--botdgt-temporal-lr", type=float, default=1e-5)
    parser.add_argument("--botdgt-structural-dropout", type=float, default=0.0)
    parser.add_argument("--botdgt-temporal-dropout", type=float, default=0.5)
    parser.add_argument("--botdgt-weight-decay", type=float, default=1e-2)
    parser.add_argument("--botdgt-loss-coefficient", type=float, default=1.1)
    parser.add_argument("--botdgt-epochs", type=int, default=20)
    parser.add_argument(
        "--botdgt-ablation",
        type=str,
        default="full",
        choices=("full", "no_profile", "no_text", "no_graph", "all"),
        help="BotDGT modality ablation mode",
    )
    # TIGN-v2 独立参数
    parser.add_argument("--tignv2-epochs", type=int, default=80)
    parser.add_argument("--tignv2-patience", type=int, default=20)
    parser.add_argument("--tignv2-structural-lr", type=float, default=3e-4)
    parser.add_argument("--tignv2-temporal-lr", type=float, default=1e-4)
    parser.add_argument("--tignv2-weight-decay", type=float, default=5e-3)
    parser.add_argument("--tignv2-loss-coefficient", type=float, default=1.05)
    parser.add_argument("--tignv2-structural-dropout", type=float, default=0.1)
    parser.add_argument("--tignv2-temporal-dropout", type=float, default=0.3)
    parser.add_argument("--tignv2-embedding-dropout", type=float, default=0.3)
    parser.add_argument("--tignv2-structural-heads", type=int, default=4)
    parser.add_argument("--tignv2-temporal-heads", type=int, default=4)
    parser.add_argument("--tignv2-batch-size", type=int, default=64)
    parser.add_argument("--tignv2-interval", type=str, default="six_months")
    parser.add_argument("--tign-num-age-buckets", type=int, default=3)
    parser.add_argument("--tign-intra-class-weight", type=float, default=0.02)
    parser.add_argument("--tignv2-cross-modal-weight", type=float, default=0.0)
    parser.add_argument("--tignv2-temporal-invariance-weight", type=float, default=0.0)
    parser.add_argument("--tignv2-specific-decorr-weight", type=float, default=0.0)

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
        botsai_invariant_weight=args.botsai_invariant_weight,
        botsai_attention_heads=args.botsai_attention_heads,
        botdgt_snapshot_count=args.botdgt_snapshot_count,
        botdgt_min_keep_ratio=args.botdgt_min_keep_ratio,
        botdgt_temporal_module=args.botdgt_temporal_module,
        botdgt_temporal_heads=args.botdgt_temporal_heads,
        botdgt_temporal_smoothness_weight=args.botdgt_temporal_smoothness_weight,
        botdgt_temporal_consistency_weight=args.botdgt_temporal_consistency_weight,
        botdgt_structural_lr=args.botdgt_structural_lr,
        botdgt_temporal_lr=args.botdgt_temporal_lr,
        botdgt_structural_dropout=args.botdgt_structural_dropout,
        botdgt_temporal_dropout=args.botdgt_temporal_dropout,
        botdgt_weight_decay=args.botdgt_weight_decay,
        botdgt_loss_coefficient=args.botdgt_loss_coefficient,
        botdgt_epochs=args.botdgt_epochs,
        botdgt_ablation=args.botdgt_ablation,
        tignv2_epochs=args.tignv2_epochs,
        tignv2_patience=args.tignv2_patience,
        tignv2_structural_lr=args.tignv2_structural_lr,
        tignv2_temporal_lr=args.tignv2_temporal_lr,
        tignv2_weight_decay=args.tignv2_weight_decay,
        tignv2_loss_coefficient=args.tignv2_loss_coefficient,
        tignv2_structural_dropout=args.tignv2_structural_dropout,
        tignv2_temporal_dropout=args.tignv2_temporal_dropout,
        tignv2_embedding_dropout=args.tignv2_embedding_dropout,
        tignv2_structural_heads=args.tignv2_structural_heads,
        tignv2_temporal_heads=args.tignv2_temporal_heads,
        tignv2_batch_size=args.tignv2_batch_size,
        tignv2_interval=args.tignv2_interval,
        tign_num_age_buckets=args.tign_num_age_buckets,
        tign_intra_class_weight=args.tign_intra_class_weight,
        tignv2_cross_modal_weight=args.tignv2_cross_modal_weight,
        tignv2_temporal_invariance_weight=args.tignv2_temporal_invariance_weight,
        tignv2_specific_decorr_weight=args.tignv2_specific_decorr_weight,
        visualization_sample_size=args.visualization_sample_size,
        random_state=args.random_state,
    )


def main() -> None:
    parser = build_parser()
    argv = _normalize_command_first_args(sys.argv[1:])
    args = parser.parse_args(argv)

    config = make_config(args)
    config.ensure_directories()
    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers = [logging.StreamHandler(), logging.FileHandler(config.logs_dir / f"{args.command}.log", encoding="utf-8")]
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)

    import os as _os
    if getattr(args, "only_botdgt", False):
        _os.environ["ONLY_BOTDGT"] = "1"
    elif getattr(args, "only_tignv2", False):
        _os.environ["ONLY_TIGNV2"] = "1"
    elif getattr(args, "only_tign", False):
        _os.environ["ONLY_TIGN"] = "1"

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


def _normalize_command_first_args(argv: list[str]) -> list[str]:
    commands = {"prepare", "train", "visualize", "report", "run-all"}
    if argv and argv[0] in commands:
        return [*argv[1:], argv[0]]
    return argv
