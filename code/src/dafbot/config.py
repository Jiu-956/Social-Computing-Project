from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path or Path("code") / "configs" / "default.yaml").resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    root_dir = config_path.parent.parent
    defaults = {
        "project": {"name": "dynamic_adaptive_fusion_bot_detector", "seed": 42, "device": "cpu"},
        "paths": {
            "data_root": "../data/raw",
            "processed_dir": "../data/processed",
            "snapshot_dir": "../data/processed/dynamic_graph/snapshots",
            "output_dir": "../outputs",
            "checkpoint_dir": "../outputs/checkpoints",
            "log_dir": "../outputs/logs",
            "figure_dir": "../outputs/figures",
            "table_dir": "../outputs/tables",
        },
        "preprocess": {
            "max_tweets_per_user": 20,
            "tweet_separator": " [SEP] ",
            "empty_token": "[EMPTY]",
            "fallback_created_at": "2006-03-01T00:00:00+00:00",
        },
        "text": {
            "encoder_type": "sentence_transformer",
            "fallback_encoder_type": "tfidf_svd",
            "model_name": "all-MiniLM-L6-v2",
            "batch_size": 64,
            "max_features": 20000,
            "svd_dim": 256,
            "min_df": 2,
        },
        "graph": {
            "granularity": "monthly",
            "max_snapshots": None,
            "snapshot_sampling": "uniform",
            "relation_map": {"follow": 0, "friend": 1},
        },
        "model": {
            "d_model": 128,
            "dropout": 0.3,
            "graph_hidden_dim": 128,
            "text_hidden_dim": 256,
            "attr_hidden_dim": 128,
            "temporal_heads": 1,
            "fusion_hidden_dim": 128,
        },
        "training": {
            "epochs": 50,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "patience": 8,
            "lambda_modal": 0.0,
            "lambda_temporal": 0.0,
        },
        "evaluation": {
            "decision_threshold": 0.5,
            "top_temporal_cases": 32,
            "degree_bins": [[0, 5], [6, 20], [21, None]],
        },
    }
    config = _deep_merge(defaults, config)

    path_config = config["paths"]
    resolved_paths = {"root_dir": root_dir}
    for key, value in path_config.items():
        resolved_paths[key] = (root_dir / value).resolve()
    config["paths"] = resolved_paths
    return config


def ensure_directories(config: dict[str, Any]) -> None:
    paths = config["paths"]
    for key in (
        "data_root",
        "processed_dir",
        "snapshot_dir",
        "output_dir",
        "checkpoint_dir",
        "log_dir",
        "figure_dir",
        "table_dir",
    ):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)
