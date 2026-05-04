from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.nn import TransformerConv
    TRANSFORMER_CONV_AVAILABLE = True
except Exception:  # pragma: no cover
    TransformerConv = None
    TRANSFORMER_CONV_AVAILABLE = False

from ..config import ProjectConfig
from .builders import (
    _build_age_relation_graph,
    _build_botdgt_snapshot_bundle,
    _build_combined_edge_index,
    _build_relation_graph,
)
from .models import (
    FeatureTextGraphBotDGT,
    FeatureTextGraphBotRGCN,
    FeatureTextGraphBotSAI,
    FeatureTextGraphGAT,
    FeatureTextGraphGCN,
    FeatureTextGraphTIGN,
)
from .train import _scaled_tensor, _train_gnn_model, GNNResult, ModelTrainConfig

LOGGER = logging.getLogger(__name__)


def run_graph_neural_models(
    config: ProjectConfig,
    users: pd.DataFrame,
    graph_edges: pd.DataFrame,
    description_columns: list[str],
    tweet_columns: list[str],
    num_property_columns: list[str],
    cat_property_columns: list[str],
) -> list[GNNResult]:
    if users.empty:
        return []

    users = users.copy()
    users[description_columns + tweet_columns + num_property_columns + cat_property_columns] = (
        users[description_columns + tweet_columns + num_property_columns + cat_property_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    description_tensor = _scaled_tensor(users, description_columns)
    tweet_tensor = _scaled_tensor(users, tweet_columns)
    num_prop_tensor = _scaled_tensor(users, num_property_columns)
    cat_prop_tensor = torch.tensor(users[cat_property_columns].to_numpy(dtype=np.float32), dtype=torch.float32)

    labels = torch.tensor(users["label_id"].clip(lower=0).to_numpy(dtype=np.int64), dtype=torch.long)
    train_indices = torch.tensor(np.flatnonzero(((users["split"] == "train") & (users["label_id"] >= 0)).to_numpy()), dtype=torch.long)
    val_indices = torch.tensor(np.flatnonzero(((users["split"] == "val") & (users["label_id"] >= 0)).to_numpy()), dtype=torch.long)
    test_indices = torch.tensor(np.flatnonzero(((users["split"] == "test") & (users["label_id"] >= 0)).to_numpy()), dtype=torch.long)

    user_ids = users["user_id"].tolist()
    id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    combined_edge_index = _build_combined_edge_index(id_to_index, graph_edges)
    relation_edge_index, relation_edge_type = _build_relation_graph(id_to_index, graph_edges)

    num_age_buckets = config.tign_num_age_buckets
    user_age_buckets_np = pd.to_numeric(users["account_age_bucket"], errors="coerce").fillna(1).to_numpy(dtype=np.int64)
    user_age_buckets_tensor = torch.tensor(user_age_buckets_np, dtype=torch.long)
    age_relation_edge_index, age_relation_edge_type = _build_age_relation_graph(
        id_to_index, user_age_buckets_tensor, graph_edges, num_age_buckets=num_age_buckets
    )

    outputs = []
    model_specs: list[tuple[str, Any, Any, Any]] = [
        (
            "feature_text_graph_gcn",
            FeatureTextGraphGCN(
                hidden_dim=config.gnn_hidden_dim,
                description_dim=description_tensor.shape[1],
                tweet_dim=tweet_tensor.shape[1],
                num_prop_dim=num_prop_tensor.shape[1],
                cat_prop_dim=cat_prop_tensor.shape[1],
                dropout=config.gnn_dropout,
            ),
            combined_edge_index,
            None,
        ),
        (
            "feature_text_graph_gat",
            FeatureTextGraphGAT(
                hidden_dim=config.gnn_hidden_dim,
                description_dim=description_tensor.shape[1],
                tweet_dim=tweet_tensor.shape[1],
                num_prop_dim=num_prop_tensor.shape[1],
                cat_prop_dim=cat_prop_tensor.shape[1],
                dropout=config.gnn_dropout,
            ),
            combined_edge_index,
            None,
        ),
        (
            "feature_text_graph_botrgcn",
            FeatureTextGraphBotRGCN(
                hidden_dim=config.gnn_hidden_dim,
                description_dim=description_tensor.shape[1],
                tweet_dim=tweet_tensor.shape[1],
                num_prop_dim=num_prop_tensor.shape[1],
                cat_prop_dim=cat_prop_tensor.shape[1],
                dropout=config.gnn_dropout,
                relation_count=2,
            ),
            relation_edge_index,
            relation_edge_type,
        ),
    ]

    if TRANSFORMER_CONV_AVAILABLE:
        # BotSAI and TIGN use labeled nodes for training
        labeled_user_ids = set(users.loc[users["label_id"] >= 0, "user_id"])
        labeled_mask = labels >= 0
        labeled_indices = torch.nonzero(labeled_mask, as_tuple=False).squeeze(-1)

        # Helper to map global-user-index to within-labeled-subset index
        global_to_labeled = {int(idx): i for i, idx in enumerate(labeled_indices.tolist())}

        # --- BotDGT: use ALL nodes for full-graph structural encoding ---
        dynamic_bundle = _build_botdgt_snapshot_bundle(
            users=users,
            graph_edges=graph_edges,
            id_to_index=id_to_index,
            snapshot_count=config.botdgt_snapshot_count,
            min_keep_ratio=config.botdgt_min_keep_ratio,
        )

        # Train/val/test indices within the labeled subset (re-indexed)
        botdgt_labeled_users = users[users["user_id"].isin(labeled_user_ids)]
        botdgt_train = torch.tensor(
            [global_to_labeled[i] for i, split in enumerate(users["split"])
             if split == "train" and users["label_id"].iloc[i] >= 0],
            dtype=torch.long,
        )
        botdgt_val = torch.tensor(
            [global_to_labeled[i] for i, split in enumerate(users["split"])
             if split == "val" and users["label_id"].iloc[i] >= 0],
            dtype=torch.long,
        )
        botdgt_test = torch.tensor(
            [global_to_labeled[i] for i, split in enumerate(users["split"])
             if split == "test" and users["label_id"].iloc[i] >= 0],
            dtype=torch.long,
        )

        model_specs.extend(
            [
                (
                    "feature_text_graph_botsai",
                    FeatureTextGraphBotSAI(
                        hidden_dim=config.gnn_hidden_dim,
                        description_dim=description_tensor.shape[1],
                        tweet_dim=tweet_tensor.shape[1],
                        num_prop_dim=num_prop_tensor.shape[1],
                        cat_prop_dim=cat_prop_tensor.shape[1],
                        dropout=config.gnn_dropout,
                        relation_count=2,
                        invariant_weight=config.botsai_invariant_weight,
                        attention_heads=config.botsai_attention_heads,
                    ),
                    relation_edge_index,
                    relation_edge_type,
                ),
                (
                    "feature_text_graph_botdgt",
                    FeatureTextGraphBotDGT(
                        hidden_dim=config.gnn_hidden_dim,
                        description_dim=description_tensor.shape[1],
                        tweet_dim=tweet_tensor.shape[1],
                        num_prop_dim=num_prop_tensor.shape[1],
                        cat_prop_dim=cat_prop_tensor.shape[1],
                        dropout=config.gnn_dropout,
                        relation_count=2,
                        invariant_weight=config.botsai_invariant_weight,
                        attention_heads=config.botsai_attention_heads,
                        temporal_module=config.botdgt_temporal_module,
                        temporal_heads=config.botdgt_temporal_heads,
                        temporal_smoothness_weight=config.botdgt_temporal_smoothness_weight,
                        temporal_consistency_weight=config.botdgt_temporal_consistency_weight,
                        structural_dropout=config.botdgt_structural_dropout,
                        temporal_dropout=config.botdgt_temporal_dropout,
                    ),
                    dynamic_bundle,
                    None,
                ),
                (
                    "feature_text_graph_tign",
                    FeatureTextGraphTIGN(
                        hidden_dim=config.gnn_hidden_dim,
                        description_dim=description_tensor.shape[1],
                        tweet_dim=tweet_tensor.shape[1],
                        num_prop_dim=num_prop_tensor.shape[1],
                        cat_prop_dim=cat_prop_tensor.shape[1],
                        dropout=config.gnn_dropout,
                        num_age_buckets=num_age_buckets,
                        relation_count=2,
                        invariant_weight=config.botsai_invariant_weight,
                        intra_class_weight=config.tign_intra_class_weight,
                        attention_heads=config.botsai_attention_heads,
                    ),
                    age_relation_edge_index,
                    age_relation_edge_type,
                ),
                (
                    "feature_text_graph_tignv2",
                    None,
                    None,
                    None,
                ),
            ]
        )
    else:
        LOGGER.warning(
            "TransformerConv is unavailable, skipping feature_text_graph_botsai, feature_text_graph_botdgt and feature_text_graph_tign."
        )

    # Support --only-tign / --only-botdgt / --only-tignv2 flags
    import os as _os
    if _os.environ.get("ONLY_BOTDGT"):
        model_specs = [s for s in model_specs if s[0] == "feature_text_graph_botdgt"]
    elif _os.environ.get("ONLY_TIGNV2"):
        model_specs = [s for s in model_specs if s[0] == "feature_text_graph_tignv2"]
    elif _os.environ.get("ONLY_TIGN"):
        model_specs = [s for s in model_specs if s[0] == "feature_text_graph_tign"]

    # BotDGT uses the new independent module (NeighborLoader + calendar snapshots)
    botdgt_specs = [s for s in model_specs if s[0] == "feature_text_graph_botdgt"]
    tignv2_specs = [s for s in model_specs if s[0] == "feature_text_graph_tignv2"]
    model_specs = [s for s in model_specs if s[0] not in ("feature_text_graph_botdgt", "feature_text_graph_tignv2")]

    for name, _, _, _ in botdgt_specs:
        LOGGER.info("Running graph neural model: %s (new BotDGT module)", name)
        from .botdgt import run_botdgt
        botdgt_result = run_botdgt(config=config)
        from .train import GNNResult
        outputs.append(GNNResult(
            metrics_rows=botdgt_result["metrics_rows"],
            predictions=botdgt_result["predictions"],
            best_val_f1=botdgt_result["best_val_f1"],
            artifact_path=botdgt_result["artifact_path"],
            training_history=botdgt_result["training_history"],
        ))

    for name, _, _, _ in tignv2_specs:
        LOGGER.info("Running graph neural model: %s (TIGN-v2: temporal invariance)", name)
        from .tignv2 import run_tignv2
        tignv2_result = run_tignv2(config=config)
        from .train import GNNResult
        outputs.append(GNNResult(
            metrics_rows=tignv2_result["metrics_rows"],
            predictions=tignv2_result["predictions"],
            best_val_f1=tignv2_result["best_val_f1"],
            artifact_path=tignv2_result["artifact_path"],
            training_history=tignv2_result["training_history"],
        ))

    for name, model, edge_index, edge_type, *rest in model_specs:
        LOGGER.info("Running graph neural model: %s", name)

        train_cfg: ModelTrainConfig | None = None
        if rest:
            train_cfg = rest[0]

        outputs.append(
            _train_gnn_model(
                config=config,
                name=name,
                users=users,
                model=model,
                description_tensor=description_tensor,
                tweet_tensor=tweet_tensor,
                num_prop_tensor=num_prop_tensor,
                cat_prop_tensor=cat_prop_tensor,
                labels=labels,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
                edge_index=edge_index,
                edge_type=edge_type,
                train_cfg=train_cfg,
            )
        )
    return outputs
