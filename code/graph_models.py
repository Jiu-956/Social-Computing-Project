from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from torch_geometric.nn import GATConv, GCNConv, RGCNConv

    try:
        from torch_geometric.nn import TransformerConv

        TRANSFORMER_CONV_AVAILABLE = True
    except Exception:  # pragma: no cover - version dependent
        TransformerConv = None
        TRANSFORMER_CONV_AVAILABLE = False

    TORCH_GEOMETRIC_AVAILABLE = True
    TORCH_GEOMETRIC_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on local environment
    GATConv = None
    GCNConv = None
    RGCNConv = None
    TransformerConv = None
    TRANSFORMER_CONV_AVAILABLE = False
    TORCH_GEOMETRIC_AVAILABLE = False
    TORCH_GEOMETRIC_IMPORT_ERROR = exc

from .config import ProjectConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GNNResult:
    metrics_rows: list[dict[str, float | str]]
    predictions: pd.DataFrame
    best_val_f1: float
    artifact_path: Path


class _FeatureTextGraphBase(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        description_dim: int,
        tweet_dim: int,
        num_prop_dim: int,
        cat_prop_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        block_dim = max(4, hidden_dim // 4)
        self.description_mlp = nn.Sequential(nn.Linear(description_dim, block_dim), nn.LeakyReLU())
        self.tweet_mlp = nn.Sequential(nn.Linear(tweet_dim, block_dim), nn.LeakyReLU())
        self.num_mlp = nn.Sequential(nn.Linear(num_prop_dim, block_dim), nn.LeakyReLU())
        self.cat_mlp = nn.Sequential(nn.Linear(cat_prop_dim, block_dim), nn.LeakyReLU())
        self.input_mlp = nn.Sequential(nn.Linear(block_dim * 4, hidden_dim), nn.LeakyReLU())
        self.output_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU())
        self.output_head = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def encode_inputs(
        self,
        description: torch.Tensor,
        tweet: torch.Tensor,
        num_prop: torch.Tensor,
        cat_prop: torch.Tensor,
    ) -> torch.Tensor:
        d, t, n, c = self.encode_modalities(description, tweet, num_prop, cat_prop)
        x = torch.cat((d, t, n, c), dim=1)
        x = self.dropout(x)
        return self.input_mlp(x)

    def encode_modalities(
        self,
        description: torch.Tensor,
        tweet: torch.Tensor,
        num_prop: torch.Tensor,
        cat_prop: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.description_mlp(description)
        t = self.tweet_mlp(tweet)
        n = self.num_mlp(num_prop)
        c = self.cat_mlp(cat_prop)
        return d, t, n, c


class FeatureTextGraphGCN(_FeatureTextGraphBase):
    def __init__(
        self,
        hidden_dim: int,
        description_dim: int,
        tweet_dim: int,
        num_prop_dim: int,
        cat_prop_dim: int,
        dropout: float,
    ) -> None:
        super().__init__(hidden_dim, description_dim, tweet_dim, num_prop_dim, cat_prop_dim, dropout)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

    def forward(
        self,
        description: torch.Tensor,
        tweet: torch.Tensor,
        num_prop: torch.Tensor,
        cat_prop: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.encode_inputs(description, tweet, num_prop, cat_prop)
        x = self.gcn1(x, edge_index)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)
        x = self.output_mlp(x)
        return self.output_head(x)


class FeatureTextGraphGAT(_FeatureTextGraphBase):
    def __init__(
        self,
        hidden_dim: int,
        description_dim: int,
        tweet_dim: int,
        num_prop_dim: int,
        cat_prop_dim: int,
        dropout: float,
    ) -> None:
        super().__init__(hidden_dim, description_dim, tweet_dim, num_prop_dim, cat_prop_dim, dropout)
        self.gat1 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout)

    def forward(
        self,
        description: torch.Tensor,
        tweet: torch.Tensor,
        num_prop: torch.Tensor,
        cat_prop: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.encode_inputs(description, tweet, num_prop, cat_prop)
        x = self.gat1(x, edge_index)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = self.output_mlp(x)
        return self.output_head(x)


class FeatureTextGraphBotRGCN(_FeatureTextGraphBase):
    def __init__(
        self,
        hidden_dim: int,
        description_dim: int,
        tweet_dim: int,
        num_prop_dim: int,
        cat_prop_dim: int,
        dropout: float,
        relation_count: int,
    ) -> None:
        super().__init__(hidden_dim, description_dim, tweet_dim, num_prop_dim, cat_prop_dim, dropout)
        self.rgcn1 = RGCNConv(hidden_dim, hidden_dim, num_relations=relation_count)
        self.rgcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations=relation_count)

    def forward(
        self,
        description: torch.Tensor,
        tweet: torch.Tensor,
        num_prop: torch.Tensor,
        cat_prop: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_type is None:
            raise ValueError("BotRGCN requires edge_type.")
        x = self.encode_inputs(description, tweet, num_prop, cat_prop)
        x = self.rgcn1(x, edge_index, edge_type)
        x = self.dropout(x)
        x = self.rgcn2(x, edge_index, edge_type)
        x = self.output_mlp(x)
        return self.output_head(x)


def _compatible_attention_heads(hidden_dim: int, preferred: int) -> int:
    preferred = max(1, int(preferred))
    for candidate in sorted(set([preferred, 8, 4, 2, 1]), reverse=True):
        if candidate <= hidden_dim and hidden_dim % candidate == 0:
            return candidate
    return 1


class FeatureTextGraphBotSAI(_FeatureTextGraphBase):
    def __init__(
        self,
        hidden_dim: int,
        description_dim: int,
        tweet_dim: int,
        num_prop_dim: int,
        cat_prop_dim: int,
        dropout: float,
        relation_count: int,
        invariant_weight: float,
        attention_heads: int,
    ) -> None:
        super().__init__(hidden_dim, description_dim, tweet_dim, num_prop_dim, cat_prop_dim, dropout)
        block_dim = max(4, hidden_dim // 4)
        channel_dim = block_dim * 2
        channel_heads = _compatible_attention_heads(channel_dim, attention_heads)
        graph_heads = _compatible_attention_heads(hidden_dim, attention_heads)

        self.invariant_weight = float(max(0.0, invariant_weight))
        self.invariant_projectors = nn.ModuleList([nn.Linear(block_dim, block_dim) for _ in range(4)])
        self.specific_projectors = nn.ModuleList([nn.Linear(block_dim, block_dim) for _ in range(4)])
        self.channel_attention = nn.MultiheadAttention(
            embed_dim=channel_dim,
            num_heads=channel_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.channel_norm = nn.LayerNorm(channel_dim)
        self.channel_to_hidden = nn.Sequential(nn.Linear(channel_dim, hidden_dim), nn.LeakyReLU())
        self.relation_embedding = nn.Embedding(relation_count, hidden_dim)
        self.structural_layer1 = TransformerConv(
            hidden_dim,
            hidden_dim // graph_heads,
            heads=graph_heads,
            concat=True,
            edge_dim=hidden_dim,
            dropout=dropout,
        )
        self.structural_layer2 = TransformerConv(
            hidden_dim,
            hidden_dim // graph_heads,
            heads=graph_heads,
            concat=True,
            edge_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        description: torch.Tensor,
        tweet: torch.Tensor,
        num_prop: torch.Tensor,
        cat_prop: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        modalities = self.encode_modalities(description, tweet, num_prop, cat_prop)

        invariant_parts: list[torch.Tensor] = []
        specific_parts: list[torch.Tensor] = []
        channels: list[torch.Tensor] = []
        for index, modality in enumerate(modalities):
            invariant = torch.tanh(self.invariant_projectors[index](modality))
            specific = F.leaky_relu(self.specific_projectors[index](modality))
            invariant_parts.append(invariant)
            specific_parts.append(specific)
            channels.append(torch.cat((invariant, specific), dim=1))

        channel_tensor = torch.stack(channels, dim=1)
        channel_tensor = self.dropout(channel_tensor)
        attended, _ = self.channel_attention(channel_tensor, channel_tensor, channel_tensor)
        channel_tensor = self.channel_norm(attended + channel_tensor)
        fused = self.channel_to_hidden(channel_tensor.mean(dim=1))

        if edge_type is None:
            edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
        bounded_edge_type = edge_type.long().clamp(min=0, max=self.relation_embedding.num_embeddings - 1)
        edge_attr = self.relation_embedding(bounded_edge_type)

        x = self.structural_layer1(fused, edge_index, edge_attr=edge_attr)
        x = self.dropout(F.leaky_relu(x))
        x = self.structural_layer2(x, edge_index, edge_attr=edge_attr)
        x = self.output_mlp(x)
        logits = self.output_head(x)

        inv_stack = torch.stack(invariant_parts, dim=1)
        inv_center = inv_stack.mean(dim=1, keepdim=True)
        invariance_loss = ((inv_stack - inv_center) ** 2).mean()
        specific_overlap = torch.tensor(0.0, device=logits.device)
        pair_count = 0
        for left in range(len(specific_parts)):
            for right in range(left + 1, len(specific_parts)):
                specific_overlap = specific_overlap + F.cosine_similarity(
                    specific_parts[left],
                    specific_parts[right],
                    dim=1,
                ).abs().mean()
                pair_count += 1
        if pair_count > 0:
            specific_overlap = specific_overlap / pair_count
        aux_loss = self.invariant_weight * (invariance_loss + 0.5 * specific_overlap)
        return logits, aux_loss


class FeatureTextGraphBotDGT(_FeatureTextGraphBase):
    def __init__(
        self,
        hidden_dim: int,
        description_dim: int,
        tweet_dim: int,
        num_prop_dim: int,
        cat_prop_dim: int,
        dropout: float,
        temporal_module: str,
        temporal_heads: int,
        temporal_smoothness_weight: float,
        temporal_consistency_weight: float,
    ) -> None:
        super().__init__(hidden_dim, description_dim, tweet_dim, num_prop_dim, cat_prop_dim, dropout)
        graph_heads = _compatible_attention_heads(hidden_dim, temporal_heads)
        temporal_heads = _compatible_attention_heads(hidden_dim, temporal_heads)
        self.temporal_module = str(temporal_module).lower()
        self.temporal_smoothness_weight = float(max(0.0, temporal_smoothness_weight))
        self.temporal_consistency_weight = float(max(0.0, temporal_consistency_weight))

        self.structural_layer1 = TransformerConv(
            hidden_dim,
            hidden_dim // graph_heads,
            heads=graph_heads,
            concat=True,
            dropout=dropout,
        )
        self.structural_layer2 = TransformerConv(
            hidden_dim,
            hidden_dim // graph_heads,
            heads=graph_heads,
            concat=True,
            dropout=dropout,
        )
        self.cluster_position_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh())
        self.bidirectional_position_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh())
        self.edge_density_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh())
        self.keep_ratio_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh())

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=temporal_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.temporal_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.temporal_norm1 = nn.LayerNorm(hidden_dim)
        self.temporal_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal_norm2 = nn.LayerNorm(hidden_dim)
        self.temporal_transition = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        description: torch.Tensor,
        tweet: torch.Tensor,
        num_prop: torch.Tensor,
        cat_prop: torch.Tensor,
        edge_index: dict[str, Any] | torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(edge_index, dict):
            raise ValueError("BotDGT expects a dynamic snapshot bundle.")

        snapshot_edge_indices: list[torch.Tensor] = edge_index.get("edge_indices", [])
        clustering = edge_index.get("clustering")
        bidirectional_ratio = edge_index.get("bidirectional_ratio")
        edge_density = edge_index.get("edge_density")
        keep_ratio = edge_index.get("keep_ratio")
        if not snapshot_edge_indices or clustering is None or bidirectional_ratio is None:
            raise ValueError("BotDGT snapshot bundle is incomplete.")
        if edge_density is None:
            edge_density = torch.zeros_like(clustering)
        if keep_ratio is None:
            keep_ratio = torch.zeros_like(clustering)

        base_x = self.encode_inputs(description, tweet, num_prop, cat_prop)
        structural_states: list[torch.Tensor] = []
        for snapshot_edges in snapshot_edge_indices:
            snapshot_edges = snapshot_edges.to(base_x.device)
            structural = self.structural_layer1(base_x, snapshot_edges)
            structural = self.dropout(F.leaky_relu(structural))
            structural = self.structural_layer2(structural, snapshot_edges)
            structural_states.append(F.leaky_relu(structural + base_x))

        temporal_inputs = torch.stack(structural_states, dim=1)
        cluster_signal = self.cluster_position_encoder(clustering.transpose(0, 1).to(base_x.device))
        bidirectional_signal = self.bidirectional_position_encoder(bidirectional_ratio.transpose(0, 1).to(base_x.device))
        density_signal = self.edge_density_encoder(edge_density.transpose(0, 1).to(base_x.device))
        keep_ratio_signal = self.keep_ratio_encoder(keep_ratio.transpose(0, 1).to(base_x.device))
        temporal_inputs = temporal_inputs + cluster_signal + bidirectional_signal + density_signal + keep_ratio_signal

        if self.temporal_module == "gru":
            temporal_output, _ = self.temporal_gru(temporal_inputs)
            temporal_output = self.temporal_norm2(temporal_output + temporal_inputs)
        elif self.temporal_module == "lstm":
            temporal_output, _ = self.temporal_lstm(temporal_inputs)
            temporal_output = self.temporal_norm2(temporal_output + temporal_inputs)
        else:
            sequence_length = temporal_inputs.shape[1]
            causal_mask = torch.triu(
                torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=temporal_inputs.device),
                diagonal=1,
            )
            attended, _ = self.temporal_attention(
                temporal_inputs,
                temporal_inputs,
                temporal_inputs,
                attn_mask=causal_mask,
            )
            temporal_output = self.temporal_norm1(attended + temporal_inputs)
            feedforward = self.temporal_ff(temporal_output)
            temporal_output = self.temporal_norm2(feedforward + temporal_output)

        x = self.output_mlp(temporal_output[:, -1, :])
        logits = self.output_head(x)

        if temporal_output.shape[1] > 1:
            temporal_deltas = temporal_output[:, 1:, :] - temporal_output[:, :-1, :]
            smoothness_loss = temporal_deltas.pow(2).mean()

            predicted_next = self.temporal_transition(temporal_output[:, :-1, :])
            consistency_loss = F.smooth_l1_loss(predicted_next, temporal_output[:, 1:, :].detach())

            flat_temporal = temporal_output.reshape(-1, temporal_output.shape[-1])
            step_logits = self.output_head(self.output_mlp(flat_temporal)).reshape(
                temporal_output.shape[0],
                temporal_output.shape[1],
                2,
            )
            step_probs = torch.softmax(step_logits, dim=-1)[..., 1]
            probability_drift = (step_probs[:, 1:] - step_probs[:, :-1]).abs().mean()
        else:
            smoothness_loss = torch.tensor(0.0, device=logits.device)
            consistency_loss = torch.tensor(0.0, device=logits.device)
            probability_drift = torch.tensor(0.0, device=logits.device)

        aux_loss = self.temporal_smoothness_weight * smoothness_loss
        aux_loss = aux_loss + self.temporal_consistency_weight * (consistency_loss + 0.5 * probability_drift)
        return logits, aux_loss


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
    if not TORCH_GEOMETRIC_AVAILABLE:
        LOGGER.warning(
            "Skipping graph neural experiments because torch_geometric is unavailable: %s",
            TORCH_GEOMETRIC_IMPORT_ERROR,
        )
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

    outputs = []
    model_specs: list[tuple[str, nn.Module, dict[str, Any] | torch.Tensor, torch.Tensor | None]] = [
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
        dynamic_bundle = _build_botdgt_snapshot_bundle(
            users=users,
            graph_edges=graph_edges,
            id_to_index=id_to_index,
            snapshot_count=config.botdgt_snapshot_count,
            min_keep_ratio=config.botdgt_min_keep_ratio,
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
                        temporal_module=config.botdgt_temporal_module,
                        temporal_heads=config.botdgt_temporal_heads,
                        temporal_smoothness_weight=config.botdgt_temporal_smoothness_weight,
                        temporal_consistency_weight=config.botdgt_temporal_consistency_weight,
                    ),
                    dynamic_bundle,
                    None,
                ),
            ]
        )
    else:
        LOGGER.warning(
            "TransformerConv is unavailable, skipping feature_text_graph_botsai and feature_text_graph_botdgt."
        )

    for name, model, edge_index, edge_type in model_specs:
        LOGGER.info("Running graph neural model: %s", name)
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
            )
        )
    return outputs


def _train_gnn_model(
    config: ProjectConfig,
    name: str,
    users: pd.DataFrame,
    model: nn.Module,
    description_tensor: torch.Tensor,
    tweet_tensor: torch.Tensor,
    num_prop_tensor: torch.Tensor,
    cat_prop_tensor: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    test_indices: torch.Tensor,
    edge_index: dict[str, Any] | torch.Tensor,
    edge_type: torch.Tensor | None,
) -> GNNResult:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.gnn_learning_rate,
        weight_decay=config.gnn_weight_decay,
    )
    train_labels = labels[train_indices].numpy()
    class_counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))

    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0
    patience_left = config.gnn_patience

    for _ in range(config.gnn_epochs):
        model.train()
        optimizer.zero_grad()
        raw_output = model(description_tensor, tweet_tensor, num_prop_tensor, cat_prop_tensor, edge_index, edge_type)
        logits, aux_loss = _split_model_output(raw_output)
        loss = criterion(logits[train_indices], labels[train_indices])
        if aux_loss is not None:
            loss = loss + aux_loss
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            raw_output = model(description_tensor, tweet_tensor, num_prop_tensor, cat_prop_tensor, edge_index, edge_type)
            logits, _ = _split_model_output(raw_output)
            val_probs = torch.softmax(logits[val_indices], dim=1)[:, 1].cpu().numpy()
            val_preds = (val_probs >= 0.5).astype(int)
            val_true = labels[val_indices].cpu().numpy()
            _, _, val_f1, _ = precision_recall_fscore_support(val_true, val_preds, average="binary", zero_division=0)

        if float(val_f1) > best_val_f1:
            best_val_f1 = float(val_f1)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = config.gnn_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        raw_output = model(description_tensor, tweet_tensor, num_prop_tensor, cat_prop_tensor, edge_index, edge_type)
        logits, _ = _split_model_output(raw_output)

    metrics_rows: list[dict[str, float | str]] = []
    prediction_rows: list[dict[str, float | str | int]] = []
    for split_name, indices in (("val", val_indices), ("test", test_indices)):
        split_probs = torch.softmax(logits[indices], dim=1)[:, 1].cpu().numpy()
        split_preds = (split_probs >= 0.5).astype(int)
        split_true = labels[indices].cpu().numpy()
        metrics_rows.append(
            {
                "experiment": name,
                "family": "feature_text_graph",
                "split": split_name,
                **_compute_metrics(split_true, split_preds, split_probs),
            }
        )
        user_subset = users.iloc[indices.cpu().numpy()]
        for user_id, true_label, pred_label, probability in zip(
            user_subset["user_id"].tolist(),
            split_true.tolist(),
            split_preds.tolist(),
            split_probs.tolist(),
        ):
            prediction_rows.append(
                {
                    "experiment": name,
                    "family": "feature_text_graph",
                    "split": split_name,
                    "user_id": user_id,
                    "true_label": int(true_label),
                    "pred_label": int(pred_label),
                    "bot_probability": float(probability),
                }
            )

    artifact_path = config.models_dir / f"{name}.pt"
    torch.save({"state_dict": model.state_dict(), "best_val_f1": best_val_f1}, artifact_path)
    return GNNResult(
        metrics_rows=metrics_rows,
        predictions=pd.DataFrame(prediction_rows),
        best_val_f1=best_val_f1,
        artifact_path=artifact_path,
    )


def _split_model_output(model_output: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(model_output, tuple):
        logits = model_output[0]
        aux_loss = model_output[1] if len(model_output) > 1 else None
        if isinstance(aux_loss, torch.Tensor):
            return logits, aux_loss
        return logits, None
    return model_output, None


def _scaled_tensor(users: pd.DataFrame, columns: list[str]) -> torch.Tensor:
    frame = users.reindex(columns=columns, fill_value=0.0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    matrix = frame.to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    matrix = scaler.fit_transform(matrix).astype(np.float32)
    return torch.tensor(matrix, dtype=torch.float32)


def _build_combined_edge_index(id_to_index: dict[str, int], graph_edges: pd.DataFrame) -> torch.Tensor:
    rows: list[int] = []
    cols: list[int] = []
    for row in graph_edges.itertuples(index=False):
        source = id_to_index.get(row.source_id)
        target = id_to_index.get(row.target_id)
        if source is None or target is None:
            continue
        rows.extend([source, target])
        cols.extend([target, source])
    if not rows:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long)


def _build_relation_graph(id_to_index: dict[str, int], graph_edges: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[int] = []
    cols: list[int] = []
    edge_types: list[int] = []
    relation_to_id = {"follow": 0, "friend": 1}
    for row in graph_edges.itertuples(index=False):
        source = id_to_index.get(row.source_id)
        target = id_to_index.get(row.target_id)
        relation_id = relation_to_id.get(row.relation)
        if source is None or target is None or relation_id is None:
            continue
        rows.append(source)
        cols.append(target)
        edge_types.append(relation_id)
    if not rows:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long), torch.tensor(edge_types, dtype=torch.long)


def _build_botdgt_snapshot_bundle(
    users: pd.DataFrame,
    graph_edges: pd.DataFrame,
    id_to_index: dict[str, int],
    snapshot_count: int,
    min_keep_ratio: float,
) -> dict[str, Any]:
    snapshot_count = max(3, int(snapshot_count))
    node_count = len(users)
    min_keep_ratio = float(np.clip(min_keep_ratio, 0.05, 0.9))

    if "account_age_days" in users.columns:
        account_age = pd.to_numeric(users["account_age_days"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    else:
        account_age = np.zeros((node_count,), dtype=np.float32)

    keep_ratios = np.linspace(min_keep_ratio, 1.0, snapshot_count)
    quantiles = np.clip(1.0 - keep_ratios, 0.0, 1.0)
    thresholds = [float(np.quantile(account_age, quantile)) for quantile in quantiles]

    source_indices: list[int] = []
    target_indices: list[int] = []
    for row in graph_edges.itertuples(index=False):
        source = id_to_index.get(row.source_id)
        target = id_to_index.get(row.target_id)
        if source is None or target is None:
            continue
        source_indices.append(source)
        target_indices.append(target)

    if not source_indices:
        return {
            "edge_indices": [torch.empty((2, 0), dtype=torch.long) for _ in range(snapshot_count)],
            "clustering": torch.zeros((snapshot_count, node_count, 1), dtype=torch.float32),
            "bidirectional_ratio": torch.zeros((snapshot_count, node_count, 1), dtype=torch.float32),
            "edge_density": torch.zeros((snapshot_count, node_count, 1), dtype=torch.float32),
            "keep_ratio": torch.tensor(keep_ratios, dtype=torch.float32).view(snapshot_count, 1, 1).repeat(1, node_count, 1),
        }

    source_array = np.asarray(source_indices, dtype=np.int64)
    target_array = np.asarray(target_indices, dtype=np.int64)

    edge_indices: list[torch.Tensor] = []
    clustering_list: list[torch.Tensor] = []
    bidirectional_list: list[torch.Tensor] = []
    density_list: list[torch.Tensor] = []

    previous_mask = np.zeros(source_array.shape[0], dtype=bool)
    denominator = max(1.0, float(node_count * max(node_count - 1, 1)))
    for keep_ratio, threshold in zip(keep_ratios, thresholds, strict=False):
        exists = account_age >= (threshold - 1e-9)
        valid_edges = exists[source_array] & exists[target_array]
        valid_edges = valid_edges | previous_mask
        previous_mask = valid_edges
        snapshot_source = source_array[valid_edges]
        snapshot_target = target_array[valid_edges]

        if snapshot_source.size == 0:
            edge_indices.append(torch.empty((2, 0), dtype=torch.long))
            clustering_list.append(torch.zeros((node_count, 1), dtype=torch.float32))
            bidirectional_list.append(torch.zeros((node_count, 1), dtype=torch.float32))
            density_list.append(torch.zeros((node_count, 1), dtype=torch.float32))
            continue

        edge_indices.append(torch.tensor(np.stack([snapshot_source, snapshot_target], axis=0), dtype=torch.long))

        undirected_degree = np.bincount(
            np.concatenate([snapshot_source, snapshot_target]),
            minlength=node_count,
        ).astype(np.float32)
        max_degree = max(1.0, float(undirected_degree.max()))
        clustering_proxy = (undirected_degree / max_degree).reshape(-1, 1)
        clustering_list.append(torch.tensor(clustering_proxy, dtype=torch.float32))

        out_degree = np.bincount(snapshot_source, minlength=node_count).astype(np.float32)
        reciprocal = np.zeros((node_count,), dtype=np.float32)
        directed_edges = set(zip(snapshot_source.tolist(), snapshot_target.tolist()))
        for source, target in directed_edges:
            if (target, source) in directed_edges:
                reciprocal[source] += 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(reciprocal, out_degree, out=np.zeros_like(reciprocal), where=out_degree > 0)
        bidirectional_list.append(torch.tensor(ratio.reshape(-1, 1), dtype=torch.float32))

        density_value = float(snapshot_source.size / denominator)
        density_list.append(torch.full((node_count, 1), density_value, dtype=torch.float32))

    return {
        "edge_indices": edge_indices,
        "clustering": torch.stack(clustering_list, dim=0),
        "bidirectional_ratio": torch.stack(bidirectional_list, dim=0),
        "edge_density": torch.stack(density_list, dim=0),
        "keep_ratio": torch.tensor(keep_ratios, dtype=torch.float32).view(snapshot_count, 1, 1).repeat(1, node_count, 1),
    }


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc_roc = roc_auc_score(y_true, probabilities)
    except ValueError:
        auc_roc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(auc_roc),
    }
