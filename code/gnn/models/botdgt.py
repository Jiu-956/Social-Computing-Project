from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import TransformerConv
except Exception:  # pragma: no cover
    TransformerConv = None

from .base import _FeatureTextGraphBase, _compatible_attention_heads


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
