from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import TransformerConv
except Exception:  # pragma: no cover
    TransformerConv = None

from .base import _compatible_attention_heads
from ..builders.graph_structural_layer import GraphStructuralLayer
from ..builders.position_encoding import PositionEncodingClusteringCoefficient, PositionEncodingBidirectionalLinks


class GraphTemporalLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float,
        num_time_steps: int,
        temporal_module_type: str,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.temporal_module_type = temporal_module_type

        self.Q_embedding_weights = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.feedforward_linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.PReLU()
        self.feedforward_linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.num_time_steps = num_time_steps

        self.position_embedding_temporal = nn.Embedding(num_time_steps, hidden_dim)
        self.GRU = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.LSTM = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.init_weights()

    def _rebuild_position_embedding(self, num_time_steps: int, device: torch.device) -> None:
        if self.position_embedding_temporal.num_embeddings != num_time_steps:
            self.position_embedding_temporal = nn.Embedding(num_time_steps, self.hidden_dim).to(device)

    def forward(self, structural_output, position_embedding_clustering_coefficient,
                position_embedding_bidirectional_links_ratio, exist_nodes):
        if self.temporal_module_type == 'gru':
            gru_output, _ = self.GRU(structural_output)
            y = structural_output + gru_output
            y = self.layer_norm(y)
            return self.feed_forward(y)
        elif self.temporal_module_type == 'lstm':
            lstm_output, _ = self.LSTM(structural_output)
            y = structural_output + lstm_output
            y = self.layer_norm(y)
            return self.feed_forward(y)
        else:
            structural_input = structural_output
            B = structural_output.shape[0]
            T = self.num_time_steps
            position_inputs = torch.arange(0, T, device=structural_output.device).reshape(1, -1).repeat(B, 1).long()
            position_embedding_temporal = self.position_embedding_temporal(position_inputs)
            temporal_inputs = (structural_output
                              + position_embedding_temporal
                              + position_embedding_clustering_coefficient
                              + position_embedding_bidirectional_links_ratio)
            temporal_inputs = self.layer_norm(temporal_inputs)
            q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2], [0]))
            k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2], [0]))
            v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2], [0]))
            split_size = int(q.shape[-1] / self.n_heads)
            q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)
            k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)
            v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)
            outputs = torch.matmul(q_, k_.permute(0, 2, 1))
            outputs = outputs / (split_size ** 0.5)
            diag_val = torch.ones(T, T, device=structural_output.device)
            tril = torch.tril(diag_val)
            sequence_mask = tril[None, :, :].repeat(outputs.shape[0], 1, 1)
            total_mask = sequence_mask
            total_mask = total_mask.float()
            padding = torch.ones_like(total_mask) * (-1e9)
            outputs = torch.where(total_mask == 0, padding, outputs)
            outputs = F.softmax(outputs, dim=2)
            outputs = self.attention_dropout(outputs)
            outputs = torch.matmul(outputs, v_)
            multi_head_attention_output = torch.cat(
                torch.split(outputs, split_size_or_sections=int(outputs.shape[0] / self.n_heads), dim=0),
                dim=2,
            )
            multi_head_attention_output = multi_head_attention_output + structural_input
            multi_head_attention_output = self.layer_norm(multi_head_attention_output)
            multi_head_attention_output = self.feed_forward(multi_head_attention_output)
            return multi_head_attention_output

    def init_weights(self):
        nn.init.kaiming_uniform_(self.Q_embedding_weights)
        nn.init.kaiming_uniform_(self.K_embedding_weights)
        nn.init.kaiming_uniform_(self.V_embedding_weights)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()

    def feed_forward(self, inputs):
        out = self.feedforward_linear_1(inputs)
        out = self.activation(out)
        out = self.feedforward_linear_2(out)
        return out


class NodeFeatureEmbeddingLayer(nn.Module):
    def __init__(self, hidden_dim: int, numerical_feature_size: int = 5,
                 categorical_feature_size: int = 3, des_feature_size: int = 768,
                 tweet_feature_size: int = 768, dropout: float = 0.3):
        super().__init__()
        self.activation = nn.PReLU()
        self.numerical_feature_linear = nn.Sequential(
            nn.Linear(numerical_feature_size, hidden_dim // 4),
            self.activation,
        )
        self.categorical_feature_linear = nn.Sequential(
            nn.Linear(categorical_feature_size, hidden_dim // 4),
            self.activation,
        )
        self.des_feature_linear = nn.Sequential(
            nn.Linear(des_feature_size, hidden_dim // 4),
            self.activation,
        )
        self.tweet_feature_linear = nn.Sequential(
            nn.Linear(tweet_feature_size, hidden_dim // 4),
            self.activation,
        )
        self.total_feature_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
        )
        self.init_weights()

    def forward(self, des_tensor, tweet_tensor, num_prop, category_prop):
        num_prop = self.numerical_feature_linear(num_prop)
        category_prop = self.categorical_feature_linear(category_prop)
        des_tensor = self.des_feature_linear(des_tensor)
        tweet_tensor = self.tweet_feature_linear(tweet_tensor)
        x = torch.cat((num_prop, category_prop, des_tensor, tweet_tensor), dim=1)
        x = self.total_feature_linear(x)
        return x

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()


class FeatureTextGraphBotDGT(nn.Module):
    """
    BotDGT model faithfully reconstructed from https://github.com/Peien429/BotDGT
    """

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
        temporal_module: str,
        temporal_heads: int,
        temporal_smoothness_weight: float,
        temporal_consistency_weight: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        structural_heads = _compatible_attention_heads(hidden_dim, attention_heads)
        temporal_head_count = _compatible_attention_heads(hidden_dim, temporal_heads)

        self.node_feature_embedding_layer = NodeFeatureEmbeddingLayer(
            hidden_dim=hidden_dim,
            numerical_feature_size=num_prop_dim,
            categorical_feature_size=cat_prop_dim,
            des_feature_size=description_dim,
            tweet_feature_size=tweet_dim,
            dropout=dropout,
        )
        self.position_encoding_clustering_coefficient_layer = PositionEncodingClusteringCoefficient(
            hidden_dim=hidden_dim,
        )
        self.position_encoding_bidirectional_links_ratio_layer = PositionEncodingBidirectionalLinks(
            hidden_dim=hidden_dim,
        )
        self.structural_layer = GraphStructuralLayer(
            hidden_dim=hidden_dim,
            n_heads=structural_heads,
            dropout=dropout,
        )
        self.num_time_steps = 0
        self.temporal_layer = GraphTemporalLayer(
            hidden_dim=hidden_dim,
            n_heads=temporal_head_count,
            dropout=dropout,
            num_time_steps=0,
            temporal_module_type=temporal_module,
        )
        self.output_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        description: torch.Tensor,
        tweet: torch.Tensor,
        num_prop: torch.Tensor,
        cat_prop: torch.Tensor,
        edge_index,  # dynamic snapshot bundle dict
        edge_type=None,
    ):
        """
        Args:
            description: [N, description_dim]
            tweet: [N, tweet_dim]
            num_prop: [N, num_prop_dim]
            cat_prop: [N, num_prop_dim]
            edge_index: dict with keys: edge_indices, clustering, bidirectional_ratio, keep_ratio
            edge_type: unused

        Processes nodes in batches to avoid OOM with large graphs (e.g. 200K+ nodes).
        Only labeled nodes (first len(known_indices)) need correct outputs; support nodes
        only contribute as neighbors in message passing.
        """
        snapshot_edge_indices = edge_index.get("edge_indices", [])
        clustering = edge_index.get("clustering")
        bidirectional_ratio = edge_index.get("bidirectional_ratio")

        if not snapshot_edge_indices:
            raise ValueError("BotDGT snapshot bundle has no edge indices.")
        num_time_steps = len(snapshot_edge_indices)
        total_nodes = num_prop.shape[0]

        if self.num_time_steps != num_time_steps:
            self.num_time_steps = num_time_steps
            self.temporal_layer.num_time_steps = num_time_steps
            self.temporal_layer._rebuild_position_embedding(num_time_steps, description.device)

        node_features = self.node_feature_embedding_layer(
            description, tweet, num_prop, cat_prop,
        )

        all_snapshots_structural_output: list[torch.Tensor] = []
        for t in range(num_time_steps):
            snapshot_edges = snapshot_edge_indices[t]
            full_output = self.structural_layer(node_features, snapshot_edges)
            all_snapshots_structural_output.append(full_output)

        all_snapshots_structural_output = torch.stack(all_snapshots_structural_output, dim=0)
        all_snapshots_structural_output = all_snapshots_structural_output.transpose(0, 1)

        position_embedding_clustering_coefficient = torch.stack([
            self.position_encoding_clustering_coefficient_layer(clustering[t].clamp(-100, 100))
            for t in range(num_time_steps)
        ], dim=0).transpose(0, 1)
        position_embedding_bidirectional_links_ratio = torch.stack([
            self.position_encoding_bidirectional_links_ratio_layer(bidirectional_ratio[t].clamp(0, 1))
            for t in range(num_time_steps)
        ], dim=0).transpose(0, 1)

        exist_nodes = torch.ones(
            num_time_steps, total_nodes,
            dtype=torch.long, device=description.device,
        )

        temporal_output = self.temporal_layer(
            all_snapshots_structural_output,
            position_embedding_clustering_coefficient,
            position_embedding_bidirectional_links_ratio,
            exist_nodes,
        )

        logits = self.output_head(temporal_output[:, -1, :])

        aux_loss = torch.tensor(0.0, device=logits.device)
        return logits, aux_loss