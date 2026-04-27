from __future__ import annotations

import torch

try:
    from torch_geometric.nn import RGCNConv
except Exception:  # pragma: no cover
    RGCNConv = None

from .base import _FeatureTextGraphBase


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
