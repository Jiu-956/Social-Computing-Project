from __future__ import annotations

import torch

try:
    from torch_geometric.nn import GATConv
except Exception:  # pragma: no cover
    GATConv = None

from .base import _FeatureTextGraphBase


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
