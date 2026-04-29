from __future__ import annotations

import torch

try:
    from torch_geometric.nn import GCNConv
except Exception:  # pragma: no cover
    GCNConv = None

from .base import _FeatureTextGraphBase


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
        base_x = self.encode_inputs(description, tweet, num_prop, cat_prop)
        x = self.gcn1(base_x, edge_index)
        x = self.dropout(x) + base_x
        x = self.gcn2(x, edge_index)
        x = self.output_mlp(x + base_x)
        return self.output_head(x)
