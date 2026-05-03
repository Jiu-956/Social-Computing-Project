from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import RGCNConv
except Exception:  # pragma: no cover
    RGCNConv = None

from .base import _FeatureTextGraphBase


class FeatureTextGraphBotRGCN(_FeatureTextGraphBase):
    """BotRGCN faithfully reconstructed from TwiBot-22 baseline.

    Architecture:
    - 4 modality encoders -> concat -> input_mlp
    - 2-layer RGCNConv (num_relations=2 for follower/following) with LayerNorm
    - dropout between layers
    - output_mlp -> output_head
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
    ) -> None:
        super().__init__(hidden_dim, description_dim, tweet_dim, num_prop_dim, cat_prop_dim, dropout)
        self.rgcn1 = RGCNConv(hidden_dim, hidden_dim, num_relations=relation_count)
        self.rgcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations=relation_count)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

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
        d = self.description_mlp(description)
        t = self.tweet_mlp(tweet)
        n = self.num_mlp(num_prop)
        c = self.cat_mlp(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.input_mlp(x)
        x = self.norm1(x)
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.rgcn2(x, edge_index, edge_type)
        x = F.leaky_relu(x)
        x = self.output_mlp(x)
        return self.output_head(x)