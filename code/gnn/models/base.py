from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compatible_attention_heads(hidden_dim: int, preferred: int) -> int:
    preferred = max(1, int(preferred))
    for candidate in sorted(set([preferred, 8, 4, 2, 1]), reverse=True):
        if candidate <= hidden_dim and hidden_dim % candidate == 0:
            return candidate
    return 1


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
