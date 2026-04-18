from __future__ import annotations

import torch
import torch.nn as nn


class TemporalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.pool = nn.Linear(d_model, 1)

    def forward(self, sequences: torch.Tensor, exist_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key_padding_mask = ~exist_mask
        attended, _ = self.attention(
            sequences,
            sequences,
            sequences,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attended = attended.masked_fill(~exist_mask.unsqueeze(-1), 0.0)
        scores = self.pool(attended).squeeze(-1)
        scores = scores.masked_fill(~exist_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(exist_mask, weights, torch.zeros_like(weights))
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        weights = weights / denom
        pooled = torch.sum(attended * weights.unsqueeze(-1), dim=1)
        return pooled, weights
