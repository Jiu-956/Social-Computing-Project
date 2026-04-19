from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        del num_heads
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.Linear(d_model, 1)

    def forward(self, sequences: torch.Tensor, exist_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attended = sequences.masked_fill(~exist_mask.unsqueeze(-1), 0.0)
        flattened = attended.contiguous().view(-1, attended.size(-1))
        projected = torch.tanh(F.linear(flattened, self.proj.weight, self.proj.bias))
        projected = self.dropout(projected)
        scores = F.linear(projected, self.pool.weight, self.pool.bias).view(attended.size(0), attended.size(1))
        scores = scores.masked_fill(~exist_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(exist_mask, weights, torch.zeros_like(weights))
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        weights = weights / denom
        pooled = torch.sum(attended * weights.unsqueeze(-1), dim=1)
        return pooled, weights
