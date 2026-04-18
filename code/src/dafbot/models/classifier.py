from __future__ import annotations

import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.network(x)
