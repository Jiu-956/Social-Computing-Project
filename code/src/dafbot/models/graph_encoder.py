from __future__ import annotations

import torch
import torch.nn as nn


class GraphSAGELayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.lin_self = nn.Linear(input_dim, output_dim)
        self.lin_neigh = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        device = x.device
        if edge_index.numel() == 0:
            neighbor_mean = torch.zeros_like(x)
        else:
            source, target = edge_index[0], edge_index[1]
            aggregated = torch.zeros_like(x)
            aggregated.index_add_(0, target, x[source])
            degree = torch.zeros(num_nodes, 1, device=device, dtype=x.dtype)
            degree.index_add_(0, target, torch.ones(target.size(0), 1, device=device, dtype=x.dtype))
            neighbor_mean = aggregated / degree.clamp_min(1.0)
        return self.lin_self(x) + self.lin_neigh(neighbor_mean)


class SnapshotGraphEncoder(nn.Module):
    def __init__(self, attr_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        input_dim = attr_dim + 3
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GraphSAGELayer(hidden_dim, hidden_dim)
        self.conv2 = GraphSAGELayer(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, attr_features: torch.Tensor, snapshot: dict[str, torch.Tensor]) -> torch.Tensor:
        positional = torch.cat(
            [
                snapshot["clustering_coefficient"],
                snapshot["bidirectional_links_ratio"],
                snapshot["exist_nodes"].unsqueeze(-1),
            ],
            dim=1,
        )
        x = torch.cat([attr_features, positional], dim=1)
        exist_mask = snapshot["exist_nodes"].unsqueeze(-1)
        x = self.input_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv1(x, snapshot["edge_index"])
        x = self.activation(x)
        x = self.dropout(x)
        x = x * exist_mask
        x = self.conv2(x, snapshot["edge_index"])
        x = x * exist_mask
        return x
