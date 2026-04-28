from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import TransformerConv
except Exception:  # pragma: no cover
    TransformerConv = None

from .base import _FeatureTextGraphBase, _compatible_attention_heads


class FeatureTextGraphBotSAI(_FeatureTextGraphBase):
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
    ) -> None:
        super().__init__(hidden_dim, description_dim, tweet_dim, num_prop_dim, cat_prop_dim, dropout)
        block_dim = max(4, hidden_dim // 4)
        channel_dim = block_dim * 2
        channel_heads = _compatible_attention_heads(channel_dim, attention_heads)
        graph_heads = _compatible_attention_heads(hidden_dim, attention_heads)

        self.invariant_weight = float(max(0.0, invariant_weight))
        self.invariant_projectors = nn.ModuleList([nn.Linear(block_dim, block_dim) for _ in range(4)])
        self.specific_projectors = nn.ModuleList([nn.Linear(block_dim, block_dim) for _ in range(4)])
        self.channel_attention = nn.MultiheadAttention(
            embed_dim=channel_dim,
            num_heads=channel_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.channel_norm = nn.LayerNorm(channel_dim)
        self.channel_to_hidden = nn.Sequential(nn.Linear(channel_dim, hidden_dim), nn.LeakyReLU())
        self.relation_embedding = nn.Embedding(relation_count, hidden_dim)
        self.structural_layer1 = TransformerConv(
            hidden_dim,
            hidden_dim // graph_heads,
            heads=graph_heads,
            concat=True,
            edge_dim=hidden_dim,
            dropout=dropout,
        )
        self.structural_layer2 = TransformerConv(
            hidden_dim,
            hidden_dim // graph_heads,
            heads=graph_heads,
            concat=True,
            edge_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        description: torch.Tensor,
        tweet: torch.Tensor,
        num_prop: torch.Tensor,
        cat_prop: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        modalities = self.encode_modalities(description, tweet, num_prop, cat_prop)

        invariant_parts: list[torch.Tensor] = []
        specific_parts: list[torch.Tensor] = []
        channels: list[torch.Tensor] = []
        for index, modality in enumerate(modalities):
            invariant = torch.tanh(self.invariant_projectors[index](modality))
            specific = F.leaky_relu(self.specific_projectors[index](modality))
            invariant_parts.append(invariant)
            specific_parts.append(specific)
            channels.append(torch.cat((invariant, specific), dim=1))

        channel_tensor = torch.stack(channels, dim=1)
        channel_tensor = self.dropout(channel_tensor)
        attended, _ = self.channel_attention(channel_tensor, channel_tensor, channel_tensor)
        channel_tensor = self.channel_norm(attended + channel_tensor)
        fused = self.channel_to_hidden(channel_tensor.mean(dim=1))

        if edge_type is None:
            edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
        bounded_edge_type = edge_type.long().clamp(min=0, max=self.relation_embedding.num_embeddings - 1)
        edge_attr = self.relation_embedding(bounded_edge_type)

        x = self.structural_layer1(fused, edge_index, edge_attr=edge_attr)
        x = self.dropout(F.leaky_relu(x)) + fused
        x = self.structural_layer2(x, edge_index, edge_attr=edge_attr)
        x = self.dropout(F.leaky_relu(x)) + fused
        x = self.output_mlp(x)
        logits = self.output_head(x + fused)

        inv_stack = torch.stack(invariant_parts, dim=1)
        inv_center = inv_stack.mean(dim=1, keepdim=True)
        invariance_loss = ((inv_stack - inv_center) ** 2).mean()
        specific_overlap = torch.tensor(0.0, device=logits.device)
        pair_count = 0
        for left in range(len(specific_parts)):
            for right in range(left + 1, len(specific_parts)):
                specific_overlap = specific_overlap + F.cosine_similarity(
                    specific_parts[left],
                    specific_parts[right],
                    dim=1,
                ).abs().mean()
                pair_count += 1
        if pair_count > 0:
            specific_overlap = specific_overlap / pair_count
        aux_loss = self.invariant_weight * (invariance_loss + 0.5 * specific_overlap)
        return logits, aux_loss
