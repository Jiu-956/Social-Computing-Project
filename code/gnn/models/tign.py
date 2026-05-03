from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import TransformerConv
except Exception:  # pragma: no cover
    TransformerConv = None

from .base import _FeatureTextGraphBase, _compatible_attention_heads


class FeatureTextGraphTIGN(_FeatureTextGraphBase):
    """Temporal Invariant Graph Network — BotSAI + Age-Aware Relation Embedding.

    Core innovation: edge types encode (relation_type, src_age_bucket, tgt_age_bucket),
    yielding 2*3*3=18 relation types. Edge embeddings are generated via a small MLP
    from learned relation + age bucket embeddings, enabling the model to distinguish
    e.g. "old_follows_new" vs "new_follows_old".

    Invariant loss is retained from BotSAI. Additionally, an intra-class variance
    term encourages bot accounts' invariant embeddings to cluster together and human
    accounts to cluster together, reducing within-class dispersion.
    """

    def __init__(
        self,
        hidden_dim: int,
        description_dim: int,
        tweet_dim: int,
        num_prop_dim: int,
        cat_prop_dim: int,
        dropout: float,
        num_age_buckets: int,
        relation_count: int,
        invariant_weight: float,
        intra_class_weight: float,
        attention_heads: int,
    ) -> None:
        super().__init__(hidden_dim, description_dim, tweet_dim, num_prop_dim, cat_prop_dim, dropout)
        self.num_age_buckets = num_age_buckets
        total_relations = relation_count * num_age_buckets * num_age_buckets

        block_dim = max(4, hidden_dim // 4)
        channel_dim = block_dim * 2
        channel_heads = _compatible_attention_heads(channel_dim, attention_heads)
        graph_heads = _compatible_attention_heads(hidden_dim, attention_heads)

        self.invariant_weight = float(max(0.0, invariant_weight))
        self.intra_class_weight = float(max(0.0, intra_class_weight))
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

        # Dynamic relation embedding: (relation_type, src_bucket, tgt_bucket) -> embedding
        self.rel_embedding = nn.Embedding(total_relations, hidden_dim)
        age_bucket_dim = num_age_buckets
        self.age_bucket_embedding = nn.Embedding(age_bucket_dim, hidden_dim // 4)
        self.edge_embed_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4 * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.structural_layer1 = TransformerConv(
            hidden_dim,
            hidden_dim // graph_heads,
            heads=graph_heads,
            concat=True,
            edge_dim=hidden_dim,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.structural_layer2 = TransformerConv(
            hidden_dim,
            hidden_dim // graph_heads,
            heads=graph_heads,
            concat=True,
            edge_dim=hidden_dim,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.structural_layer3 = TransformerConv(
            hidden_dim,
            hidden_dim // graph_heads,
            heads=graph_heads,
            concat=True,
            edge_dim=hidden_dim,
            dropout=dropout,
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def _build_edge_attr(self, edge_type: torch.Tensor) -> torch.Tensor:
        batch_size = edge_type.shape[0]
        device = edge_type.device
        buckets = self.num_age_buckets

        rel_id = edge_type // (buckets * buckets)
        src_bucket = (edge_type // buckets) % buckets
        tgt_bucket = edge_type % buckets

        rel_embed = self.rel_embedding(rel_id)
        src_age_embed = self.age_bucket_embedding(src_bucket)
        tgt_age_embed = self.age_bucket_embedding(tgt_bucket)

        combined = torch.cat([rel_embed, src_age_embed, tgt_age_embed], dim=-1)
        return self.edge_embed_mlp(combined)

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
        bounded_type = edge_type.long().clamp(min=0, max=self.rel_embedding.num_embeddings - 1)
        edge_attr = self._build_edge_attr(bounded_type)

        x = self.structural_layer1(self.norm1(fused), edge_index, edge_attr=edge_attr)
        x = self.dropout(F.leaky_relu(x)) + fused
        x = self.structural_layer2(self.norm2(x), edge_index, edge_attr=edge_attr)
        x = self.dropout(F.leaky_relu(x)) + fused
        x = self.structural_layer3(self.norm3(x), edge_index, edge_attr=edge_attr)
        x = self.output_mlp(x + fused)
        logits = self.output_head(x)

        inv_stack = torch.stack(invariant_parts, dim=1)
        inv_center = inv_stack.mean(dim=1, keepdim=True)
        invariance_loss = ((inv_stack - inv_center) ** 2).mean()

        specific_overlap = torch.tensor(0.0, device=logits.device)
        pair_count = 0
        for left in range(len(specific_parts)):
            for right in range(left + 1, len(specific_parts)):
                specific_overlap = specific_overlap + F.cosine_similarity(
                    specific_parts[left], specific_parts[right], dim=1
                ).abs().mean()
                pair_count += 1
        if pair_count > 0:
            specific_overlap = specific_overlap / pair_count

        inv_loss = self.invariant_weight * (invariance_loss + 0.5 * specific_overlap)

        # Intra-class variance: encourage invariant parts of same-label samples to be close
        intra_var = torch.tensor(0.0, device=logits.device)
        if self.intra_class_weight > 0 and logits.shape[0] > 1:
            inv_center_per_sample = inv_center.squeeze(1)
            bot_mask = torch.nonzero(logits[:, 1] >= logits[:, 0], as_tuple=False).squeeze(-1)
            human_mask = torch.nonzero(logits[:, 0] > logits[:, 1], as_tuple=False).squeeze(-1)
            if bot_mask.shape[0] > 1:
                bot_center = inv_center_per_sample[bot_mask].mean(dim=0, keepdim=True)
                intra_var = intra_var + ((inv_center_per_sample[bot_mask] - bot_center) ** 2).mean()
            if human_mask.shape[0] > 1:
                human_center = inv_center_per_sample[human_mask].mean(dim=0, keepdim=True)
                intra_var = intra_var + ((inv_center_per_sample[human_mask] - human_center) ** 2).mean()

        aux_loss = inv_loss + self.intra_class_weight * intra_var
        return logits, aux_loss