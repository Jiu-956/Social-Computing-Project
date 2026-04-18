from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from dafbot.models.attr_encoder import AttrEncoder
from dafbot.models.classifier import ClassifierHead
from dafbot.models.fusion import AdaptiveFusion, ConcatFusion
from dafbot.models.graph_encoder import SnapshotGraphEncoder
from dafbot.models.temporal_encoder import TemporalSelfAttention
from dafbot.models.text_encoder import TextEncoder


class DynamicAdaptiveFusionBotDetector(nn.Module):
    def __init__(
        self,
        attr_dim: int,
        text_dim: int,
        quality_dim: int,
        model_config: dict[str, Any],
        variant: dict[str, Any],
    ) -> None:
        super().__init__()
        d_model = int(model_config["d_model"])
        dropout = float(model_config["dropout"])
        self.variant = variant
        self.use_attr = bool(variant.get("use_attr", True))
        self.use_text = bool(variant.get("use_text", True))
        self.use_graph = bool(variant.get("use_graph", True))
        self.temporal_enabled = bool(variant.get("temporal_enabled", True))
        self.use_quality = bool(variant.get("use_quality", True))
        self.graph_mode = variant.get("graph_mode", "dynamic")

        self.attr_encoder = AttrEncoder(attr_dim, int(model_config["attr_hidden_dim"]), d_model, dropout) if self.use_attr else None
        self.text_encoder = TextEncoder(text_dim, int(model_config["text_hidden_dim"]), d_model, dropout) if self.use_text else None
        self.graph_encoder = SnapshotGraphEncoder(attr_dim, int(model_config["graph_hidden_dim"]), d_model, dropout) if self.use_graph else None
        self.temporal_encoder = (
            TemporalSelfAttention(d_model, int(model_config["temporal_heads"]), dropout)
            if self.use_graph and self.temporal_enabled and self.graph_mode == "dynamic"
            else None
        )

        fusion_type = variant.get("fusion_type", "adaptive")
        if fusion_type == "concat":
            self.fusion = ConcatFusion(d_model, int(model_config["fusion_hidden_dim"]), dropout)
        else:
            self.fusion = AdaptiveFusion(
                d_model=d_model,
                quality_dim=quality_dim,
                hidden_dim=int(model_config["fusion_hidden_dim"]),
                dropout=dropout,
                use_quality=self.use_quality,
            )
        self.classifier = ClassifierHead(d_model, dropout)

    def _encode_dynamic_graph(self, attr_features: torch.Tensor, snapshots: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.graph_mode == "static" or self.temporal_encoder is None:
            static_snapshot = snapshots[-1]
            encoded = self.graph_encoder(attr_features, static_snapshot)
            weights = static_snapshot["exist_nodes"].unsqueeze(1)
            return encoded, weights

        encoded_snapshots = []
        exist_masks = []
        for snapshot in snapshots:
            encoded_snapshots.append(self.graph_encoder(attr_features, snapshot))
            exist_masks.append(snapshot["exist_nodes"].bool())
        sequence = torch.stack(encoded_snapshots, dim=1)
        exist_mask = torch.stack(exist_masks, dim=1)
        h_dyn, temporal_weights = self.temporal_encoder(sequence, exist_mask)
        return h_dyn, temporal_weights

    def forward(
        self,
        attr_features: torch.Tensor,
        text_features: torch.Tensor,
        snapshots: list[dict[str, torch.Tensor]],
        quality_features: torch.Tensor,
        node_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        h_attr = self.attr_encoder(attr_features) if self.use_attr and self.attr_encoder is not None else None
        h_text = self.text_encoder(text_features) if self.use_text and self.text_encoder is not None else None
        h_dyn = None
        temporal_weights = None
        if self.use_graph and self.graph_encoder is not None:
            h_dyn, temporal_weights = self._encode_dynamic_graph(attr_features, snapshots)

        fused, modal_weights = self.fusion(
            h_attr=h_attr,
            h_text=h_text,
            h_dyn=h_dyn,
            quality_features=quality_features if self.use_quality else None,
            active_modalities={"attr": self.use_attr, "text": self.use_text, "dyn": self.use_graph},
        )
        logits = self.classifier(fused)
        if node_indices is not None:
            logits = logits[node_indices]

        aux = {
            "h_attr": h_attr,
            "h_text": h_text,
            "h_dyn": h_dyn,
            "h_fused": fused,
            "modal_weights": modal_weights,
            "temporal_weights": temporal_weights,
        }
        return logits, aux
