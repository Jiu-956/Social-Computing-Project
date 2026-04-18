from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


MODALITY_ORDER = ["attr", "text", "dyn"]


class AdaptiveFusion(nn.Module):
    def __init__(self, d_model: int, quality_dim: int, hidden_dim: int, dropout: float, use_quality: bool) -> None:
        super().__init__()
        self.use_quality = use_quality
        self.attr_proj = nn.Linear(d_model, d_model)
        self.text_proj = nn.Linear(d_model, d_model)
        self.dyn_proj = nn.Linear(d_model, d_model)
        gate_input_dim = d_model * 3 + (quality_dim if use_quality else 0)
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        h_attr: torch.Tensor | None,
        h_text: torch.Tensor | None,
        h_dyn: torch.Tensor | None,
        quality_features: torch.Tensor | None,
        active_modalities: dict[str, bool],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = {
            "attr": self.attr_proj(h_attr) if h_attr is not None else None,
            "text": self.text_proj(h_text) if h_text is not None else None,
            "dyn": self.dyn_proj(h_dyn) if h_dyn is not None else None,
        }

        available = [name for name in MODALITY_ORDER if active_modalities.get(name, False)]
        if len(available) == 1:
            chosen = projected[available[0]]
            weights = torch.zeros(chosen.size(0), 3, device=chosen.device, dtype=chosen.dtype)
            weights[:, MODALITY_ORDER.index(available[0])] = 1.0
            return chosen, weights

        zero_template = next(tensor for tensor in projected.values() if tensor is not None)
        gate_inputs = []
        for name in MODALITY_ORDER:
            tensor = projected[name]
            gate_inputs.append(tensor if tensor is not None else torch.zeros_like(zero_template))
        if self.use_quality and quality_features is not None:
            gate_inputs.append(quality_features)
        gate_logits = self.gate(torch.cat(gate_inputs, dim=1))

        mask = torch.tensor([active_modalities.get(name, False) for name in MODALITY_ORDER], device=gate_logits.device)
        gate_logits = gate_logits.masked_fill(~mask.unsqueeze(0), -1e9)
        weights = torch.softmax(gate_logits, dim=1)
        fused = 0.0
        for index, name in enumerate(MODALITY_ORDER):
            if projected[name] is not None and active_modalities.get(name, False):
                fused = fused + weights[:, index : index + 1] * projected[name]
        return fused, weights


class ConcatFusion(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(
        self,
        h_attr: torch.Tensor | None,
        h_text: torch.Tensor | None,
        h_dyn: torch.Tensor | None,
        _: torch.Tensor | None,
        active_modalities: dict[str, bool],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = []
        template = next(tensor for tensor in (h_attr, h_text, h_dyn) if tensor is not None)
        for tensor, key in zip((h_attr, h_text, h_dyn), MODALITY_ORDER):
            if active_modalities.get(key, False) and tensor is not None:
                tensors.append(tensor)
            else:
                tensors.append(torch.zeros_like(template))
        fused = self.network(torch.cat(tensors, dim=1))
        weights = torch.zeros(fused.size(0), 3, device=fused.device, dtype=fused.dtype)
        active_count = sum(int(active_modalities.get(name, False)) for name in MODALITY_ORDER)
        if active_count > 0:
            for index, name in enumerate(MODALITY_ORDER):
                if active_modalities.get(name, False):
                    weights[:, index] = 1.0 / active_count
        return fused, weights
