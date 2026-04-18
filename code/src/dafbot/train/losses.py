from __future__ import annotations

import torch
import torch.nn.functional as F


def modal_entropy_regularizer(modal_weights: torch.Tensor | None) -> torch.Tensor:
    if modal_weights is None:
        return torch.tensor(0.0)
    probs = modal_weights.clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum(dim=1).mean()
    return entropy


def temporal_entropy_regularizer(temporal_weights: torch.Tensor | None) -> torch.Tensor:
    if temporal_weights is None:
        return torch.tensor(0.0)
    probs = temporal_weights.clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum(dim=1).mean()
    return entropy


def build_total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    aux: dict[str, torch.Tensor | None],
    lambda_modal: float,
    lambda_temporal: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    cls_loss = F.cross_entropy(logits[indices], labels[indices])
    modal_reg = modal_entropy_regularizer(aux.get("modal_weights"))
    temporal_reg = temporal_entropy_regularizer(aux.get("temporal_weights"))
    device = cls_loss.device
    if modal_reg.device != device:
        modal_reg = modal_reg.to(device)
    if temporal_reg.device != device:
        temporal_reg = temporal_reg.to(device)
    total = cls_loss + lambda_modal * modal_reg + lambda_temporal * temporal_reg
    return total, {
        "loss_cls": float(cls_loss.detach().cpu()),
        "loss_modal": float(modal_reg.detach().cpu()),
        "loss_temporal": float(temporal_reg.detach().cpu()),
        "loss_total": float(total.detach().cpu()),
    }
