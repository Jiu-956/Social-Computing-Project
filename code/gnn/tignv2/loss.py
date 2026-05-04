from __future__ import annotations

import torch
import torch.nn.functional as F


def one_snapshot_loss(criterion, output, label, exist_nodes):
    output = output[torch.where(exist_nodes == 1)]
    label = label[torch.where(exist_nodes == 1)]
    if output.numel() == 0:
        return torch.tensor(0.0, device=output.device)
    return criterion(output, label)


def all_snapshots_loss(criterion, output, label, exist_nodes, coefficient=1.1):
    """BotDGT 的指数加权快照损失"""
    snapshot_num = output.shape[0]
    loss_coefficient = [coefficient ** i for i in range(snapshot_num)]
    total_loss = torch.tensor(0.0, device=output.device)
    for i in range(snapshot_num):
        if torch.all(exist_nodes[i] == 0):
            continue
        loss = one_snapshot_loss(criterion, output[i], label[i], exist_nodes[i])
        total_loss = total_loss + loss * loss_coefficient[i]
    return total_loss


def cross_modal_invariance_loss(invariant_stacks):
    """
    BotSAI 的跨模态不变性损失。
    每个快照内，4 个模态的 invariant 表示应彼此接近。

    Args:
        invariant_stacks: [B, T, 4, block_dim] — 每模态每快照的 invariant 表示
    Returns:
        scalar loss
    """
    B, T, num_mods, block_dim = invariant_stacks.shape
    center = invariant_stacks.mean(dim=2)  # [B, T, block_dim]
    loss = F.mse_loss(invariant_stacks, center.unsqueeze(2).expand(-1, -1, num_mods, -1))
    return loss


def cross_temporal_invariance_loss(invariant_stacks, exist_nodes=None):
    """
    新提出的跨时间不变性损失。
    同一模态的 invariant 表示在相邻快照间应平滑变化。

    Args:
        invariant_stacks: [B, T, 4, block_dim]
        exist_nodes: [T, B] — 每个快照中节点是否存在的掩码（可选）
    Returns:
        scalar loss
    """
    B, T, num_mods, block_dim = invariant_stacks.shape
    if T < 2:
        return torch.tensor(0.0, device=invariant_stacks.device)
    loss = torch.tensor(0.0, device=invariant_stacks.device)
    count = 0
    for t in range(1, T):
        loss = loss + F.mse_loss(invariant_stacks[:, t], invariant_stacks[:, t - 1])
        count += 1
    return loss / count


def specific_decorrelation_loss(specific_stacks):
    """
    BotSAI 的特异性去相关损失。
    不同模态的 specific 表示应编码不同信息（低余弦相似度）。

    Args:
        specific_stacks: [B, T, 4, block_dim]
    Returns:
        scalar loss
    """
    B, T, num_mods, block_dim = specific_stacks.shape
    if num_mods < 2:
        return torch.tensor(0.0, device=specific_stacks.device)
    total = torch.tensor(0.0, device=specific_stacks.device)
    count = 0
    for i in range(num_mods):
        for j in range(i + 1, num_mods):
            cos = F.cosine_similarity(
                specific_stacks[:, :, i].reshape(-1, block_dim),
                specific_stacks[:, :, j].reshape(-1, block_dim),
                dim=-1,
            )
            total = total + cos.abs().mean()
            count += 1
    return total / count if count > 0 else total


def composite_loss(
    criterion, output, label, exist_nodes,
    invariant_stacks, specific_stacks,
    coefficient=1.1,
    cross_modal_weight=0.05,
    temporal_inv_weight=0.03,
    specific_decorr_weight=0.025,
):
    """
    TIGN-v2 复合损失函数。

    L_total = L_CE (all_snapshots)
            + λ1 * L_cross_modal_invariance
            + λ2 * L_cross_temporal_invariance
            + λ3 * L_specific_decorrelation
    """
    ce_loss = all_snapshots_loss(criterion, output, label, exist_nodes, coefficient)

    cm_loss = cross_modal_invariance_loss(invariant_stacks)
    ct_loss = cross_temporal_invariance_loss(invariant_stacks)
    sd_loss = specific_decorrelation_loss(specific_stacks)

    total = ce_loss
    total = total + cross_modal_weight * cm_loss
    total = total + temporal_inv_weight * ct_loss
    total = total + specific_decorr_weight * sd_loss

    return total, {
        "ce_loss": ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss,
        "cross_modal_inv": cm_loss.item(),
        "temporal_inv": ct_loss.item(),
        "specific_decorr": sd_loss.item(),
    }
