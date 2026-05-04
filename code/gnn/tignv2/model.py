from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..botdgt.model import (
    GraphStructuralLayer,
    NodeFeatureEmbeddingLayer,
    PositionEncodingBidirectionalLinks,
    PositionEncodingClusteringCoefficient,
)


class _PerModalityInvariantEncoder(nn.Module):
    """对单个模态做 invariant/specific 分解（参考 BotSAI 机制，共享于所有快照）"""

    def __init__(self, input_dim: int, block_dim: int, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, block_dim),
            nn.PReLU(),
        )
        self.invariant_projector = nn.Sequential(
            nn.Linear(block_dim, block_dim),
            nn.Tanh(),
        )
        self.specific_projector = nn.Sequential(
            nn.Linear(block_dim, block_dim),
            nn.LeakyReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, x):
        h = self.encoder(x)
        h = self.dropout(h)
        inv = self.invariant_projector(h)
        spc = self.specific_projector(h)
        return inv, spc

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class _ChannelAttentionFusion(nn.Module):
    """BotSAI 风格的通道自注意力融合（4 个模态通道 → 1 个融合表示）"""

    def __init__(self, channel_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.channel_attn = nn.MultiheadAttention(
            channel_dim, num_heads, dropout=0.1, batch_first=True,
        )
        self.channel_norm = nn.LayerNorm(channel_dim)
        self.channel_to_hidden = nn.Sequential(
            nn.Linear(channel_dim, hidden_dim),
            nn.PReLU(),
        )

    def forward(self, channels):
        # channels: [B, 4, channel_dim]
        attended, _ = self.channel_attn(channels, channels, channels)
        attended = self.channel_norm(attended + channels)  # residual
        fused = attended.mean(dim=1)  # [B, channel_dim]
        return self.channel_to_hidden(fused)  # [B, hidden_dim]


class TIGNv2Model(nn.Module):
    """
    Temporal Invariant Graph Network v2.

    核心创新：
    1. 日历月快照 + 每快照多模态 invariant/specific 分解
    2. 跨时间不变性约束（同一模态的 invariant 在相邻快照间平滑）
    3. 双流时序处理（invariant GRU + specific GRU，交叉注意力融合）
    """

    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.structural_heads = args.structural_head_config
        self.structural_drop = args.structural_drop
        self.temporal_heads = args.temporal_head_config
        self.temporal_drop = args.temporal_drop
        self.window_size = args.window_size
        self.temporal_module_type = args.temporal_module_type
        self.block_dim = max(16, self.hidden_dim // 2)

        # 每模态 invariant/specific 编码器（权重跨时间共享）
        self.modality_encoders = nn.ModuleList([
            _PerModalityInvariantEncoder(768, self.block_dim, args.embedding_dropout),
            _PerModalityInvariantEncoder(768, self.block_dim, args.embedding_dropout),
            _PerModalityInvariantEncoder(5, self.block_dim, args.embedding_dropout),
            _PerModalityInvariantEncoder(3, self.block_dim, args.embedding_dropout),
        ])

        # 通道注意力融合
        channel_dim = self.block_dim * 2  # [inv, spc] 拼接
        self.channel_fusion = _ChannelAttentionFusion(
            channel_dim, self.hidden_dim, num_heads=4,
        )

        # 图结构编码（权重跨时间共享）
        self.structural_layer = GraphStructuralLayer(
            hidden_dim=self.hidden_dim,
            n_heads=self.structural_heads,
            dropout=self.structural_drop,
        )

        # 位置编码
        self.pos_clustering = PositionEncodingClusteringCoefficient(self.hidden_dim)
        self.pos_blr = PositionEncodingBidirectionalLinks(self.hidden_dim)
        self.temporal_pos_embedding = nn.Embedding(200, self.hidden_dim)  # 最多 200 个快照

        # 双流时序 GRU
        self.invariant_gru = nn.GRU(
            self.block_dim, self.block_dim, num_layers=1,
            batch_first=True, bidirectional=False,
        )
        self.specific_gru = nn.GRU(
            self.block_dim, self.block_dim, num_layers=1,
            batch_first=True, bidirectional=False,
        )

        # 交叉流注意力（specific 查询 invariant）
        self.cross_stream_attn = nn.MultiheadAttention(
            self.block_dim, num_heads=4, dropout=0.1, batch_first=True,
        )
        self.cross_stream_norm = nn.LayerNorm(self.block_dim)

        # 时序融合层: cross_attended(block_dim) + struct_stack(hidden_dim) + temporal_pos(hidden_dim)
        self.temporal_fusion = nn.Sequential(
            nn.Linear(self.block_dim + self.hidden_dim * 2, self.hidden_dim),
            nn.PReLU(),
        )

        # 输出头
        self.output_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.PReLU(),
            nn.Dropout(0.3),
        )
        self.output_head = nn.Linear(self.hidden_dim, 2)

        self.activation = nn.PReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def _process_one_snapshot(self, des, tweet, num, cat, edge_index, clustering, blr,
                              batch_size):
        """
        处理单个快照：模态编码 → 不变性分解 → 通道融合 → 图结构编码 → 位置编码
        返回 fused_struct [batch_size, hidden_dim]、inv_stack [batch_size, 4, block_dim]、
        spc_stack [batch_size, 4, block_dim]
        """
        # 1. 每模态编码 + 不变性/特异性分解
        all_inv, all_spc = [], []
        for i, (encoder, feat) in enumerate(zip(
            self.modality_encoders, [des, tweet, num, cat]
        )):
            inv, spc = encoder(feat)
            all_inv.append(inv)  # [N, block_dim]
            all_spc.append(spc)  # [N, block_dim]

        # 2. 构建通道: [inv, spc] 拼接 → [N, 4, channel_dim]
        channels = []
        for inv, spc in zip(all_inv, all_spc):
            channels.append(torch.cat([inv, spc], dim=-1))
        channels = torch.stack(channels, dim=1)  # [N, 4, channel_dim]

        # 3. 通道注意力融合 → [N, hidden_dim]
        fused = self.channel_fusion(channels)

        # 4. 图结构编码
        struct_out = self.structural_layer(fused, edge_index)

        # 5. 位置编码
        pos_c = self.pos_clustering(clustering)
        pos_b = self.pos_blr(blr)
        fused_struct = struct_out + pos_c + pos_b

        # 切片到当前批次大小
        fused_struct = fused_struct[:batch_size]

        # 收集 invariant/specific
        inv_stack = torch.stack([inv[:batch_size] for inv in all_inv], dim=1)  # [B, 4, block_dim]
        spc_stack = torch.stack([spc[:batch_size] for spc in all_spc], dim=1)  # [B, 4, block_dim]

        return fused_struct, inv_stack, spc_stack

    def forward(self, all_snapshots_des, all_snapshots_tweet, all_snapshots_num,
                all_snapshots_cat, all_snapshots_edge_index, all_snapshots_clustering,
                all_snapshots_blr, all_snapshots_exist_nodes, current_batch_size):
        """
        Args:
            all_snapshots_*: 列表，每个元素对应一个快照
            current_batch_size: 当前批次的节点数
        Returns:
            logits: [window_size, batch_size, 2]
            aux: dict with invariant_stacks [B, T, 4, block_dim],
                 specific_stacks [B, T, 4, block_dim]
        """
        T = len(all_snapshots_des)
        B = current_batch_size

        all_struct = []
        all_inv_stacks = []
        all_spc_stacks = []

        for t in range(T):
            fused_struct, inv_stack, spc_stack = self._process_one_snapshot(
                all_snapshots_des[t], all_snapshots_tweet[t],
                all_snapshots_num[t], all_snapshots_cat[t],
                all_snapshots_edge_index[t],
                all_snapshots_clustering[t],
                all_snapshots_blr[t],
                current_batch_size,
            )
            all_struct.append(fused_struct)       # [B, H]
            all_inv_stacks.append(inv_stack)      # [B, 4, block_dim]
            all_spc_stacks.append(spc_stack)      # [B, 4, block_dim]

        # 堆叠时间维
        struct_stack = torch.stack(all_struct, dim=1)          # [B, T, H]
        inv_stacks = torch.stack(all_inv_stacks, dim=1)       # [B, T, 4, block_dim]
        spc_stacks = torch.stack(all_spc_stacks, dim=1)       # [B, T, 4, block_dim]

        # 聚合每模态 invariant/specific 用于时序处理
        # 对 4 个模态取平均 → [B, T, block_dim]
        inv_mean = inv_stacks.mean(dim=2)   # [B, T, block_dim]
        spc_mean = spc_stacks.mean(dim=2)   # [B, T, block_dim]

        # 双流 GRU 时序处理
        inv_stream, _ = self.invariant_gru(inv_mean)   # [B, T, block_dim]
        spc_stream, _ = self.specific_gru(spc_mean)    # [B, T, block_dim]

        # 残差连接
        inv_stream = inv_stream + inv_mean
        spc_stream = spc_stream + spc_mean

        # 交叉流注意力: specific 关注 invariant
        cross_attended, _ = self.cross_stream_attn(spc_stream, inv_stream, inv_stream)
        cross_attended = self.cross_stream_norm(cross_attended + spc_stream)

        # 时序位置编码
        positions = torch.arange(T, device=struct_stack.device).unsqueeze(0).expand(B, -1)
        temporal_pos = self.temporal_pos_embedding(positions)  # [B, T, H]

        # 融合: [cross_attended | struct_stack | temporal_pos]
        combined = torch.cat([cross_attended, struct_stack, temporal_pos], dim=-1)
        combined = self.temporal_fusion(combined)  # [B, T, H]

        # 输出头
        combined = self.output_mlp(combined)
        logits = self.output_head(combined)  # [B, T, 2]

        # 转置以匹配 all_snapshots_loss 的期望: [T, B, 2]
        logits = logits.transpose(0, 1)

        return logits, {
            "invariant_stacks": inv_stacks,   # [B, T, 4, block_dim]
            "specific_stacks": spc_stacks,    # [B, T, 4, block_dim]
        }
