# BotDGT + BotSAI 论文级复现计划

## Context

用户要求将两篇论文的方法达到论文效果：
- **2404.15070v2 — BotDGT**: TwiBot-20 目标 F1=0.8689, Acc=0.8721
- **2408.03096v1 — BotSAI**: TwiBot-20 目标 F1=0.8738, Acc=0.8801

上一轮已完成架构对齐（BotDGT model 结构、all_snapshots_loss、ModelTrainConfig 等），当前结果 BotDGT=0.8466, BotSAI=0.8619。

**根本原因分析**: 参考 BotDGT 处理全部 229,580 节点（含大量未标注节点），GNN 结构编码能利用完整图拓扑；而当前实现仅处理 ~8,278 个标注节点，丢失了大部分图结构信息。这是无法达到论文效果的核心问题。

## 参考 vs 当前：最深层次差异

参考代码 (D:\作业\社会计算\BotDGT) 的完整数据流：

```
node.json (229,580 users) → 提取全部节点
  ├─ 按月历日期构建快照 (2008-01 ~ 2020-09, ~156 snapshots)
  ├─ 每个快照: edge_index + edge_type + exist_nodes 掩码
  ├─ networkx 计算 clustering_coefficient + bidirectional_links_ratio
  ├─ 预计算全部节点的 Transformer 嵌入 (des 768d + tweet 768d)
  └─ NeighborLoader (num_neighbors=[2560,2560], batch_size=64)
       └─ 每 batch 采样子图 → structural layer 编码全部子图节点
            → 仅取 [:batch_size] 标注节点计算 loss
```

当前实现的数据流：

```
node.json → 仅提取标注节点 (~8,278) + 可选 support set
  ├─ 仅标注节点之间的边
  ├─ 按年龄分位数构建快照 (8 snapshots)
  ├─ 简化的位置编码 (度数代理)
  └─ Full-batch 训练 (仅标注节点)
```

**核心差距**: 标注节点之间的边只占全图边的一小部分。GNN 的消息传递需要完整的邻居信息才能有效学习节点表示。

## 修复方案

### Phase 1: 全图预处理 — 关键突破

**新建 `code/gnn/builders/full_graph.py`**

参考 preprocess_twibot20.py 的做法，为全部 229,580 节点:

1. **加载全部节点**: 从 node.json 流式读取全部用户（不仅是标注节点）
2. **构建全图边缘**: 从 edge.csv 读取全部 follow/friend 边，使用全局索引
3. **构建基于日期的快照**: 
   - 使用 `created_at` 字段，按月分组
   - 每个快照包含到该日期为止存在的所有节点和边（单调累积）
   - 生成 exist_nodes 掩码
4. **位置编码**: 使用 networkx 计算 clustering coefficient 和 bidirectional links ratio
5. **缓存**: 将预处理结果保存为 .pt 文件，避免重复计算

**关键设计决策**: 
- 特征维度匹配: 参考用 768d Transformer 嵌入，我们先用 `--disable-transformer` 模式下的数值特征维度
- 快照窗口: 支持 `--botdgt-window-size` 参数（默认 -1 = 全部快照，与参考一致）

### Phase 2: BotDGT 全图训练适配

**修改 `code/gnn/models/botdgt.py`**:
- forward 接受全图节点特征 (N_all, dim) 和快照 edge_index
- 结构层编码全部 N_all 个节点
- 通过 labeled_mask 提取标注节点用于后续时序编码和 loss 计算
- 支持 exist_nodes

**修改 `code/gnn/train.py`**:
- 新增 `full_graph_mode` 训练路径
- 支持 `labeled_mask` 参数，结构编码在全图上进行，loss 仅在标注节点上计算
- 分离全图前向传播和标注节点 loss 计算

**修改 `code/gnn/run.py`**:
- BotDGT 调用全图预处理，获取全图数据
- 传入 `labeled_mask` 等额外参数

### Phase 3: BotSAI 调优

**修改 `code/gnn/models/botsai.py`**:
- 检查与论文描述的一致性
- 调整默认参数匹配论文（invariant_weight=0.05 等）
- 考虑也使用全图结构编码（与 BotDGT 类似，但单快照）

### Phase 4: 参数和种子对齐

**修改 `code/config.py`**:
- 新增 `botdgt_window_size: int = -1` (全部快照)
- 新增 `botdgt_neighbor_samples: tuple = (2560, 2560)`
- BotDGT 种子使用 1234（参考值）
- BotSAI 种子使用适当值

**修改 `code/cli.py`**:
- 新增 `--botdgt-window-size` 参数

## 涉及文件

| 文件 | 操作 |
|------|------|
| `code/gnn/builders/full_graph.py` | **新建** — 全图快照预处理 |
| `code/gnn/models/botdgt.py` | 修改 — 支持全图节点 + labeled_mask |
| `code/gnn/train.py` | 修改 — 支持全图模式 |
| `code/gnn/run.py` | 修改 — BotDGT 接入全图数据 |
| `code/config.py` | 修改 — 新参数 + 种子对齐 |
| `code/cli.py` | 修改 — 新 CLI 参数 |

## 验证方式

```bash
# 第一步: 验证全图预处理可运行
python -m code prepare

# 第二步: 验证 BotDGT + BotSAI 效果
python -m code run-all --disable-transformer --skip-node2vec

# 第三步: 完整运行（含 Transformer）
python -m code run-all
```

检查要点:
- BotDGT test_f1 >= 0.86 (接近论文 0.8689)
- BotSAI test_f1 >= 0.87 (接近论文 0.8738)
- 全图预处理内存不超限
- 训练速度可接受
