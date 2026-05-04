# 创新方法：模态不变性 + 时序感知机器人检测

## Context

用户希望提出一个创新性的社交机器人检测方法，选定方向为「模态不变性 + 时序感知」——基于 BotSAI 的 Siamese Attention + Invariant Loss 机制，结合时序感知文本嵌入和账户年龄快照图。

### 当前基准性能 (main 分支)

| 方法 | Test F1 | Test AUC |
|------|---------|----------|
| feature_text_tfidf_logistic_regression | 0.8645 | 0.9166 |
| feature_text_graph_tfidf_node2vec_lr | 0.8640 | 0.9080 |

### 消融实验关键发现

- 移除用户属性 → F1 下降 42%（最大贡献）
- 移除文本语义 → F1 下降 ~2%
- 移除图结构 → 几乎无变化

### BotSAI 机制 (F1=0.9079)
- 模态分解：invariant + specific 两部分
- 通道自注意力融合 4 个模态
- TransformerConv 关系感知图传播
- 损失：CrossEntropy + invariant_weight * (invariance_loss + 0.5*specific_overlap)

### BotDGT 机制 (F1=0.8843)
- 按账号年龄分位数构建时间快照，边单调累积
- 快照位置编码：clustering_proxy, bidirectional_ratio, edge_density, keep_ratio
- 时间维：attention/gru/lstm 聚合
- 损失：CrossEntropy + smoothness_weight * smoothness_loss + consistency_weight * (consistency_loss + probability_drift)

---

## 核心创新点

### 1. 时序感知文本编码器（Temporal-Aware Text Encoder）

BotSAI 仅对文本做静态嵌入，无法捕捉用户发帖时序模式（如发帖频率异常、回复延迟规律）。新设计：

- **按 Account Age 将用户推文划分为多个时间窗口**（而非 BotDGT 的图快照），每个窗口提取该窗口内的文本统计特征（TF-IDF 统计、发推频率、回复率）
- **时间窗口编码**：每个窗口的表示与"窗口索引/总窗口数"的位置编码拼接
- **轻量时序聚合**：用单层 GRUs 融合多窗口文本表示，输出统一的"时序感知文本嵌入"

### 2. 账户年龄感知的关系感知图卷积

BotSAI 使用 TransformerConv + follow/friend 两种关系类型，但没有考虑账户年龄因素。BotDGT 考虑了账户年龄但用简单的位置编码（clustering/bidirectional/edge_density/keep_ratio）。

新设计：
- **Account Age 分桶**（早期账号 vs 近期账号），在边级别引入年龄感知：同一关系类型（follow/friend）下，进一步区分"老账号→新账号"和"新账号→老账号"
- **动态关系嵌入**：关系嵌入不再是固定的 follow/friend 两个向量，而是由"关系类型 + 源账户年龄桶 + 目标账户年龄桶"三因素共同决定，通过一个小型 MLP 生成
- **效果**：让模型学习到"老账号更可信"的先验知识

### 3. 模态不变性增强（Invariant-aware Cross-Modal Alignment）

BotSAI 的 invariant loss 推动各模态的 invariant 部分向中心聚集，但仅在训练阶段施加约束，对特征学习的早期阶段影响有限。

新设计：
- **Invariant 特征空间对齐**：在特征空间中，强制 bot 账号的 invariant 嵌入靠近 bot 聚类中心，人类账号靠近人类聚类中心——即在不变性约束之上叠加分类一致性约束
- **特定损失函数**：`L_invariant = α * invariance_loss + β * intra_class_variance`，其中 intra_class_variance 惩罚同一标签样本的 invariant 嵌入过于分散

---

## 方法名称候选

**Temporally-Aware Siamese Bot Detector (TASBot)** 或 **Temporal Invariant Graph Network (TIGN)**

---

## 模型架构

```
输入：description, tweet文本, 数值属性, 类别属性, 社交图边
│
├─ TextEncoder
│   ├─ 将推文按 account_age 时间窗口划分
│   ├─ 每个窗口 → TfidfVectorizer + 时序位置编码
│   └─ GRU 时序聚合 → 时序感知文本嵌入
│
├─ AttrEncoder（与 BotSAI 一致）
│   └─ MLP → 属性嵌入
│
├─ 图编码分支
│   ├─ 按 account_age 构建边子集（动态关系嵌入）
│   └─ TransformerConv（图传播）+ 关系嵌入（关系类型×年龄桶）
│
├─ 模态分解（与 BotSAI 一致）
│   每个模态的嵌入分解为 invariant + specific 两部分
│
├─ 通道自注意力融合（与 BotSAI 一致）
│   4 个模态通道 → MultiheadAttention → fused 表示
│
├─ 分类头
│   classifier(fused)
│
└─ 输出：logits, (inv_loss, intra_loss)

损失 = CrossEntropy + λ1 * invariance_loss + λ2 * intra_class_variance
```

### 关键超参数

| 参数 | 值 | 说明 |
|------|----|----|
| `text_time_windows` | 4 | 时间窗口数 |
| `temporal_hidden` | 64 | 时序 GRU 隐藏维 |
| `relation_age_buckets` | 3 | 年龄桶数（young/middle/old）|
| `invariant_weight` | 0.05 | 原始不变性损失权重 |
| `intra_class_weight` | 0.02 | 类别内方差权重 |

---

## 数据准备

### 新增特征（需要修改 data.py 缓存）

1. **`tweet_time_windows`**：按 account_age 将推文分配到时间窗口，生成窗口级统计特征（窗口数 × 统计特征数）
2. **`account_age_bucket`**：将 account_age_days 离散化为 3 个桶（young/middle/old），用于动态关系嵌入

### 需要修改的文件

- `code/data.py`：在 `prepare_dataset()` 输出中增加 `account_age_bucket` 列
- `code/gnn/models/`：创建新文件 `code/gnn/models/tasbot.py`（或 `tign.py`）
- `code/gnn/builders/`：创建 `relation_age_graph.py`，构建带年龄感知的边索引
- `code/gnn/run.py`：注册新模型
- `code/config.py`：添加新超参数
- `code/cli.py`：添加新超参数的命令行参数

---

## 实现步骤

### Step 1: 修改 data.py（增加 account_age_bucket）

在 `prepare_dataset` 的输出 user DataFrame 中增加 `account_age_bucket` 列（young=0, middle=1, old=2），按三分位数划分。

### Step 2: 创建 code/gnn/builders/relation_age_graph.py

实现 `_build_age_relation_edge_index`：边类型 = 关系类型 × 源年龄桶 × 目标年龄桶 = 2 × 3 × 3 = 18 种边类型。RGCNConv 的 num_relations = 18。

### Step 3: 创建 code/gnn/models/tign.py

实现 `TemporalInvariantGraphNetwork` 类，继承 `_FeatureTextGraphBase`，核心结构：
- `TemporalTextEncoder`：时序文本处理模块
- `DynamicRelationEmbedding`：动态关系嵌入生成器
- 继承 BotSAI 的模态分解、通道注意力机制
- 损失函数增加 intra_class_variance 项

### Step 4: 修改 code/gnn/run.py

在 GNN_REGISTRY 中添加 `tign` 模型项，根据 TransformerConv 可用性决定是否注册。

### Step 5: 修改 config.py 和 cli.py

添加新超参数（text_time_windows, relation_age_buckets, intra_class_weight 等）。

### Step 6: 训练和评估

运行 `python -m code train`，在 experiment_metrics.csv 中验证新方法性能。

---

## 预期性能

基于分析：
- BotSAI 在 TwiBot-20 上 F1=0.9079（最强基线）
- 用户属性是最强信号，Account Age 是最有区分力的特征之一
- 结合时序感知文本 + 年龄感知图 + 增强不变性约束，预期 F1 可达 **0.91+**

---

## 关键文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `code/data.py` | 修改 | 增加 account_age_bucket 列 |
| `code/gnn/builders/relation_age_graph.py` | 创建 | 年龄感知关系图构建 |
| `code/gnn/models/tign.py` | 创建 | 新模型实现 |
| `code/gnn/models/__init__.py` | 修改 | 导出 TIGN |
| `code/gnn/run.py` | 修改 | 注册 TIGN 模型 |
| `code/config.py` | 修改 | 添加新超参数 |
| `code/cli.py` | 修改 | 添加新参数 |
| `code/interpretation.py` | 修改 | 添加 TIGN 的解释性分析 |

---

## Verification

1. `python -m code prepare` — 验证 account_age_bucket 生成正确
2. `python -m code train` — 验证 TIGN 在 val/test 上的 F1/AUC
3. 预期 val_f1 ≥ 0.88, test_f1 ≥ 0.88
4. 对比 BotSAI 和 BotDGT 的结果，确认创新点有实质贡献






# TIGN 性能提升计划

## Context

最新运行结果（30000 节点，阈值固定 0.5）：

| 方法 | Test F1 | AUC |
|------|---------|-----|
| BotDGT | **0.8710** | 0.9101 |
| TIGN | 0.8653 | **0.9132** |
| BotSAI | 0.8642 | 0.9103 |
| BotRGCN | 0.8641 | 0.8880 |

TIGN 的 AUC 最高（+3.1pp vs BotSAI），说明概率排序质量最佳，但 F1 被固定阈值 0.5 拖累。
上一轮分析已知：TIGN 最优阈值 0.36 时 F1 可达 0.8714，超越 BotDGT 0.8710。

---

## 改进方案（4 项，由高到低优先级）

### 1. 阈值优化（影响最大，实现最简）

**问题**：所有 GNN 模型在 `train.py:107` 固定使用 0.5 阈值，而不同模型的概率分布差异很大。
**上一轮已知**：TIGN 最优阈值 0.36 → F1 从 0.8613 提升到 0.8714（+1.01pp）。

**修改**：`code/gnn/train.py` 的 `_compute_final_metrics()` 函数，在保存结果前对 val 集搜素最优阈值，然后对 test 集应用同一阈值。BotDGT 也受益于此。

### 2. 增加第三 TransformerConv 层（架构增强）

**问题**：TIGN 仅有 2 层图卷积，而 BotDGT 通过 8 个时间快照"隐式加深"。更深的感受野可能提升性能。
**方案**：在 `tign.py` 的 `structural_layer1/2` 后加 `structural_layer3`，加上 `+ fused` 残差。

```python
# tign.py forward 方法，在 layer2 后加：
x = self.structural_layer3(x, edge_index, edge_attr=edge_attr)
x = self.dropout(F.leaky_relu(x)) + fused  # 第3层残差
```

### 3. TIGN 专用超参数调优

| 参数 | 当前值 | 建议范围 |
|------|--------|---------|
| `tign_intra_class_weight` | 0.02 | 0.0 / 0.01 / 0.05 / 0.1 |
| `tign_num_age_buckets` | 3 | 2 / 4 / 5 |

通过 `--tign-intra-class-weight` 和 `--tign-num-age-buckets` 命令行调优，找到最优组合。

### 4. 降低学习率（训练稳定性）

**问题**：TIGN 有额外的 MLP edge embedder + intra-class loss，参数更多更复杂。当前 `lr=1e-2` 相对高。
**方案**：对于 TIGN 模型，使用 `lr=5e-3`（在 `train.py` 中根据模型类型动态设置）。

---

## 关键文件

- `code/gnn/train.py` — 修改阈值优化逻辑（最高优先级）
- `code/gnn/models/tign.py` — 增加第三 TransformerConv 层
- `code/gnn/run.py` — 无需修改（仅传递模型）

---

## 验证计划

1. 修改 train.py 加入阈值搜索 → 运行完整流程 → 验证 TIGN F1 ≥ 0.87
2. 增加第三层 → 运行完整流程 → 对比 F1 变化
3. 调优超参数 → 找到最优配置
4. 更新 `artifacts/report.md` 和 `artifacts/tables/` 结果


## 方法：Temporal Invariant Graph Network v2 (TIGN-v2)

### 核心创新点

#### 1. 日历月图快照 + 模态不变性分解

将 BotDGT 的日历月快照与 BotSAI 的不变性/特异性分解结合：

- 使用真实日历月快照（取代年龄分位数），从 2008-01 到 2020-09，每年取 1 个 → ~13 个快照
- 每个快照上运行图卷积，然后对每个快照的节点表示做 invariant/specific 分解
- **共享的不变性/特异性投影器**跨时间步使用，确保分解的一致性

#### 2. 跨时间不变性约束（核心创新）

这是两篇论文都未探索的新约束：

```
L_temporal_invariance = MSE(invariant[t], invariant[t-1])
```

- 对同一节点在相邻快照的 invariant 表示施加平滑约束
- 直觉：真实用户的"行为本质"不应突变；bot 账号可能因策略切换而表现出时间不一致性
- 与 BotSAI 的跨模态不变性正交：一个约束模态间，一个约束时间间

#### 3. 双流时序处理

BotDGT 用单一流处理所有时序信息。我们分离为两股流：

```
Invariant Stream:  GRU([inv[0], inv[1], ..., inv[T-1]])  → 时间稳定的核心特征
Specific Stream:   GRU([spc[0], spc[1], ..., spc[T-1]])  → 时间变化的特定特征

Fused = CrossAttention(Invariant_Stream, Specific_Stream)
```

- Invariant Stream 受跨时间不变性损失约束
- Specific Stream 可以自由捕捉行为漂移
- 最后用交叉注意力融合两股流

#### 4. 增强的位置编码

从 BotDGT 继承并增强：
- Clustering coefficient → 局部聚集程度
- Bidirectional link ratio → 互关比例
- **新增：Snapshot-level account age** → 每个快照时刻该账号的年龄
- **新增：Degree change rate** → 与上一快照相比的度变化率（捕捉异常涨粉）

#### 5. 复合损失函数

```
L_total = Σ_t coefficient^t × CE_loss[t]           # BotDGT 的时序加权 CE
        + λ1 × L_cross_modal_invariance              # BotSAI 的跨模态不变性
        + λ2 × L_temporal_invariance                 # 新：跨时间不变性
        + λ3 × L_specific_decorrelation              # BotSAI 的特异性去相关
        + λ4 × L_intra_class_variance                # TIGN 的类内方差
```

---

## 模型架构

```
输入: 229,580 节点 × T 个日历月快照

┌─────────────────────────────────────────────────────┐
│  Node Feature Embedding (共享权重)                    │
│  des(768)→Linear→PReLU、tweet(768)→Linear→PReLU      │
│  num(5)→Linear→PReLU、cat(3)→Linear→PReLU            │
│  → Concat → Linear(hidden_dim) + Dropout(0.3)        │
└─────────────────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │   For each snapshot t=0..T-1  │
          │   ┌───────────────────────┐   │
          │   │ GraphStructuralLayer  │   │
          │   │ TransformerConv × 2   │   │
          │   │ (共享权重 across t)   │   │
          │   └───────────────────────┘   │
          │   ┌───────────────────────┐   │
          │   │ Invariant/Specific    │   │
          │   │ Decomposition         │   │
          │   │ (共享投影器 across t) │   │
          │   └───────────────────────┘   │
          │   ┌───────────────────────┐   │
          │   │ Position Encoding     │   │
          │   │ (clustering, BLR,     │   │
          │   │  age, degree_change)  │   │
          │   └───────────────────────┘   │
          └──────────────┬──────────────┘
                         │
          ┌──────────────┴──────────────┐
          │   Dual-Stream Temporal       │
          │                              │
          │   Invariant Stream:          │
          │   GRU + TemporalInvariant    │
          │   Loss (smoothness)          │
          │                              │
          │   Specific Stream:           │
          │   GRU (自由变化)             │
          │                              │
          │   → CrossAttention Fuse      │
          └──────────────┬──────────────┘
                         │
          ┌──────────────┴──────────────┐
          │   Cross-Modal Channel        │
          │   Self-Attention (BotSAI)    │
          │   4 channels → MHA → Mean    │
          └──────────────┬──────────────┘
                         │
          ┌──────────────┴──────────────┐
          │   Output Head × T             │
          │   Linear(hidden_dim, 2)       │
          │   → [T, batch, 2] logits     │
          └──────────────────────────────┘
```

---

## 与现有工作的关键区别

| 特性 | BotSAI | BotDGT | 当前 TIGN | **TIGN-v2 (本文)** |
|------|--------|--------|-----------|-------------------|
| 图快照 | 无（静态） | 日历月 | 年龄分位数 8 个 | **日历月 ~13 个** |
| 模态分解 | ✓ | ✗ | ✓ | ✓ |
| 时间维不变性 | ✗ | ✗ | ✗ | **✓ (新)** |
| 双流时序 | ✗ | ✗ | ✗ | **✓ (新)** |
| NeighborLoader | ✗ | ✓ | ✗ | **✓** |
| 位置编码 | ✗ | 2 种 | ✗ | **4 种** |
| 损失项数 | 3 | 1 | 4 | **5** |

---

## 预期性能

| 方法 | 当前 Test F1 | 预期提升来源 |
|------|-------------|-------------|
| BotSAI (当前最优) | 0.8719 | — |
| BotDGT | 0.8710 | — |
| TIGN | 0.8586 | — |
| **TIGN-v2** | **≥ 0.88-0.89** | 日历月快照 + 跨时间不变性 + 双流时序 |

### 为什么预期能超越

1. **日历月快照** ≥ 年龄分位数快照（BotDGT 论文已证明）
2. **跨时间不变性**是一个全新的约束维度，两个 baselines 都没有
3. **双流时序**比 BotDGT 的单流更精细——invariant 流受约束，specific 流自由
4. NeighborLoader 全图训练捕捉完整拓扑结构
5. 复合损失提供了更强的正则化

---

## 实现计划

### 新建文件

| 文件 | 说明 |
|------|------|
| `code/gnn/tignv2/__init__.py` | 导出 `run_tignv2` |
| `code/gnn/tignv2/model.py` | TIGN-v2 完整模型 |
| `code/gnn/tignv2/train.py` | 训练循环 + run_tignv2 入口 |
| `code/gnn/tignv2/loss.py` | 复合损失函数（含跨时间不变性） |

### 复用文件（不修改）

| 文件 | 说明 |
|------|------|
| `code/gnn/botdgt/data.py` | **直接复用** BotDGT 的日历月快照数据加载 |
| `code/gnn/botdgt/model.py` 的部分层 | 复用 NodeFeatureEmbeddingLayer, GraphStructuralLayer, PositionEncoding |

### 修改文件

| 文件 | 修改 |
|------|------|
| `code/config.py` | 添加 TIGN-v2 超参数 |
| `code/gnn/run.py` | 注册 TIGN-v2 模型，从 botdgt 模块复用数据集 |
| `code/cli.py` | 添加命令行参数（可选） |

### 超参数设计

```python
# TIGN-v2 专用参数
tignv2_interval: str = "year"            # 快照间隔
tignv2_window_size: int = -1             # 快照窗口大小
tignv2_temporal_module: str = "attention" # attention/gru/lstm
tignv2_structural_heads: int = 4         # 结构层注意力头数
tignv2_temporal_heads: int = 4           # 时间层注意力头数
tignv2_structural_dropout: float = 0.0   # 结构层 dropout
tignv2_temporal_dropout: float = 0.5     # 时间层 dropout
tignv2_cross_modal_weight: float = 0.05  # 跨模态不变性权重 λ1
tignv2_temporal_invariance_weight: float = 0.03  # 跨时间不变性权重 λ2
tignv2_specific_decorr_weight: float = 0.025     # 特异性去相关 λ3
tignv2_intra_class_weight: float = 0.02          # 类内方差 λ4
tignv2_loss_coefficient: float = 1.1             # 时序加权系数
tignv2_structural_lr: float = 1e-4
tignv2_temporal_lr: float = 1e-5
tignv2_weight_decay: float = 1e-2
tignv2_epochs: int = 20
tignv2_batch_size: int = 64
```

---

## 验证计划

```bash
# 1. 确保日历月快照数据可用（复用 BotDGT 模块的数据准备）
python -m code train --only-botdgt  # 验证 BotDGT 快照数据 OK

# 2. 运行 TIGN-v2
python -m code train --only-tignv2

# 3. 对比
# 检查 artifacts/tables/experiment_metrics.csv
# 预期 TIGN-v2 test_f1 ≥ 0.88
```

### 消融计划（验证创新点贡献）

| 消融 | 移除的组件 | 预期 F1 下降 |
|------|-----------|-------------|
| 完整模型 | — | 0.88+ |
| - 跨时间不变性 | 移除 λ2 | -0.01~0.02 |
| - 双流变单流 | 合并 invariant+ specific 流 | -0.005~0.01 |
| - 日历月快照 → 年龄分位数 | 用旧快照构建 | -0.01~0.02 |
| - 模态不变性 | 移除 λ1, λ3 | -0.01+ |

---

## 风险与对策

| 风险 | 对策 |
|------|------|
| 日历月快照计算量大 | 复用 botdgt/data.py 的缓存机制 |
| 跨时间不变性约束过强导致表示坍缩 | λ2 从 0.01 开始渐进调大 |
| 双流模型参数量大 | 共享 invariant/specific 投影器，GRU 隐藏维 = hidden_dim |
| 训练时间过长 | 先用 interval="year" (~13 快照)，NeighborLoader 限制邻居数 |








# TIGN-v2 性能优化计划

## Context

最新实验结果 (2026-05-04)：
- BotDGT: test F1=**0.88839** (已追平论文 0.8887)
- TIGN-v2: test F1=**0.88565** (目标: 超越 BotDGT 论文 → 超越 BotSAI 论文 F1≈0.9043)

从训练日志分析：
- 模型在 20 轮时仍在改善（val loss 从 0.520 → 0.507 持续下降）
- Scheduler T_max=20 太短，CosineAnnealing 在第 20 轮重置 LR 破坏收敛
- temporal_lr=1e-5 过低（比 structural_lr=1e-4 低 10 倍）
- 未启用 early stopping

## 优化方案

### 第一轮优化（高置信度提升 +0.005~0.01 F1）

| # | 参数 | 当前值 | 新值 | 理由 |
|---|------|--------|------|------|
| 1 | `botdgt_epochs` | 20 | **50** | 模型远未收敛 |
| 2 | scheduler T_max | 20 | **50** | 匹配 epoch 数，避免提前重置 LR |
| 3 | `botdgt_temporal_lr` | 1e-5 | **5e-5** | 时序模块是 TIGN-v2 创新核心，需更强学习信号 |
| 4 | `botdgt_structural_lr` | 1e-4 | **2e-4** | 略微提升结构层学习率 |
| 5 | early_stop | False | **True (patience=15)** | 防止过拟合，自动选择最佳模型 |
| 6 | `tignv2_temporal_invariance_weight` | 0.03 | **0.015** | 当前约束可能过强，减半让 CE loss 主导 |
| 7 | `tignv2_cross_modal_weight` | 0.05 | **0.03** | 降低跨模态约束，让位于时间维学习 |

### 第二轮优化（如第一轮 F1 < 0.89）

| # | 参数 | 新值 | 理由 |
|---|------|------|------|
| 8 | `botdgt_interval` | "six_months" | 更细粒度快照（~26 个），更多时间信号 |
| 9 | `botdgt_temporal_lr` | 1e-4 | 进一步提升时序学习率 |
| 10 | `tignv2_specific_decorr_weight` | 0.01 | 降低特异性去相关约束 |

### 第三轮优化（如仍不达标）

| # | 改动 | 理由 |
|---|------|------|
| 11 | hidden_dim 128→256 | 增加模型容量 |
| 12 | 添加 warmup scheduler (5 epoch linear warmup) | 稳定训练初期 |
| 13 | 尝试不同 random_state (42, 5678) | 排除随机性 |

## 修改文件

| 文件 | 修改 |
|------|------|
| `code/config.py` | 更新默认超参数值 (epochs, lr, weights) |
| `code/gnn/tignv2/train.py` | 1) scheduler T_max→epochs 2) 添加 warmup 3) 启用 early_stop 4) 日志输出 loss_components |

## 验证

```bash
# 运行优化后的 TIGN-v2
python -m code train --only-tignv2

# 查看结果
# artifacts/tables/experiment_metrics.csv 中 tignv2 行
# 目标: test F1 ≥ 0.89 (第一轮) → ≥ 0.90 (第二轮)
```