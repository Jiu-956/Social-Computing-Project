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