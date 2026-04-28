# 方法设计与实现

本文档说明每个方法的核心机制与实现要点。阅读前提：已了解 README.md 中的项目结构与研究问题背景。

## 1. 方法族映射

每个具体方法映射到一个信息源方法族（F/T/G/FT/FG/FTG），方便横向对比。

| 方法名 | 方法族 | 信息源 |
|--------|--------|--------|
| `feature_only_logistic_regression` | F | 用户属性 |
| `feature_only_random_forest` | F | 用户属性 |
| `text_only_tfidf_logistic_regression` | T | 文本（TF-IDF） |
| `text_only_transformer_logistic_regression` | T | 文本（Transformer） |
| `graph_only_structure_random_forest` | G | 图结构统计 |
| `graph_only_node2vec_logistic_regression` | G | 图嵌入（Node2Vec） |
| `feature_text_tfidf_logistic_regression` | FT | 属性 + 文本 |
| `feature_text_transformer_logistic_regression` | FT | 属性 + 文本 |
| `feature_graph_random_forest` | FG | 属性 + 图 |
| `feature_graph_node2vec_logistic_regression` | FG | 属性 + 图 |
| `feature_text_graph_tfidf_node2vec_lr` | FTG | 属性 + 文本 + 图 |
| `feature_text_graph_gcn` | FTG | 属性 + 文本 + 图 |
| `feature_text_graph_gat` | FTG | 属性 + 文本 + 图 |
| `feature_text_graph_botrgcn` | FTG | 属性 + 文本 + 图 |
| `feature_text_graph_botsai` | FTG | 属性 + 文本 + 图 |
| `feature_text_graph_botdgt` | FTG | 属性 + 文本 + 图 |
| `feature_text_graph_tign` | FTG | 属性 + 文本 + 图（创新） |

## 2. 基线方法（sklearn）

### 2.1 特征类（F）

- `feature_only_logistic_regression`：仅用数值/类别属性，最直接的基线，可解释性强
- `feature_only_random_forest`：非线性融合，可输出特征重要性

### 2.2 文本类（T）

- `text_only_tfidf_logistic_regression`：description + tweet 合并，TF-IDF 向量化，适合抓关键词
- `text_only_transformer_logistic_regression`：SentenceTransformer 稠密嵌入，更偏语义相似性

### 2.3 图类（G）

- `graph_only_structure_random_forest`：仅用度数、聚类系数等结构统计特征
- `graph_only_node2vec_logistic_regression`：Node2Vec 学习拓扑位置嵌入

### 2.4 双源融合（FT / FG）

- 属性 + TF-IDF / Transformer：结构化特征与文本关键词/语义的拼接融合
- 属性 + 图统计 / Node2Vec：账号画像与网络位置的融合

### 2.5 三源拼接基线（FTG）

- `feature_text_graph_tfidf_node2vec_lr`：属性 + TF-IDF + Node2Vec 拼接，适合做消融分析

## 3. 图神经网络（torch-geometric）

所有 GNN 模型共享同一输入结构：description 嵌入、tweet 嵌入、数值属性、类别属性、边索引。

### 3.1 `feature_text_graph_gcn`

最基础的图卷积网络。使用 `GCNConv` 在无向图上均匀聚合邻居信息，作为融合基线。

### 3.2 `feature_text_graph_gat`

用 `GATConv` 替代 GCN，通过注意力机制为不同邻居分配不同权重。适合观察注意力机制是否优于均匀聚合。

### 3.3 `feature_text_graph_botrgcn`

用 `RGCNConv` 区分边类型（follow / friend），回答"关系感知传播是否让图结构真正发挥作用"。

### 3.4 `feature_text_graph_botsai`

核心机制：
1. **模态分解**：每个模态（description / tweet / 数值属性 / 类别属性）分别投影为 invariant（不变表示）+ specific（特定表示）
2. **通道自注意力融合**：四个模态通道经 `MultiheadAttention` 融合
3. **图传播**：用 `TransformerConv` + 关系嵌入做结构传播
4. **不变性约束**：训练时加入 `inv_loss = invariant_weight * (invariance_loss + 0.5 * specific_overlap)`

### 3.5 `feature_text_graph_botdgt`

核心机制：
1. **动态快照**：按账号年龄分位数由稀到密构建 8 个图快照，边单调累积
2. **快照位置编码**：clustering_proxy、bidirectional_ratio、edge_density、keep_ratio
3. **时间维聚合**：attention / GRU / LSTM 聚合多快照表示
4. **时序约束**：smoothness_weight × 平滑损失 + consistency_weight × 一致性损失

> BotSAI / BotDGT 为"论文思想驱动的工程化实现"，用于在统一流水线中做可比实验，不等同于官方逐行复现。

### 3.6 `feature_text_graph_tign`（创新方法）

在 BotSAI 基础上引入**年龄感知动态关系嵌入**：

**动机**：BotSAI 仅区分 follow / friend 两种边类型，没有利用账户年龄信息。账户年龄是机器人检测的重要特征，老账号→新账号的边关系与新账号→老账号应有不同权重。

**边类型设计**：
- 边类型 = `(关系类型) × (源账户年龄桶) × (目标账户年龄桶)`
- 默认 3 个年龄桶（young / middle / old），共 2 × 3 × 3 = **18 种边类型**
- 年龄桶按 account_age_days 的 33%/67% 分位数划分

**动态边嵌入生成**（`tign.py` `_build_edge_attr`）：
```
边嵌入 = MLP([关系类型嵌入; 源年龄桶嵌入; 目标年龄桶嵌入])
```
即不是查表得到固定向量，而是由三者经 MLP 动态生成。

**模型结构**：
```
模态编码 → 模态分解（invariant + specific）
  → 通道自注意力融合
  → 动态边嵌入生成（关系+年龄三因素 MLP）
  → TransformerConv 图传播（含残差连接）
  → 分类头

损失 = CrossEntropy + λ₁ × 不变性损失 + λ₂ × 类别内方差
```

**关键超参数**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--tign-num-age-buckets` | 年龄桶数 | 3 |
| `--tign-intra-class-weight` | 类别内方差权重 | 0.02 |

## 4. 解释性分析

对应 `code/interpretation.py`，输出三类表：

| 输出文件 | 内容 |
|----------|------|
| `source_contribution_summary.csv` | 各信息源的平均增益 |
| `source_ablation.csv` | 融合模型中移除某信息源的损失 |
| `feature_signals.csv` | 账号特征重要性与方向 |
| `text_signals.csv` | TF-IDF 关键词权重与方向 |
| `graph_signals.csv` | 图结构统计特征重要性 |

## 5. 可视化

对应 `code/visualization.py`，围绕三个研究问题输出图表：

| 图表 | 对应问题 |
|------|---------|
| `information_effectiveness.png` | 问题一 |
| `method_differences.png` | 问题二 |
| `explainability_signals.png` | 问题三 |
| `feature_signal_map.png` | 特征重要性与方向 |
| `embedding_separation_map.png` | 嵌入空间类分布 |
| `local_network_patterns.png` | 高置信样本的局部网络 |
