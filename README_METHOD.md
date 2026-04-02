# 方法文档

## 1. 方法设计不再只围绕“谁分数最高”

本项目现在用统一框架回答三个研究问题：

1. 哪类信息更有效？
2. 不同方法差异在哪里？
3. 结果是否可解释？

因此，方法文档的组织顺序也对应调整为：

- 数据如何被整理成统一输入；
- 六类方法如何映射到研究问题；
- 解释性分析如何和实验结果接起来；
- 哪些代码模块分别负责训练、解释和可视化。

## 2. 数据组织与输入表示

### 2.1 原始输入文件

项目直接使用 `TwiBot-20` 原始文件：

- `data/label.csv`
- `data/split.csv`
- `data/edge.csv`
- `data/node.json`

### 2.2 为什么使用流式解析

`node.json` 体积很大，因此代码不会一次性全部读入内存，而是流式解析 JSON 数组。

对应实现：

- `code/data.py`

### 2.3 用户级样本包含什么

数据准备结束后，用户级样本会统一整理为：

- `user_id`
- `split`：`train / val / test / support`
- `label_id`：`机器人 / 人类`
- `description_text`
- `tweet_text`
- `combined_text`
- 用户属性特征
- 图结构统计特征

### 2.4 为什么保留 support 节点

`support` 节点没有分类标签，但它们仍然是图的一部分。对于 `GCN`、`GAT`、`BotRGCN` 这类图神经网络，support 节点能提供邻居上下文，因此不能简单删除。

## 3. 六类方法如何映射到三个问题

### 3.1 问题一：哪类信息更有效？

这一步首先看单源方法：

- F：只看用户属性
- T：只看文本语义
- G：只看图结构

然后再看双源和三源方法：

- FT：属性 + 文本
- FG：属性 + 图
- FTG：属性 + 文本 + 图

因此，“哪类信息更有效”不是只比较一个模型，而是比较六类方法族之间的表现，以及加入某类信息后平均会带来多少增益。

### 3.2 问题二：不同方法差异在哪里？

这里主要看两层差异：

#### 第一层：信息差异

例如：

- 从 `text_only` 到 `feature_text`，是在“只用文本”基础上增加了用户属性；
- 从 `feature_text` 到 `feature_text_graph`，是在“属性 + 文本”基础上增加了图结构。

这层差异由：

- `source_contribution_details.csv`

来描述。

#### 第二层：融合机制差异

即使都使用特征、文本、图三类信息，不同方法仍然会不同：

- `feature_text_graph_tfidf_node2vec_logistic_regression`
- `feature_text_graph_gcn`
- `feature_text_graph_gat`
- `feature_text_graph_botrgcn`

这里的差异来自：

- 简单拼接 vs 图传播
- 普通图卷积 vs 注意力传播
- 同质传播 vs 关系感知传播

### 3.3 问题三：结果是否可解释？

这一步不只看分数，而是做两类解释：

#### 信息源层面的解释

- 平均增益：加入某类信息后平均提升多少；
- 消融实验：在最佳可解释融合基线中，移除某类信息后损失多少。

#### 信号层面的解释

- 用户属性：模型主要在看哪些账号画像异常；
- 文本语义：模型主要抓哪些词或短语；
- 图结构：模型主要抓哪些关系模式。

## 4. 六类方法本身的角色

## 4.1 基于特征（F）

### `feature_only_logistic_regression`

- 只使用用户属性特征；
- 适合作为最直接、最可解释的基线；
- 有助于回答“单看账号画像是否足够识别机器人”。

### `feature_only_random_forest`

- 仍然只使用用户属性；
- 允许非线性关系；
- 可直接输出特征重要性，适合做解释性图表。

## 4.2 基于文本（T）

### `text_only_tfidf_logistic_regression`

- 把简介和 tweet 合并为文本；
- 使用 TF-IDF；
- 适合抓关键词和短语层面的可解释信号。

### `text_only_transformer_logistic_regression`

- 使用 SentenceTransformer 生成稠密文本表示；
- 更偏语义相似性；
- 适合与 TF-IDF 对照，观察“关键词”和“语义嵌入”谁更有效。

## 4.3 基于图（G）

### `graph_only_structure_random_forest`

- 只使用结构统计特征；
- 用来回答“网络连接形态本身是否足够区分机器人和人类”。

### `graph_only_node2vec_logistic_regression`

- 用 `Node2Vec` 学习节点在图中的结构角色；
- 比单纯度数统计更强调整体拓扑位置。

## 4.4 基于特征和文本（FT）

### `feature_text_tfidf_logistic_regression`

- 将用户属性与 TF-IDF 拼接；
- 是最直接的“结构化特征 + 文本关键词”融合基线。

### `feature_text_transformer_logistic_regression`

- 将用户属性与 Transformer 文本嵌入拼接；
- 检查“文本语义与结构化属性是否互补”。

## 4.5 基于特征和图（FG）

### `feature_graph_random_forest`

- 将用户属性与图统计特征一起输入；
- 适合观察“加图统计后是否稳定提升”。

### `feature_graph_node2vec_logistic_regression`

- 将用户属性、图统计和图嵌入一起输入；
- 用于观察“账号画像 + 网络位置”是否互补。

## 4.6 基于特征、文本和图（FTG）

### `feature_text_graph_tfidf_node2vec_logistic_regression`

- 把属性、TF-IDF 文本、Node2Vec 一起拼接；
- 是非 GNN 的可解释融合基线；
- 适合做消融分析。

### `feature_text_graph_gcn`

- 输入 description、tweet、数值属性、类别属性和边；
- 用 `GCNConv` 在图上传播；
- 对应最基础的图神经融合方式。

### `feature_text_graph_gat`

- 输入结构与 GCN 相同；
- 用 `GATConv` 对不同邻居分配不同权重；
- 适合观察注意力传播是否优于普通卷积。

### `feature_text_graph_botrgcn`

- 输入结构与 GCN/GAT 相同；
- 用 `RGCNConv` 区分 `follow` 与 `friend` 等关系类型；
- 适合回答“图结构是否只有在关系感知传播里才真正发挥作用”。

## 5. 解释性分析模块现在负责什么

对应实现：

- `code/interpretation.py`

它现在负责三类输出：

### 5.1 信息源平均增益

输出：

- `artifacts/tables/source_contribution_summary.csv`
- `artifacts/tables/source_contribution_details.csv`

作用：

- 说明加入用户属性、文本语义、图结构后，平均会带来多大提升；
- 说明哪些方法差异来自“增加了什么信息”。

### 5.2 融合模型消融

输出：

- `artifacts/tables/source_ablation.csv`

作用：

- 说明在最佳可解释融合基线中，哪类信息是核心输入，哪类信息只是边际补充。

### 5.3 解释性信号表

输出：

- `artifacts/tables/feature_signals.csv`
- `artifacts/tables/text_signals.csv`
- `artifacts/tables/graph_signals.csv`

作用：

- 用户属性：哪些账号特征更像机器人，哪些更像人类；
- 文本：哪些关键词/短语更像机器人，哪些更像人类；
- 图结构：哪些结构统计模式更像机器人，哪些更像人类。

## 6. 可视化模块现在回答什么

对应实现：

- `code/visualization.py`

现在新增并重点维护三张问题导向图：

- `artifacts/figures/information_effectiveness.png`
  回答“哪类信息更有效”。

- `artifacts/figures/method_differences.png`
  回答“不同方法差异在哪里”。

- `artifacts/figures/explainability_signals.png`
  回答“结果是否可解释”。

- `artifacts/figures/feature_signal_map.png`
  把“重要特征”与“机器人 / 人类方向差异”放在一张图里。

- `artifacts/figures/embedding_separation_map.png`
  展示文本语义嵌入和图嵌入在二维空间里的类分布与重叠程度。

- `artifacts/figures/local_network_patterns.png`
  展示高置信机器人 / 人类的局部网络子图和关注 / 好友关系模式。

当前保留的核心图主要就是上面这几张，目的是让汇报时始终围绕“整体表现 -> 信息贡献 -> 可解释性”这一条主线展开，而不是再回到零散的中间分析图。

## 7. 报告模块现在怎么组织

对应实现：

- `code/reporting.py`

生成的 `artifacts/report.md` 现在不再按“先贴一堆排行榜，再做零散解释”的方式组织，而是直接按三个问题展开：

1. 哪类信息更有效？
2. 不同方法差异在哪里？
3. 结果是否可解释？

这样 `README`、方法文档、可视化、自动报告会保持同一条逻辑主线，而不是各写各的。
