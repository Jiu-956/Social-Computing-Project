# 第 3 章 方法设计

## 3.1 研究任务与方法框架

### 3.1.1 研究任务定义

本文面向 TwiBot-20 数据集上的社交机器人检测任务。设用户集合为：

\[
\mathcal{U}=\{u_1,u_2,\dots,u_N\}
\]

对于任意用户 \(u_i\)，需要结合其属性信息、文本信息以及社交关系结构信息，预测该用户属于 human 还是 bot。记类别标签为：

\[
y_i \in \{0,1\}
\]

其中：

- \(y_i=0\) 表示 human；
- \(y_i=1\) 表示 bot。

与仅依赖静态属性或静态图结构的方法不同，本文的核心目标是同时回答两个问题：第一，用户的多模态信息能否被统一建模；第二，用户进入社交网络的时间先后是否能够通过动态图快照带来额外判别能力。

### 3.1.2 方法设计动机

现有社交机器人检测方法通常存在两个局限。其一，属性、文本和图结构往往被简单拼接，难以反映不同样本上不同模态的重要性差异。其二，图结构常被视为单一静态图，忽略了用户在网络中“逐步出现”的时间过程。

针对上述问题，本文提出 Dynamic-Aware Adaptive Fusion Bot Detector。该方法包含三个核心思想：

1. 将用户属性、文本和图结构分别编码，避免不同模态相互干扰；
2. 依据用户 `created_at` 构造累积式动态图快照，建模用户随时间进入网络后的结构演化；
3. 利用带质量感知的自适应融合模块，为不同样本动态分配模态权重。

### 3.1.3 整体框架

本文方法的整体流程如下：

1. 从原始数据中对齐用户、标签与数据划分；
2. 提取用户属性特征，构建属性输入 \(\mathbf{x}^{attr}_i\)；
3. 拼接用户简介与 tweet 文本，构建文本输入 \(\mathbf{x}^{text}_i\)；
4. 根据用户创建时间生成多时刻动态图快照 \(G^{(1)},G^{(2)},\dots,G^{(T)}\)；
5. 通过属性编码器、文本编码器和动态图编码器分别得到 \(\mathbf{h}^{attr}_i\)、\(\mathbf{h}^{text}_i\) 和 \(\mathbf{h}^{dyn}_i\)；
6. 根据质量特征 \(\mathbf{q}_i\) 自适应计算三种模态的融合权重；
7. 将融合后的表示送入分类器，输出用户为 bot 或 human 的概率。

因此，本文方法是一个“多模态编码 + 动态图建模 + 自适应融合”的统一框架。

---

## 3.2 数据组织与预处理

### 3.2.1 用户集合、标签与划分对齐

原始数据集中主要使用以下文件：

- `node.json`：包含用户节点和 tweet 节点；
- `edge.csv`：包含 `follow`、`friend`、`post` 三类边；
- `label.csv`：给出用户标签；
- `split.csv`：给出训练集、验证集和测试集划分。

预处理首先读取 `split.csv` 和 `label.csv`，统一用户标识为 `user_id`，并建立监督标签：

- `human -> 0`
- `bot -> 1`

随后得到完整的用户主表 `user_table`，作为后续所有模态特征构建的基础索引表。为了保证图结构张量、属性张量和文本张量的索引统一，系统进一步建立：

- `uid2index`
- `index2uid`

从而将每一个用户映射到全局索引空间。

### 3.2.2 关系边筛选与 tweet 文本回收

`edge.csv` 中同时包含用户之间的社交关系边和用户到 tweet 的关联边。本文对三类边做如下处理：

1. `follow` 和 `friend` 边保留，用于构造用户关系图；
2. `post` 边不进入图结构建模，而仅用于回收 tweet 节点文本；
3. 对每个用户最多保留固定数量的 tweet 节点，以控制文本长度与预处理成本。

记原始用户关系图为：

\[
G=(V,E)
\]

其中 \(V\) 表示用户节点集合，\(E\) 只包含 `follow` 和 `friend` 两类用户-用户边。与此同时，每个用户还对应一个 tweet 文本集合：

\[
\mathcal{T}_i=\{t_{i1},t_{i2},\dots,t_{ik}\}
\]

这些 tweet 文本不作为图节点参与训练，而是被吸收进文本模态。

### 3.2.3 用户属性恢复与标准用户表构建

在 `node.json` 中，用户节点提供了多种 profile 信息，例如：

- `followers_count`
- `friends_count`
- `listed_count`
- `favourites_count`
- `statuses_count`
- `verified`
- `protected`
- `description`
- `username`
- `name`
- `location`
- `url`
- `created_at`

本文通过流式方式解析 `node.json`，只提取目标用户节点与已采样的 tweet 节点，从而避免在超大 JSON 文件上一次性载入全部内容。之后，将用户属性、tweet 文本、关系统计量和标签信息统一合并，形成标准用户表 `user_table.csv`。

该表不仅保存原始字段，还进一步补充如下统计信息：

- `follow_in_count`
- `follow_out_count`
- `friend_in_count`
- `friend_out_count`
- `total_in_degree`
- `total_out_degree`
- `neighbor_num`
- `in_out_ratio`
- `friend_share_out`
- `follow_share_out`
- `post_count`
- `account_age_days`
- `profile_missing_count`

它们会被后续的属性特征构建、质量特征构建和可解释性分析共同使用。

---

## 3.3 多模态特征表示

### 3.3.1 属性特征表示

本文将属性模态表示为用户级数值特征与布尔特征的拼接。对于用户 \(u_i\)，其属性特征记为：

\[
\mathbf{x}^{attr}_i=[\mathbf{x}^{num}_i;\mathbf{x}^{bool}_i]
\]

其中数值特征部分主要包括：

- `followers_count`
- `friends_count`
- `listed_count`
- `favourites_count`
- `statuses_count`
- `followers/friends`
- `friends/followers`
- `favourites/statuses`
- `account_age_days`
- `posting_intensity`
- `username_length`
- `username_digit_ratio`
- `description_length`
- `tweet_num`

其中：

\[
\text{posting\_intensity}=
\frac{\text{statuses\_count}}{\max(\text{account\_age\_days},1)}
\]

布尔特征部分主要包括：

- `verified`
- `default_profile`
- `default_profile_image`
- `has_extended_profile`
- `protected`
- `geo_enabled`
- `url_is_empty`
- `description_is_empty`

为了提升数值稳定性，本文对数值特征采用如下归一化流程：

1. 缺失值以中位数填充；
2. 对非负数值执行 `log1p` 变换；
3. 进行 z-score 标准化。

经过处理后，得到属性模态输入张量：

\[
\mathbf{X}^{attr}\in \mathbb{R}^{N\times d_{attr}}
\]

### 3.3.2 文本输入构建

对于每个用户，本文将简介和 tweet 文本统一组织为单一文本串，拼接模板为：

```text
[DESC] user_description [TWEETS] tweet_1 [SEP] tweet_2 [SEP] ... [SEP] tweet_k
```

若简介为空或 tweet 为空，则使用 `[EMPTY]` 占位，以避免空文本输入影响编码稳定性。

文本模态的设计遵循“先构造统一文本，再映射到固定维度向量”的思路。记用户 \(u_i\) 的拼接文本为：

\[
s_i = \text{Concat}(\text{description}_i,\text{tweets}_i)
\]

### 3.3.3 文本编码策略

为了兼顾表示能力与工程可复现性，本文采用离线文本向量方案。默认情况下，使用 Sentence-Transformer 对拼接文本进行编码，得到：

\[
\mathbf{x}^{text}_i \in \mathbb{R}^{d_{text}}
\]

当预训练模型无法正常加载时，系统自动回退到 `TF-IDF + TruncatedSVD` 的轻量方案。这种设计具有两点优势：

1. 训练阶段不需要在线微调大型文本编码器，显著降低实验复杂度；
2. 即使外部预训练模型不可用，整个实验流程仍能完整运行。

最终得到文本模态输入张量：

\[
\mathbf{X}^{text}\in \mathbb{R}^{N\times d_{text}}
\]

### 3.3.4 质量特征构建

为了在融合阶段判断不同模态的可靠性，本文进一步构造用户级质量特征：

\[
\mathbf{q}_i \in \mathbb{R}^{d_q}
\]

当前实现中主要包括：

- `text_len`
- `tweet_num`
- `neighbor_num`
- `missing_ratio`
- `account_age_days`
- `statuses_count`
- `exist_ratio_in_snapshots`

其中：

\[
\text{exist\_ratio\_in\_snapshots}(u_i)=
\frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[u_i\in V^{(t)}]
\]

这些特征不直接参与属性编码或图编码，而是作为融合模块的辅助输入，用于估计哪种模态对当前用户更可信。

---

## 3.4 基于用户创建时间的动态图建模

### 3.4.1 动态图构建思想

本文最关键的设计在于：动态图并不是依据 tweet 发表时间构造，而是依据用户账户的创建时间 `created_at` 构造。

其出发点是：如果一个用户在某个时间点尚未“进入”网络，那么在该时间点之前，该用户不应当出现在图结构快照中。由此，动态图更准确地表示了用户逐步进入社交网络后的结构累积过程。

### 3.4.2 累积式快照定义

给定时间边界序列：

\[
\tau_1,\tau_2,\dots,\tau_T
\]

对任意时间点 \(\tau_t\)，定义第 \(t\) 个动态图快照为：

\[
G^{(t)}=(V^{(t)},E^{(t)})
\]

其中节点集合定义为：

\[
V^{(t)}=\{u_i \mid created\_at(u_i)<\tau_t\}
\]

边集合定义为：

\[
E^{(t)}=\{(u,v)\in E \mid u\in V^{(t)} \land v\in V^{(t)}\}
\]

由此可见，本文的快照具有“累积式”特征：

- 时间越晚，快照中用户越多；
- 后续快照包含前期快照中的所有节点与其可用边；
- 最后一个快照可视为完整静态图。

### 3.4.3 时间粒度设置

本文实现支持三种时间粒度：

- `monthly`
- `quarterly`
- `yearly`

默认使用 `monthly`。在实际训练时，还可进一步从全部快照中均匀采样固定数量的 snapshot，以控制显存压力与训练时间。

### 3.4.4 快照结构特征

为了让图编码器不仅感知边连接关系，还感知节点在局部结构中的位置，本文在每个快照上为每个节点构造两个附加结构特征：

#### 1. 聚类系数

在当前快照对应的无向图上计算：

\[
\text{clustering}^{(t)}_i
\]

它反映用户周围邻域的局部闭合程度。

#### 2. 双向连接比例

在有向关系图上，统计用户的出边中有多少属于双向互联关系：

\[
\text{bidir}^{(t)}_i=
\frac{\#\{\text{mutual outgoing neighbors}\}}{\#\{\text{outgoing neighbors}\}}
\]

该特征反映用户社交关系的互惠性。

### 3.4.5 快照输出格式

每个 snapshot 最终保存以下内容：

- `edge_index`
- `edge_type`
- `exist_nodes`
- `global_index`
- `clustering_coefficient`
- `bidirectional_links_ratio`

其中 `exist_nodes` 是长度为 \(N\) 的 0/1 mask，用来指示当前节点是否已在该时刻出现。该设计使得后续图编码和时间编码都能在统一的全局节点空间上完成。

---

## 3.5 Dynamic-Aware Adaptive Fusion 模型

### 3.5.1 整体模型表示

在得到三类输入后，本文分别构建三个编码分支：

- 属性分支：得到 \(\mathbf{h}^{attr}_i\)
- 文本分支：得到 \(\mathbf{h}^{text}_i\)
- 动态图分支：得到 \(\mathbf{h}^{dyn}_i\)

然后通过自适应融合模块得到最终融合表示：

\[
\mathbf{h}^{fused}_i
\]

最后送入分类头输出：

\[
\hat{\mathbf{y}}_i \in \mathbb{R}^{2}
\]

### 3.5.2 属性编码分支

属性编码器采用两层全连接网络：

\[
\mathbf{h}^{attr}_i=f_{attr}(\mathbf{x}^{attr}_i)
\]

其结构为：

```text
Linear(d_attr -> hidden_attr)
ReLU
Dropout
Linear(hidden_attr -> d_model)
```

这一分支主要建模用户 profile 层面的静态特征。

### 3.5.3 文本编码分支

文本编码器同样采用两层 MLP：

\[
\mathbf{h}^{text}_i=f_{text}(\mathbf{x}^{text}_i)
\]

其结构为：

```text
Linear(d_text -> hidden_text)
ReLU
Dropout
Linear(hidden_text -> d_model)
```

文本分支的作用是建模用户自我描述、语言风格和 tweet 内容特征。

### 3.5.4 快照图编码分支

对于第 \(t\) 个快照，本文首先构造节点输入：

\[
\mathbf{x}^{graph,(t)}_i=
[\mathbf{x}^{attr}_i;
\text{clustering}^{(t)}_i;
\text{bidir}^{(t)}_i;
\text{exist}^{(t)}_i]
\]

这表明图分支并不是纯结构空图，而是以属性特征为基础，再叠加当前快照下的结构位置特征。

快照图编码器使用两层 GraphSAGE。对任一节点，其单层聚合形式可以写为：

\[
\mathbf{m}_i=\text{Mean}\{\mathbf{h}_j \mid j\in\mathcal{N}(i)\}
\]

\[
\mathbf{h}'_i = W_{self}\mathbf{h}_i + W_{neigh}\mathbf{m}_i
\]

在两层聚合之后，得到当前快照下的节点结构表示：

\[
\mathbf{z}^{(t)}_i \in \mathbb{R}^{d_{model}}
\]

对于当前 snapshot 中尚不存在的节点，模型利用 `exist_mask` 将其输出置零，从而避免无效噪声进入时间模块。

### 3.5.5 时间聚合模块

对每个用户 \(u_i\)，收集其在所有快照上的结构表示序列：

\[
\mathbf{Z}_i=
[\mathbf{z}^{(1)}_i,\mathbf{z}^{(2)}_i,\dots,\mathbf{z}^{(T)}_i]
\]

为了建模不同时间快照之间的重要性差异，本文采用 Temporal Self-Attention 对序列进行编码。首先利用多头自注意力得到跨时间上下文表示：

\[
\mathbf{H}_i=\text{MHA}(\mathbf{Z}_i,\mathbf{Z}_i,\mathbf{Z}_i)
\]

然后利用一个线性 pooling 层得到各时间步的打分：

\[
s_i^{(t)}=W_p \mathbf{h}_i^{(t)}
\]

在存在节点的时间步上做 masked softmax 后，得到时间权重：

\[
\beta_i^{(t)}=\text{softmax}(s_i^{(t)})
\]

最终动态图表示为：

\[
\mathbf{h}^{dyn}_i=
\sum_{t=1}^{T}\beta_i^{(t)}\mathbf{z}^{(t)}_i
\]

这里的 \(\beta_i^{(t)}\) 具有清晰的可解释性，能够反映“对某个样本而言，哪几个时间快照更重要”。

### 3.5.6 自适应融合模块

在获得三种模态表示后，本文并不直接拼接，而是使用自适应融合模块动态估计模态权重。

首先，对三种模态表示进行线性投影：

\[
\tilde{\mathbf{h}}^{attr}_i=W_a\mathbf{h}^{attr}_i
\]

\[
\tilde{\mathbf{h}}^{text}_i=W_t\mathbf{h}^{text}_i
\]

\[
\tilde{\mathbf{h}}^{dyn}_i=W_d\mathbf{h}^{dyn}_i
\]

然后，将它们与质量特征 \(\mathbf{q}_i\) 拼接：

\[
\mathbf{g}_i=
[\tilde{\mathbf{h}}^{attr}_i;
\tilde{\mathbf{h}}^{text}_i;
\tilde{\mathbf{h}}^{dyn}_i;
\mathbf{q}_i]
\]

通过 gating 网络计算三种模态的权重：

\[
[\alpha_i^{attr},\alpha_i^{text},\alpha_i^{dyn}]
=\text{softmax}(\text{MLP}(\mathbf{g}_i))
\]

进而得到融合表示：

\[
\mathbf{h}^{fused}_i=
\alpha_i^{attr}\tilde{\mathbf{h}}^{attr}_i+
\alpha_i^{text}\tilde{\mathbf{h}}^{text}_i+
\alpha_i^{dyn}\tilde{\mathbf{h}}^{dyn}_i
\]

这种设计有两个优点：

1. 不同用户可拥有不同的模态权重；
2. 权重与质量特征相关，因此更符合“模态可靠性不一致”的实际情况。

### 3.5.7 分类头

最终分类头采用两层感知机：

\[
\hat{\mathbf{y}}_i=f_{cls}(\mathbf{h}^{fused}_i)
\]

结构为：

```text
Linear(d_model -> 64)
ReLU
Dropout
Linear(64 -> 2)
```

输出用户为 human 或 bot 的 logits。

---

## 3.6 模型训练与优化

### 3.6.1 训练目标

本文采用交叉熵损失作为主损失函数：

\[
\mathcal{L}_{cls}=\text{CE}(\hat{\mathbf{y}},y)
\]

训练时仅在训练集索引对应的用户节点上计算损失。

### 3.6.2 模态与时间正则项

为了避免 gating 过早塌缩，代码中还预留了两类熵正则项。

#### 模态权重熵正则

\[
\mathcal{L}_{modal}
=-\sum_m \alpha_m \log \alpha_m
\]

#### 时间权重熵正则

\[
\mathcal{L}_{temp}
=-\sum_t \beta_t \log \beta_t
\]

总损失可写为：

\[
\mathcal{L}
=\mathcal{L}_{cls}
 + \lambda_1 \mathcal{L}_{modal}
 + \lambda_2 \mathcal{L}_{temp}
\]

在当前默认配置中：

- \(\lambda_1=0\)
- \(\lambda_2=0\)

因此第一版实验以分类性能为主要目标。

### 3.6.3 优化策略

训练采用 AdamW 优化器，默认参数为：

- 学习率：\(1\times10^{-3}\)
- 权重衰减：\(1\times10^{-4}\)
- 训练轮数：50
- Early Stopping patience：8

每轮训练后在验证集上计算 F1，并以最佳验证 F1 对应的模型作为最终模型。

---

## 3.7 实验变体与消融设计

### 3.7.1 Baseline 设计

为了验证各模态和融合方式的有效性，本文实现了以下实验变体：

1. `Attr-only`
2. `Text-only`
3. `Static-Graph-only`
4. `Attr + Text (Concat)`
5. `Attr + Text + Static Graph (Concat)`
6. `Dynamic Graph only`
7. `Dynamic Graph + Concat Fusion`
8. `Dynamic Graph + Adaptive Fusion`

这些 baseline 覆盖了“单模态”“静态图”“动态图”“固定融合”“自适应融合”等多个对比维度。

### 3.7.2 消融实验设计

为分析各关键组件的作用，本文设置如下消融实验：

#### 1. 去掉时间模块

将动态图序列退化为最终时刻静态图，只保留最后一个 snapshot，不再进行跨时间注意力聚合。

#### 2. 去掉自适应融合

用简单的 `Concat + MLP` 替代 Adaptive Gating，观察固定融合与动态融合的差异。

#### 3. 去掉质量特征

保留 gating 结构，但不输入质量特征 \(\mathbf{q}_i\)，用于验证“模态可靠性感知”是否有效。

---

## 3.8 结果解释性设计

除了分类性能之外，本文还特别关注模型的可解释性。当前实现支持以下分析输出：

### 3.8.1 模态权重统计

统计：

- 全样本平均模态权重；
- human 平均模态权重；
- bot 平均模态权重。

该分析可以帮助观察不同类别更依赖哪种信息源。

### 3.8.2 按邻居规模分组的模态权重

将用户按度数分为若干区间，例如：

- 0~5
- 6~20
- 21+

分析动态图模态在稀疏图节点与稠密图节点上的作用差异。

### 3.8.3 时间权重热力图

导出部分样本在所有快照上的时间权重矩阵，从而观察：

- 哪些时间快照最重要；
- bot 与 human 是否呈现出不同的时间关注模式。

### 3.8.4 错误样本分析

对测试集中被误分类的样本，输出：

- 真实标签
- 预测标签
- 预测概率
- 模态权重
- 时间权重

该分析有助于识别模型在何种类型用户上更容易失败。

---

## 3.9 本章小结

本章提出了一种面向 TwiBot-20 的 Dynamic-Aware Adaptive Fusion 检测框架。该方法从三方面展开设计：

1. 在数据层面，统一构建用户属性、文本和动态图快照三类输入；
2. 在模型层面，分别建立属性分支、文本分支和动态图分支，并利用时间注意力提取跨快照结构表示；
3. 在融合层面，引入质量特征驱动的自适应 gating 机制，实现对不同模态贡献的动态分配。

与传统静态图或简单拼接方法相比，本文方法能够同时建模用户是谁、用户说了什么，以及用户何时进入并如何嵌入社交网络，从而为后续实验提供更完整的判别基础。
