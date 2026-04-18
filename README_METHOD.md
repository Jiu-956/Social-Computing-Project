# 模态可靠性感知动态融合方法

## 1. 方法定位

这个分支不再比较多种方法，而是只保留一套新的方法：

`Modality Reliability-Aware Dynamic Fusion`

中文可以表述为：

“提出一种面向社交机器人检测的模态可靠性感知动态融合方法，根据用户文本完整度、图结构稀疏度和属性异常程度，动态分配 feature / text / graph 三个分支的贡献权重。”

它的核心不是“再堆一个更大的模型”，而是解决一个更具体也更可解释的问题：

- 同样是账号 A 和账号 B，为什么不能用完全相同的模态融合方式？
- 为什么有的账号应该更相信文本，有的账号应该更相信图结构？
- 为什么固定拼接会把强信号和弱信号混在一起，削弱最终判别能力？

这个方法的答案是：

不是对所有样本采用同一套融合权重，而是让模型先判断“当前这个账号哪一种模态更可靠”，再进行融合。

## 2. 整体思路

方法分成三层：

### 2.1 三个基础分支

1. `Feature branch`
   读取账号的结构化属性，例如粉丝数、关注数、账号年龄、简介长度、默认头像、认证状态等。

2. `Text branch`
   读取账号简介和采样 tweet 的语义表示。

3. `Graph branch`
   在 `follow` 和 `friend` 两类关系图上做关系感知传播，建模节点在社交网络中的上下文位置。

### 2.2 可靠性评估层

模型不会直接把三个分支拼起来，而是先读取一组“模态质量指标”：

- 文本是否丰富：`quality_text_richness`
- 文本是否完整：`quality_text_completeness`
- 图连接是否充分：`quality_graph_connectivity`
- 关系模式是否稳定：`quality_graph_reciprocity`
- 账号属性是否完整：`quality_feature_completeness`
- 账号属性是否异常：`quality_profile_anomaly`
- 账号是否足够成熟：`quality_account_maturity`

这些指标不直接用于最终分类，而是作为“门控依据”，告诉模型：

- 这条样本的文本值不值得信；
- 这条样本的图结构值不值得信；
- 这条样本的属性信号是不是特别强。

### 2.3 动态融合层

基于上面的质量指标，模型通过一个 gating network 输出三路权重：

`w_feature, w_text, w_graph`

并满足：

`softmax([w_feature, w_text, w_graph])`

也就是说，每个账号都会有自己的一组融合权重。

然后模型先做一轮加权融合，再接一层轻量 attention fusion，让三路表示还能彼此交互，而不是只做机械加权。

## 3. 为什么这是一种创新

传统多模态融合最常见的问题，是对所有样本都采用同一种融合规则：

- 直接拼接；
- 固定加权；
- 统一 attention；
- 统一图传播后输出。

但在社交机器人检测里，不同账号的信息质量本身就不一样：

- 有些账号 tweet 很多、简介也很长，文本分支更可靠；
- 有些账号文本很少，但网络关系很清晰，图分支更可靠；
- 有些账号属性极不自然，例如粉丝关注比异常、发文频率异常，这时 feature 分支更可靠。

因此，这个方法的创新点不是“多了一个模态”，而是：

`把固定融合，改成样本级自适应融合。`

## 4. 代码结构

核心实现集中在以下文件：

- `code/graph_models.py`
  只保留这一套动态融合模型。

- `code/experiments.py`
  只负责准备文本稠密表示，并训练这一套方法。

- `code/visualization.py`
  只画与该方法有关的图：性能图、分支权重图、可靠性画像图。

- `code/reporting.py`
  只生成这套方法的报告，不再做多方法比较。

## 5. 核心模块解释

## 5.1 Local modality encoders

在 `code/graph_models.py` 中，模型先分别编码三类信息：

- `description_encoder`
- `tweet_encoder`
- `numeric_encoder`
- `categorical_encoder`

随后构成两个局部表示：

- `text_repr = f(description, tweet)`
- `feature_repr = f(numeric, categorical)`

其中：

- `text_repr` 表示当前账号本地可见的语义信息；
- `feature_repr` 表示当前账号本地可见的结构化画像信息。

## 5.2 Graph branch

图分支不是直接使用原始图特征，而是先构造：

`graph_seed = f(feature_repr, text_repr)`

然后在关系图上执行两层 `RGCNConv`：

- `follow`
- `friend`

这样得到的 `graph_repr` 不只是“这个账号自己是什么样”，而是“这个账号在社交图里处在什么关系上下文中”。

## 5.3 Quality-aware gate

`ReliabilityAwareFusion` 模块是这套方法的关键。

它包含两部分：

1. `learned gate`
   通过一个小型 MLP 从质量指标中学习 gate logits。

2. `quality prior`
   把方法设计者的先验注入 gate：
   - 文本丰富时，给 text 一个正偏置；
   - 图连接强时，给 graph 一个正偏置；
   - 属性异常明显时，给 feature 一个正偏置。

最终 gate logits 为：

`gate_logits = learned_gate(quality) + prior_scale * quality_prior(quality)`

再通过 `softmax` 变成权重。

这样做的好处是：

- 纯学习式 gate 更灵活；
- 先验偏置让模型更容易朝着我们想表达的“可靠性逻辑”收敛；
- 最后输出的权重更容易解释。

## 5.4 Lightweight attention fusion

在得到三路权重以后，模型不是停在简单加权求和，而是继续做一层轻量 attention：

- 先做 weighted sum，得到一个“主融合表示”；
- 再把这个表示作为 query，对三路模态表示做一次 attention；
- 最终输出 residual + layer norm 后的融合向量。

这一步的作用是：

- 保留 gate 的样本级权重解释性；
- 同时避免“完全独立加权”带来的信息割裂。

## 6. 训练目标

训练时使用：

- 类别不平衡感知的交叉熵损失；
- `AdamW` 优化器；
- 基于验证集 F1 的 early stopping。

因此，这个方法并不是只追求一个 logit 输出，而是围绕机器人检测这个不平衡分类任务来设计训练过程。

## 7. 模型输出

训练完成后会输出：

- `artifacts/models/modality_reliability_adaptive_fusion.pt`
  最终模型参数。

- `artifacts/tables/experiment_metrics.csv`
  验证集和测试集的指标。

- `artifacts/tables/modality_reliability_adaptive_fusion_gate_diagnostics.csv`
  每个账号的：
  - 预测结果；
  - 三路权重；
  - 主导模态；
  - 模态质量指标。

这个表非常重要，因为它可以直接支撑你的方法解释：

- 哪些样本更依赖文本？
- 哪些样本更依赖图？
- 哪些样本更依赖属性？

## 8. 适合论文或答辩的表述

可以直接这样写：

“为解决社交机器人检测中不同模态信息质量不一致的问题，本文提出一种模态可靠性感知动态融合方法。该方法首先分别学习用户属性表示、文本语义表示与关系图表示；随后基于文本完整度、图结构连通性和属性异常程度等质量指标，通过门控网络为不同样本动态分配模态权重，并结合轻量注意力机制完成最终融合。相比固定拼接式融合，该方法能够根据账号条件自动选择更可信的信息来源，从而提升模型的适应性与可解释性。”

## 9. 一句话总结

这套方法最重要的思想可以浓缩成一句话：

`不是所有账号都应该用同一种融合方式，而是先判断哪种模态更可靠，再做融合。`
