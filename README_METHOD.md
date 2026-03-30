# 方法文档

## 1. 设计目标

本项目围绕 `TwiBot-20` 机器人检测任务，按照论文和官方仓库中的思路，把方法划分为六类：

- 基于特征（F）
- 基于文本（T）
- 基于图（G）
- 基于特征和文本（FT）
- 基于特征和图（FG）
- 基于特征、文本和图（FTG）

这样做的目的不是只找一个分数最高的模型，而是系统回答：

1. 用户画像信息是否足够识别 bot？
2. 文本语义是否比传统特征更有效？
3. 图结构能否补充单账号信息？
4. 当特征、文本和图一起使用时，是否能进一步提升？

## 2. 数据处理思路

### 2.1 输入文件

项目直接使用 `TwiBot-20` 原始文件：

- `data/label.csv`
- `data/split.csv`
- `data/edge.csv`
- `data/node.json`

### 2.2 流式解析

`node.json` 体积很大，因此代码没有一次性全部读入，而是用流式方式逐条解析 JSON 数组。

对应实现：

- `code/data.py`

### 2.3 用户级样本

最终整理出的用户级样本包括：

- 用户 id
- train / val / test / support 划分
- bot / human 标签
- 简介文本 `description_text`
- tweet 合并文本 `tweet_text`
- 综合文本 `combined_text`
- 用户属性特征
- 图统计特征

### 2.4 为什么保留 support 节点

`support` 节点没有分类标签，但它们是图结构的一部分。对于 GCN、GAT、BotRGCN 这类图神经网络来说，support 节点能提供邻居上下文，因此不能简单删除。

## 3. 特征构建

### 3.1 基于特征的方法所用特征

这类方法只使用用户画像和账号属性，不使用文本编码和图卷积。

主要特征包括：

- 粉丝数 `followers_count`
- 关注数 `following_count`
- 发文数 `tweet_count`
- 列表数 `listed_count`
- 账号年龄 `account_age_days`
- 用户名长度 `username_length`
- 显示名长度 `display_name_length`
- 简介长度 `description_length`
- 是否有 location
- 是否有 URL
- 是否认证 `is_verified`
- 是否保护账号 `is_protected`
- 默认头像特征 `default_profile_image`
- 粉丝关注比
- 日均发文
- 对数变换后的计数特征

### 3.2 图结构统计特征

这类特征来自 `edge.csv`：

- `follow_in_count`
- `follow_out_count`
- `friend_in_count`
- `friend_out_count`
- `total_in_degree`
- `total_out_degree`
- `in_out_ratio`
- `friend_share_out`
- `follow_share_out`
- `post_count`
- `sampled_tweet_count`

这些特征可以单独作为“基于图”方法的一部分，也可以与用户属性一起构成“基于特征和图”方法。

## 4. 六类方法详细说明

## 4.1 基于特征（F）

### 方法 1：`feature_only_logistic_regression`

思路：

- 把每个用户看成一个独立样本
- 只使用用户属性特征
- 用逻辑回归完成 bot / human 二分类

优点：

- 可解释性强
- 训练快
- 适合作为基线

局限：

- 不利用文本语义
- 不利用图结构

### 方法 2：`feature_only_random_forest`

思路：

- 使用同样的用户属性特征
- 用随机森林替代逻辑回归

优点：

- 能处理非线性关系
- 能输出特征重要性

局限：

- 仍然不利用文本和图

## 4.2 基于文本（T）

### 方法 1：`text_only_tfidf_logistic_regression`

思路：

- 把简介和 tweet 合并成文本
- 使用 `TF-IDF` 提取稀疏文本特征
- 用逻辑回归分类

适合回答的问题：

- 单看文本内容，能否识别机器人？

### 方法 2：`text_only_transformer_logistic_regression`

思路：

- 用 `SentenceTransformer` 生成稠密文本表示
- 仅使用文本嵌入进行分类

与 TF-IDF 的区别：

- TF-IDF 更偏关键词
- Transformer 更偏语义相似性

## 4.3 基于图（G）

### 方法 1：`graph_only_structure_random_forest`

思路：

- 只使用图结构统计特征
- 不使用用户属性和文本

它回答的问题是：

- 仅从网络连接形态，能否区分 bot 和 human？

### 方法 2：`graph_only_node2vec_logistic_regression`

思路：

- 用 `Node2Vec` 对用户关系图做随机游走
- 使用 `Word2Vec` 训练节点嵌入
- 把节点嵌入送入逻辑回归

意义：

- 不再只看度数，而是学习节点在整个图中的结构角色

## 4.4 基于特征和文本（FT）

### 方法 1：`feature_text_tfidf_logistic_regression`

思路：

- 将用户属性特征和 TF-IDF 文本特征拼接
- 再用逻辑回归分类

这是最直接的多模态融合方式之一。

### 方法 2：`feature_text_transformer_logistic_regression`

思路：

- 将用户属性特征和 Transformer 文本嵌入拼接
- 用逻辑回归分类

适合检验：

- 文本语义与结构化属性是否互补

## 4.5 基于特征和图（FG）

### 方法 1：`feature_graph_random_forest`

思路：

- 将用户属性特征与图统计特征一起输入随机森林

### 方法 2：`feature_graph_node2vec_logistic_regression`

思路：

- 将用户属性特征、图统计特征、Node2Vec 图嵌入拼接
- 用逻辑回归分类

意义：

- 同时利用用户画像和节点在网络中的位置表示

## 4.6 基于特征、文本和图（FTG）

这是当前最重要的一组方法，也最接近论文和官方仓库。

### 方法 1：`feature_text_graph_tfidf_node2vec_logistic_regression`

思路：

- 将用户属性、TF-IDF 文本特征、图嵌入一起拼接
- 用逻辑回归分类

这是一个“非神经网络”的 FTG 融合基线。

### 方法 2：`feature_text_graph_gcn`

思路：

- 输入四块信息：
  - 描述文本稠密表示
  - tweet 文本稠密表示
  - 数值属性
  - 类别属性
- 先分别线性映射，再拼接
- 用两层 `GCNConv` 在图上传播
- 最后输出二分类结果

这对应官方仓库中 GCN 的思路。

### 方法 3：`feature_text_graph_gat`

思路：

- 输入结构与 GCN 相同
- 图卷积层替换为 `GATConv`

与 GCN 的区别：

- GAT 不把邻居一视同仁
- 它会学习不同邻居的重要性

### 方法 4：`feature_text_graph_botrgcn`

思路：

- 输入结构与 GCN/GAT 相同
- 图层替换为 `RGCNConv`
- 区分不同关系类型，例如 `follow` 和 `friend`

这是本项目最接近论文 `BotRGCN` 的实现。

它的核心假设是：

- 不同关系边的传播语义不同
- 因此消息传播时不能把所有边都当成一种连接

## 5. 为什么现在的图模型更接近论文

原始简化版图模型的问题是：

- 只喂了简化后的数值输入
- 没有把描述文本和 tweet 文本分别编码
- 关系感知做得不够充分

这次改进后，图模型更接近官方仓库：

1. 输入分成 description / tweet / num_prop / cat_prop 四块
2. 加入 `GCN`、`GAT` 和 `BotRGCN`
3. `BotRGCN` 使用关系类型感知的 `RGCNConv`

## 6. 与官方仓库的一致与差异

一致之处：

- 方法分组采用 F / T / G / FT / FG / FTG
- FTG 图模型使用描述文本、tweet 文本、数值属性、类别属性、图边
- 引入 GCN、GAT、BotRGCN 三种图神经网络

差异之处：

- 官方仓库在文本编码中大量使用 RoBERTa 特征
- 本项目为了兼顾本地可运行性，文本部分同时保留 TF-IDF 和轻量 Transformer
- 图模型是论文思路的课程项目级实现，而不是完全逐文件复现官方代码

## 7. 输出文件说明

- `artifacts/tables/experiment_metrics.csv`：所有模型的验证集/测试集指标
- `artifacts/tables/family_summary.csv`：六类方法各自最优模型
- `artifacts/tables/experiment_predictions.csv`：逐用户预测结果
- `artifacts/models/`：模型缓存
- `artifacts/figures/`：图表
- `artifacts/report.md`：自动汇总的实验报告
