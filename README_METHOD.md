# 当前方法实现说明

这份文档说明当前项目中的方法是如何在代码里落地的。整体上，这不是一个端到端的深度学习模型，而是一条“数据解析 -> 图与文本特征构造 -> 监督分类 -> 可疑群体发现 -> 可视化”的实验流水线。

## 1. 总体流程

当前主入口在 `code/cli.py`，完整流程由下面四步组成：

1. `prepare`：解析数据、抽取用户表、构建缓存
2. `train`：训练分类基线模型
3. `cluster`：对高风险账号做群体发现
4. `visualize`：输出统计图和可疑群体子图

执行 `python -m code run-all` 时，代码会按这个顺序依次调用：

- `prepare_dataset`
- `enrich_with_graph_features`
- `run_classification_experiments`
- `run_group_detection`
- `generate_visualizations`

## 2. 数据准备是怎么做的

实现文件：`code/data.py`

### 2.1 输入数据

默认从 `data/` 目录读取：

- `label.csv`
- `split.csv`
- `edge.csv`
- `node.json`

其中：

- `label.csv` 提供账号标签，`human=0`，`bot=1`
- `split.csv` 提供训练、验证、测试划分
- `edge.csv` 提供 `follow / friend / post` 等关系
- `node.json` 提供用户画像和 tweet 节点内容

### 2.2 标注用户和图节点

代码先读取 `label.csv` 与 `split.csv`，得到有标签用户表。

如果设置了 `--max-labeled-users`，就会在 `split × label` 分组内按比例采样，目的是做快速实验时仍尽量保持数据分布稳定。

同时，代码会从 `split.csv` 中得到图上的用户集合。这样做的含义是：

- 分类任务只对有标签用户评估
- 图结构可以保留更多上下文节点

### 2.3 边信息与 tweet 采样

`collect_edge_information()` 会遍历 `edge.csv`，完成三件事：

1. 抽取用户之间的 `follow/friend` 边，形成图结构
2. 统计每个标注用户的入边、出边和发帖数量
3. 从 `post` 边中为每个用户最多采样 `max_tweets_per_user` 条 tweet

这里有一个比较关键的实现选择：

- 图边只保留用户到用户的关系
- tweet 节点不直接进入图建模，而是被转成文本特征

### 2.4 解析 node.json

`node.json` 很大，所以项目没有一次性整体加载，而是用 `stream_json_array()` 流式解析。

之后：

- 用户节点用来提取画像特征
- tweet 节点用来回填之前采样到的文本内容

画像特征包括：

- `followers_count`
- `following_count`
- `listed_count`
- `statuses_count`
- `account_age_days`
- `description_length`
- `username_length`
- `name_length`
- `has_location`
- `has_url`
- `is_verified`
- `is_protected`
- `followers_following_ratio`

文本部分则把用户简介和采样 tweet 合并为一个字段：

- `combined_text = description_text + tweet_text`

### 2.5 prepare 阶段输出

这一阶段会把中间结果写入 `result/cache/`：

- `users.csv`
- `graph_edges.csv`
- `graph_nodes.csv`
- `manifest.json`
- `network_summary.json`

## 3. 图特征和 DeepWalk 是怎么做的

实现文件：`code/features.py`

### 3.1 构图方式

项目把 `graph_edges.csv` 构造成一个有向图 `nx.DiGraph()`。

如果同一对节点之间有重复边，权重会累加。之后：

- 图统计特征主要基于 `graph`
- DeepWalk 训练使用 `graph.to_undirected()`

### 3.2 图统计特征

`compute_graph_features()` 为每个目标用户计算以下图特征：

- 入度 `graph_in_degree`
- 出度 `graph_out_degree`
- 总度数 `graph_total_degree`
- PageRank `graph_pagerank`
- 聚类系数 `graph_clustering`
- k-core 编号 `graph_core_number`
- 连通分量大小 `graph_component_size`
- 近似介数中心性 `graph_betweenness_approx`
- 近似 harmonic centrality `graph_harmonic_approx`
- reciprocity `graph_reciprocity`

这里有两个近似项：

- 介数中心性使用 `k` 采样，避免全图精确计算太慢
- harmonic centrality 也只对部分源点采样

这是为了让实验能在普通机器上运行，而不是追求最重的图算法配置。

### 3.3 DeepWalk 实现

项目没有单独依赖现成 DeepWalk 包，而是在 `features.py` 里自己实现了一个轻量版本：

1. 用 `RandomWalkCorpus` 在无向图上做随机游走
2. 把随机游走序列交给 `gensim.Word2Vec`
3. 得到每个用户的嵌入向量 `dw_0 ... dw_n`

核心超参数包括：

- `deepwalk_dimensions`
- `deepwalk_walk_length`
- `deepwalk_num_walks`
- `deepwalk_window`
- `deepwalk_epochs`

输出写入：

- `result/cache/deepwalk_embeddings.joblib`

## 4. 分类实验是怎么做的

实现文件：`code/experiments.py`

### 4.1 特征矩阵

分类阶段会读取缓存后的用户表和 DeepWalk 嵌入，并构造两类特征：

- 数值特征：画像特征 + 图特征 + DeepWalk 向量
- 文本特征：`combined_text` 的 TF-IDF

数值特征先做 `StandardScaler` 标准化。

文本特征使用：

- `TfidfVectorizer`
- `ngram_range=(1, 2)`
- `stop_words="english"`

如果某个实验需要文本，就把数值特征和 TF-IDF 稀疏矩阵拼接起来。

### 4.2 当前内置的 3 组实验

项目里目前固定了三组基线：

1. `profile_text_logreg`
   只用画像数值特征 + 文本 TF-IDF，模型是逻辑回归
2. `graph_profile_rf`
   用画像 + 图特征 + DeepWalk，模型是随机森林
3. `full_logreg`
   用画像 + 图特征 + DeepWalk + 文本 TF-IDF，模型是逻辑回归

其中：

- 逻辑回归使用 `solver="saga"` 与 `class_weight="balanced"`
- 随机森林使用 `class_weight="balanced_subsample"`

### 4.3 训练与模型选择

训练严格按 `train / val / test` 划分进行。

每个实验都会输出：

- 验证集指标
- 测试集指标
- 全部样本预测结果

评价指标包括：

- Accuracy
- Precision
- Recall
- F1
- AUC-ROC

最终模型不是按测试集选，而是按验证集 `F1` 最高来确定最佳实验。这一点比较重要，因为它避免了直接拿测试集做模型选择。

最佳模型和实验信息写入：

- `result/models/best_classifier.joblib`
- `result/models/best_experiment.json`

表格结果写入：

- `result/tables/classification_metrics.csv`
- `result/tables/classification_predictions.csv`

## 5. 可疑群体发现是怎么做的

实现文件：`code/experiments.py` 中的 `run_group_detection()`

这一步不是直接对全体节点做聚类，而是先做候选筛选，再对候选 bot 聚类。

### 5.1 候选节点筛选

默认做法是：

- 读取最佳分类实验的预测概率
- 在指定 split 中选取 `bot_probability >= threshold` 的用户

如果阈值太高导致候选节点少于 10 个，代码会退化成：

- 选取概率最高的前 `min(200, len(split_df))` 个用户

也可以通过 `--use-ground-truth` 直接用真实 bot 标签做聚类，这通常更适合分析方法上限，而不是模拟真实场景。

### 5.2 聚类前处理

候选用户会使用下面的 dense 特征：

- 画像数值特征
- 图统计特征
- DeepWalk 嵌入

之后执行：

1. `StandardScaler`
2. `PCA` 降到最多 16 维
3. 聚类

### 5.3 聚类方法

当前支持两种方法：

- `DBSCAN`
- `SpectralClustering`

默认是 `DBSCAN`，因为它更适合“从高风险候选中找局部紧密群体”，而不是强行给所有样本分簇。

### 5.4 聚类结果汇总

对于每个非噪声簇，代码会统计：

- 簇大小
- bot_ratio
- human_ratio
- 子图密度
- 平均 bot 概率

同时输出整体指标：

- `purity`
- `nmi`
- `noise_ratio`
- `cluster_count`

结果写入：

- `result/tables/cluster_assignments.csv`
- `result/tables/cluster_summary.csv`
- `result/tables/cluster_metrics.json`

## 6. 可视化是怎么做的

实现文件：`code/visualization.py`

当前会生成三类图：

### 6.1 度分布图

`degree_distribution.png`

比较 bot 与 human 在 `graph_total_degree` 上的分布差异。

### 6.2 DeepWalk 嵌入 t-SNE 图

`embedding_tsne.png`

流程是：

1. 先对 DeepWalk 向量标准化
2. 先用 PCA 压缩到最多 32 维
3. 再做 t-SNE 到二维

这张图主要用来观察 bot/human 在表示空间中的可分性。

### 6.3 最可疑簇的子图

`top_suspicious_cluster.png`

代码会从聚类结果里选出“bot_ratio 和 size 综合最靠前”的簇，并绘制其内部连接结构。

如果节点太多，只保留度数最高的前 80 个节点，避免图完全看不清。

## 7. 当前方法的特点

从实现角度看，这套方法的核心思想是：

- 用传统机器学习而不是复杂神经网络做可复现实验基线
- 把文本、画像、图统计和图嵌入合到同一条流水线里
- 先做账号级二分类，再做高风险候选群体发现

它的优点是：

- 结构清晰
- 训练成本可控
- 中间产物完整，便于论文写作和实验分析

它的局限也很明确：

- 图表示仍是手工图特征 + DeepWalk，不是端到端图神经网络
- 群体发现依赖前一步分类质量
- DBSCAN 和阈值选择会显著影响最终聚类结果

## 8. 一句话概括当前实现

当前项目实现的是一条面向社会计算实验的可复现 baseline 流程：先把 `TwiBot-20` 解析成用户级表征，再融合画像、文本、图结构和 DeepWalk 嵌入做 bot 分类，最后对高风险账号做聚类和可视化分析。
