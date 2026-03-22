# Social-Computing-Project

基于 `TwiBot-20` 的“网络中心视角协同异常行为检测”实验框架。代码已经按你的选题文档组织成一套可复现实验流程，覆盖：

- `TwiBot-20` 大文件流式解析
- 有标签用户 + support 支持节点的关系图构建
- 用户属性特征、采样 tweet 文本特征、显式图结构特征
- DeepWalk 节点嵌入
- 监督分类基线对比
- 基于候选 bot 的异常群体发现
- t-SNE / 网络子图可视化

## 目录结构

```text
twibot20_project/
  __main__.py
  cli.py
  config.py
  data.py
  experiments.py
  features.py
  visualization.py
```

## 数据说明

当前这份 `Twibot-20` 数据的结构与标准论文设定一致，但需要注意：

- `label.csv` 只有 `11,826` 个有标签用户
- `split.csv` 中还有 `217,754` 个 `support` 节点，可作为图结构上下文
- `edge.csv` 至少包含 `post / friend / follow` 三类边
- `node.json` 同时包含用户节点和 tweet 节点

本项目默认：

- 分类和评估只针对有标签用户
- 图结构保留 `support` 节点作为上下文
- tweet 文本只对有标签用户做采样聚合，默认每个用户最多取 `8` 条推文，便于在普通机器上运行

## 运行方式

在项目根目录执行：

```bash
python -m twibot20_project prepare
python -m twibot20_project train
python -m twibot20_project cluster
python -m twibot20_project visualize
```

也可以一键跑完整流程：

```bash
python -m twibot20_project run-all
```

## 常用参数

```bash
python -m twibot20_project run-all ^
  --data-dir Twibot-20 ^
  --output-dir artifacts ^
  --max-tweets-per-user 8 ^
  --deepwalk-dimensions 64 ^
  --threshold 0.8
```

如果你想先快速验证代码，可以限制有标签用户数量：

```bash
python -m twibot20_project run-all --max-labeled-users 500
```

## 输出结果

运行后会生成：

- `artifacts/cache/users.csv`
- `artifacts/cache/graph_edges.csv`
- `artifacts/cache/deepwalk_embeddings.joblib`
- `artifacts/cache/network_summary.json`
- `artifacts/tables/classification_metrics.csv`
- `artifacts/tables/classification_predictions.csv`
- `artifacts/tables/cluster_assignments.csv`
- `artifacts/tables/cluster_summary.csv`
- `artifacts/figures/degree_distribution.png`
- `artifacts/figures/embedding_tsne.png`
- `artifacts/figures/top_suspicious_cluster.png`

## 模型与实验设计

当前内置了 3 组实验：

1. `profile_text_logreg`
仅使用账号属性 + 文本 TF-IDF 的逻辑回归基线。

2. `graph_profile_rf`
使用账号属性 + 显式图特征 + DeepWalk 嵌入的随机森林基线。

3. `full_logreg`
融合账号属性、文本、图结构指标和 DeepWalk 嵌入的完整基线。

聚类阶段支持：

- `DBSCAN`
- `Spectral Clustering`

## 和选题文档的对应关系

这套实现已经对应到文档里的核心目标：

- 图构建：`friend / follow` 形成用户关系图
- 多模态节点特征：profile + sampled tweets
- 结构特征：度、PageRank、聚类系数、k-core、近似介数、近似调和中心性、互惠性
- 图表示学习：DeepWalk
- 协同异常识别：基于分类结果筛选候选节点，再做聚类和簇摘要
- 可视化：嵌入空间图、局部异常簇网络图、度分布

## 后续可扩展

如果你后面还要继续完善课程项目，可以在这套框架上继续加：

- Node2Vec / BotRGCN / GCN
- 更强的文本编码器
- 更细的社区发现与模块度分析
- 报告自动生成脚本
