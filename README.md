# Social-Computing-Project

当前项目采用标准实验目录结构：

- `code/`：代码
- `data/`：数据
- `result/`：实验结果

方法实现与研究叙事说明见 `README_METHOD.md`。

## 当前支持的方法

项目现在同时保留了原有机器学习 baseline，并扩展了更强的方法族：

- 基于特征的方法：`profile_text_logreg`、`graph_profile_rf`、`full_logreg`
- 基于图嵌入的方法：DeepWalk、Node2Vec
- 基于更强文本编码的方法：Transformer text encoder
- 基于图神经网络的方法：GCN、BotRGCN
- 基于结构分析的方法：聚类、社区发现、模块度分析
- 自动化结果整理：Markdown 报告自动生成

## 主要命令

```bash
python -m code prepare
python -m code train
python -m code cluster
python -m code community
python -m code visualize
python -m code report
python -m code run-all
```

## 推荐完整流程

```bash
python -m code run-all ^
  --data-dir data ^
  --output-dir result ^
  --transformer-model-name sentence-transformers/all-MiniLM-L6-v2 ^
  --deepwalk-dimensions 64 ^
  --node2vec-dimensions 64 ^
  --threshold 0.8
```

## 关键说明

- `prepare`：解析 `data/` 中的数据并构造缓存
- `train`：运行 baseline、Node2Vec、Transformer、GCN、BotRGCN 等实验
- `cluster`：对高风险账号做聚类发现
- `community`：对可疑账号子图做更细的社区发现与模块度分析
- `visualize`：输出结构图、嵌入图和模块度图
- `report`：自动生成 `result/report.md`

## 主要输出

表格输出：

- `result/tables/classification_metrics.csv`
- `result/tables/classification_predictions.csv`
- `result/tables/classification_comparison.csv`
- `result/tables/cluster_assignments.csv`
- `result/tables/cluster_summary.csv`
- `result/tables/cluster_metrics.json`
- `result/tables/community_assignments.csv`
- `result/tables/community_summary.csv`
- `result/tables/community_metrics.json`

模型与缓存：

- `result/cache/users.csv`
- `result/cache/graph_edges.csv`
- `result/cache/deepwalk_embeddings.joblib`
- `result/cache/node2vec_embeddings.joblib`
- `result/cache/graph_embeddings.joblib`
- `result/cache/text_embeddings_*.joblib`
- `result/models/best_experiment.json`
- `result/models/best_model.joblib`

图像与报告：

- `result/figures/degree_distribution.png`
- `result/figures/embedding_tsne.png`
- `result/figures/top_suspicious_cluster.png`
- `result/figures/community_modularity.png`
- `result/report.md`

## 依赖

```bash
pip install -r requirements.txt
```

如果你第一次运行 Transformer 文本编码实验，本地需要能访问或已经缓存 HuggingFace 模型。
