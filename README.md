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

## Conda 环境

推荐按下面的方式手动创建环境：

```bash
conda create -n social_comp_project python=3.10 -y
conda activate social_comp_project
pip install -r requirements.txt
$env:HF_ENDPOINT="https://hf-mirror.com"
```


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

python -m code --data-dir data --output-dir result --transformer-model-name sentence-transformers/all-MiniLM-L6-v2 --logreg-max-iter 4000 --deepwalk-dimensions 64 --node2vec-dimensions 64 run-all --threshold 0.8

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

如果你第一次运行 Transformer 文本实验：

- 代码会强制下载并使用 `--transformer-model-name` 指定的 HuggingFace 模型；如果下载失败，训练会直接报错停止。
- 如果网络到 Hugging Face 不稳定，可以先配置代理，或在 PowerShell 里运行 `$env:HF_ENDPOINT="https://hf-mirror.com"` 后再启动实验。
- 下载成功后，模型会缓存到本机的 HuggingFace 缓存目录，后续运行会复用缓存。
