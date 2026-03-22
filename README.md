# Social-Computing-Project

基于 `TwiBot-20` 的社会计算实验项目，目录命名已统一为标准实验结构：

- `code/`：代码
- `data/`：数据
- `result/`：实验结果

## 目录结构

```text
code/
  __main__.py
  cli.py
  config.py
  data.py
  experiments.py
  features.py
  visualization.py
data/
result/
```

说明：

- 主运行入口是 `code`

## 数据说明

默认数据目录为 `data/`，其中应包含 `TwiBot-20` 数据集的核心文件，例如：

- `label.csv`
- `split.csv`
- `edge.csv`
- `node.json`

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行方式

推荐使用新的标准命名入口：

```bash
python -m code prepare
python -m code train
python -m code cluster
python -m code visualize
```

一键执行完整流程：

```bash
python -m code run-all
```

如需显式指定目录：

```bash
python -m code run-all ^
  --data-dir data ^
  --output-dir result ^
  --max-tweets-per-user 8 ^
  --deepwalk-dimensions 64 ^
  --threshold 0.8
```

快速烟雾测试：

```bash
python -m code run-all --max-labeled-users 500
```

## 输出结果

默认结果写入 `result/`，常见产物包括：

- `result/cache/users.csv`
- `result/cache/graph_edges.csv`
- `result/cache/deepwalk_embeddings.joblib`
- `result/cache/network_summary.json`
- `result/tables/classification_metrics.csv`
- `result/tables/classification_predictions.csv`
- `result/tables/cluster_assignments.csv`
- `result/tables/cluster_summary.csv`
- `result/tables/cluster_metrics.json`
- `result/figures/degree_distribution.png`
- `result/figures/embedding_tsne.png`
- `result/figures/top_suspicious_cluster.png`
