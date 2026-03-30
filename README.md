# 基于 TwiBot-20 的社交机器人检测研究

这个仓库已经按照新的选题要求重构为一个围绕 `TwiBot-20` 的统一实验框架，并且把方法严格组织成论文里常见的六类：

- 基于特征
- 基于文本
- 基于图
- 基于特征和文本
- 基于特征和图
- 基于特征、文本和图

主代码目录现在是 `code/`，可以直接通过下面的命令运行：

```bash
python -m code prepare
python -m code train
python -m code visualize
python -m code report
python -m code run-all
```

## 目录结构

```text
code/                  主要实验代码
data/                  TwiBot-20 原始数据
tests/                 smoke test
artifacts/             运行后自动生成的缓存、模型、表格和图像
README_METHOD.md       方法文档
```

## 环境配置

### 1. 推荐环境

- 操作系统：Windows / Linux / macOS 均可
- Python：推荐 `3.10`
- 环境管理：推荐 `conda`
- 运行方式：在仓库根目录执行 `python -m code ...`

### 2. 创建 conda 环境

```bash
conda create -n social_comp_project python=3.10 -y
conda activate social_comp_project
```

### 3. 安装依赖

先在仓库根目录执行：

```bash
pip install -r requirements.txt
```

当前依赖见 [requirements.txt](D:/作业/社会计算/Social-Computing-Project/requirements.txt)，主要包括：

- `numpy`
- `pandas`
- `scikit-learn`
- `gensim`
- `torch`
- `torch-geometric`
- `sentence-transformers`

### 4. 可选依赖说明

#### `torch-geometric`

`GCN`、`GAT`、`BotRGCN` 这三类图神经网络方法依赖 `torch-geometric`。

- 如果已经安装成功，就可以运行完整六类方法。
- 如果没有安装成功，代码现在会自动跳过这三类 GNN，不会导致整个流程报错。

只想先跑通主流程时，可以直接这样执行：

```bash
python -m code --skip-gnn run-all
```

#### `sentence-transformers`

Transformer 文本方法依赖 `sentence-transformers` 及其底层 `transformers`。

- 如果网络正常、依赖兼容，可以运行 Transformer 文本实验。
- 如果不想联网下载模型，或者环境里这部分依赖有冲突，可以先禁用。

禁用 Transformer 的运行方式：

```bash
python -m code --disable-transformer run-all
```

### 5. Hugging Face 镜像

如果直接访问 Hugging Face 较慢，可以先设置镜像：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
python -m code run-all
```

### 6. 检查环境是否安装成功

可以先运行下面的命令做最小验证：

```bash
python -m unittest tests.test_smoke -v
python -m code --help
```

如果这两条命令都能正常执行，说明代码结构和基础依赖已经就绪。

## 当前方法设计

### 1. 基于特征

- `feature_only_logistic_regression`
- `feature_only_random_forest`

输入只包含用户画像和账号属性，不使用文本语义和图结构。

### 2. 基于文本

- `text_only_tfidf_logistic_regression`
- `text_only_transformer_logistic_regression`

输入只包含简介和 tweet 文本表示。

### 3. 基于图

- `graph_only_structure_random_forest`
- `graph_only_node2vec_logistic_regression`

输入只来自图结构统计或图嵌入。

### 4. 基于特征和文本

- `feature_text_tfidf_logistic_regression`
- `feature_text_transformer_logistic_regression`

### 5. 基于特征和图

- `feature_graph_random_forest`
- `feature_graph_node2vec_logistic_regression`

### 6. 基于特征、文本和图

- `feature_text_graph_tfidf_node2vec_logistic_regression`
- `feature_text_graph_gcn`
- `feature_text_graph_gat`
- `feature_text_graph_botrgcn`

其中后三个图神经网络模型参考了 TwiBot-22 官方仓库在 `TwiBot-20` 上的输入组织方式：`description` 文本表示、`tweet` 文本表示、数值属性、类别属性和关系边共同进入图模型。

## 为什么图方法这次更接近论文

旧版实现里，图模型只看了简化后的数值特征和关系矩阵，所以表现偏弱。现在主要做了两点改进：

1. 图神经网络改成更接近官方仓库的输入结构：描述文本、tweet 文本、数值属性、类别属性一起进入模型。
2. 加入 `GAT` 和 `BotRGCN` 风格模型，而不是只有简化版 `GCN`。

这样“基于特征、文本和图”的方法更接近论文里的 `FTG` 方法。

## 运行建议

### 1. 快速跑通

```bash
python -m code run-all --max-graph-users 50000 --disable-transformer --skip-gnn
```

### 2. 运行完整实验

```bash
python -m code run-all --disable-transformer
```

### 3. 启用镜像并运行完整实验

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
python -m code run-all
```

## 主要输出

- `artifacts/cache/users.csv`
- `artifacts/cache/graph_edges.csv`
- `artifacts/cache/node2vec_embeddings.csv`
- `artifacts/tables/experiment_metrics.csv`
- `artifacts/tables/experiment_predictions.csv`
- `artifacts/tables/family_summary.csv`
- `artifacts/figures/model_comparison.png`
- `artifacts/figures/node2vec_projection.png`
- `artifacts/figures/feature_importance.png`
- `artifacts/report.md`

## 文档

详细方法说明见 [README_METHOD.md](D:/作业/社会计算/Social-Computing-Project/README_METHOD.md)。
