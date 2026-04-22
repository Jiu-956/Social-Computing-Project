# 基于 TwiBot-20 的社交机器人检测研究

这个仓库现在不再把目标定义为“只找一个分数最高的模型”，而是围绕三个研究问题组织代码、图表和报告：

1. 哪类信息更有效？
2. 不同方法差异在哪里？
3. 结果是否可解释？

对应地，代码结构也按“数据准备 -> 六类方法实验 -> 问题导向分析 -> 可视化/报告”来组织。

## 代码入口

在仓库根目录运行：

```bash
python -m code prepare
python -m code train
python -m code visualize
python -m code report
python -m code run-all
```

默认 `--data-dir` 为 `data`，程序会自动识别并读取 `data/raw`。你也可以显式指定 `--data-dir data/raw`。

## 仓库结构

```text
code/                  主要实验代码
data/                  TwiBot-20 原始数据
artifacts/             自动生成的缓存、模型、表格、图像与报告
README_METHOD.md       方法设计与分析逻辑文档
```

其中核心模块的职责如下：

- `code/data.py`：读取 `data/raw/label.csv`、`data/raw/split.csv`、`data/raw/edge.csv`、`data/raw/node.json`，生成用户级样本。
- `code/experiments.py`：统一训练 F / T / G / FT / FG / FTG 六类方法。
- `code/interpretation.py`：生成信息源增益、融合消融和解释性信号表。
- `code/visualization.py`：生成围绕三个研究问题的可视化图片。
- `code/reporting.py`：按“三个研究问题”输出自动报告。

## 当前分析主线

### 问题一：哪类信息更有效？

代码会输出：

- `artifacts/tables/family_summary.csv`
- `artifacts/tables/source_contribution_summary.csv`
- `artifacts/tables/source_ablation.csv`
- `artifacts/figures/information_effectiveness.png`

这部分回答：

- 单看用户属性、文本语义、图结构，谁更有效？
- 从单源到双源、三源融合，哪类信息的平均增益最大？
- 在融合模型里，去掉哪类信息损失最大？

### 问题二：不同方法差异在哪里？

代码会输出：

- `artifacts/tables/source_contribution_details.csv`
- `artifacts/figures/method_differences.png`

这部分不只比较“模型名不同”，而是分析：

- 增加一种信息后，方法族之间的差异来自哪里；
- 同样都使用特征、文本、图信息时，简单拼接、GCN、GAT、BotRGCN、BotSAI、BotDGT 等方法为什么会出现差距。

### 问题三：结果是否可解释？

代码会输出：

- `artifacts/tables/feature_signals.csv`
- `artifacts/tables/text_signals.csv`
- `artifacts/tables/graph_signals.csv`
- `artifacts/figures/explainability_signals.png`
- `artifacts/figures/feature_signal_map.png`
- `artifacts/figures/embedding_separation_map.png`
- `artifacts/figures/local_network_patterns.png`

这部分回答：

- 用户属性模型主要抓哪些账号画像信号；
- 文本模型主要抓哪些关键词/短语；
- 图模型主要抓哪些结构性关系模式。
- 文本嵌入和图嵌入在二维空间里是否出现机器人 / 人类分离。
- 高置信机器人 / 人类的局部关系网络有什么结构差异。

## 环境配置

### 1. 推荐环境

- 操作系统：Windows / Linux / macOS
- Python：推荐 `3.10`
- 环境管理：推荐 `conda`

### 2. 创建环境

```bash
conda create -n social_comp_project python=3.10 -y
conda activate social_comp_project
pip install -r requirements.txt
```

### 3. 可选依赖说明

#### `torch-geometric`

`GCN`、`GAT`、`BotRGCN`、`BotSAI`、`BotDGT` 依赖 `torch-geometric`。

- 安装成功：可以运行完整六类方法。
- 安装失败：可以先跳过 GNN。

```bash
python -m code --skip-gnn run-all
```

#### `sentence-transformers`

Transformer 文本方法依赖 `sentence-transformers`。

- 依赖齐全：可以运行文本语义嵌入实验。
- 不想联网下载模型：可以先禁用。

```bash
python -m code --disable-transformer run-all
```

### 4. Hugging Face 镜像

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
python -m code run-all
```

## 当前方法体系

- `feature_only_logistic_regression`
- `feature_only_random_forest`
- `text_only_tfidf_logistic_regression`
- `text_only_transformer_logistic_regression`
- `graph_only_structure_random_forest`
- `graph_only_node2vec_logistic_regression`
- `feature_text_tfidf_logistic_regression`
- `feature_text_transformer_logistic_regression`
- `feature_graph_random_forest`
- `feature_graph_node2vec_logistic_regression`
- `feature_text_graph_tfidf_node2vec_logistic_regression`
- `feature_text_graph_gcn`
- `feature_text_graph_gat`
- `feature_text_graph_botrgcn`
- `feature_text_graph_botsai`
- `feature_text_graph_botdgt`

这些方法不是孤立堆在一起，而是统一映射到 F / T / G / FT / FG / FTG 六类方法族中，方便做横向对比与解释。

## 推荐运行方式

### 快速跑通

```bash
python -m code run-all --max-graph-users 50000 --disable-transformer --skip-gnn
```

### 使用现有缓存重新生成图表和报告

```bash
python -m code visualize
python -m code report
```

### 运行完整实验

```bash
python -m code run-all --disable-transformer
```

## 建议阅读顺序

1. 先看当前 `README.md`，了解仓库现在围绕哪三个问题组织。
2. 再看 [README_METHOD.md](D:/作业/社会计算/Social-Computing-Project/README_METHOD.md)，了解六类方法和解释性分析如何衔接。
3. 最后看 `artifacts/report.md`，查看自动生成的结论与对应图表。
