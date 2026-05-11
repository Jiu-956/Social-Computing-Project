# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 项目概述

基于 TwiBot-20 数据集的社交机器人检测研究项目，围绕三个研究问题组织：
1. 哪类信息（属性/文本/图结构）更有效？
2. 不同方法差异在哪里？
3. 结果是否可解释？

## 常用命令

```bash
# 完整流水线（需要网络下载 Transformer 模型，可设置 HF_ENDPOINT 镜像）
$env:HF_ENDPOINT="https://hf-mirror.com"; python -m code run-all

# 分步执行
python -m code prepare   # 数据准备
python -m code train     # 运行实验
python -m code visualize # 生成图表
python -m code report    # 生成报告

# 快速测试（跳过 GNN 和 Transformer，限制图节点数）
python -m code run-all --max-graph-users 50000 --disable-transformer --skip-gnn

# 仅用缓存结果重新生成图表和报告
python -m code visualize && python -m code report

# 跳过 node2vec
python -m code run-all --skip-node2vec
```

## 数据集与路径

- 原始数据位置：`data/raw/`（包含 `edge.csv`, `label.csv`, `split.csv`, `node.json`）
- `--data-dir data` 时自动识别 `data/raw`；也可以显式指定 `--data-dir data/raw`
- `node.json` 流式解析，不一次性读入内存（参见 `code/data.py` 中的 `stream_json_array`）

## 架构概览

```
code/
├── __main__.py   # CLI 入口（python -m code <command>）
├── cli.py        # argparse 命令行解析
├── config.py     # ProjectConfig 配置类（所有超参数的默认值）
├── data.py       # TwiBot-20 数据加载（流式 JSON，PreparedDataset）
├── baselines/    # sklearn 基线实验
│   ├── run.py         # run_experiments() 调度，生成 experiment_metrics.csv
│   ├── specs.py       # ExperimentSpec dataclass + _make_specs() 构建 9 个基线规格
│   └── embeddings/
│       ├── dense_text.py   # TF-IDF + SVD 稠密文本嵌入
│       ├── node2vec.py
│       └── transformer.py
├── gnn/          # GNN 模型（torch-geometric）
│   ├── run.py         # run_graph_neural_models() 编排，调用 builders
│   ├── train.py       # _train_gnn_model() 训练循环 + GNNResult
│   ├── builders/
│   │   ├── edge_index.py            # 无向图边索引（双向）
│   │   ├── relation_graph.py       # follow/friend 关系图（edge_type 0/1）
│   │   ├── relation_age_graph.py    # TIGN 用：18 种边类型 × 年龄桶
│   │   ├── snapshot_bundle.py       # BotDGT 动态快照（单调累积结构）
│   │   ├── position_encoding.py     # 快照位置编码（聚类密度/双向链接比/保留比例）
│   │   └── graph_structural_layer.py # 图结构特征层
│   └── models/        # 6 个 GNN 模型（均继承 _FeatureTextGraphBase）
├── interpretation.py  # 信息源增益分析
├── visualization.py  # 三个研究问题的图表生成
└── reporting.py      # 生成 artifacts/report.md
```

### 关键设计

**数据流**：raw files → `prepare_dataset()` → 用户级 DataFrame（含 `combined_text`、数值属性、类别属性、图统计特征、`account_age_bucket`）

**PreparedDataset manifest**：保存 DataFrame 列名映射（feature_numeric_columns、feature_categorical_columns、graph_structural_columns、combined_text_column 等），供 run.py 和 gnn/run.py 查找列

**BotDGT 快照构建**：按账号年龄分位数从稀到密生成边，强制单调累积（后一快照包含前一快照的边）。位置编码含聚类密度/双向链接比/保留比例

**TIGN 边类型**：(follow/friend) × (src_age_bucket 0..K-1) × (tgt_age_bucket 0..K-1) = 2×K² 种，通过 MLP 动态生成边嵌入

### 方法族映射

| Family | 来源 |
|---|---|
| `feature_only` | feature |
| `text_only` | text |
| `graph_only` | graph |
| `feature_text` | feature + text |
| `feature_graph` | feature + graph |
| `feature_text_graph` | feature + text + graph |

## 方法体系

### 非 GNN（sklearn）

| 方法 | 特点 |
|---|---|
| `*_logistic_regression` | TF-IDF / Node2Vec 拼接特征 |
| `*_random_forest` | 非线性融合，可输出特征重要性 |

### GNN 模型（6 个，均继承 `_FeatureTextGraphBase`）

| 模型 | 核心机制 |
|---|---|
| `feature_text_graph_gcn` | 基础图卷积，邻居均匀聚合 |
| `feature_text_graph_gat` | 注意力机制分配邻居权重 |
| `feature_text_graph_botrgcn` | 关系感知卷积（follow/friend 区分边类型） |
| `feature_text_graph_botsai` | 模态融合：不变表示 + 特定表示，通道自注意力；TransformerConv + 不变性约束 |
| `feature_text_graph_botdgt` | 多快照（默认 8 个单调累积）+ TransformerConv + 时间维聚合（attention/GRU/LSTM）+ 时序平滑/一致性约束 |
| `feature_text_graph_tign` | BotSAI + 年龄感知关系嵌入。边类型 = 2 × K²（K=--tign-num-age-buckets，默认 3），MLP 生成动态边嵌入，残差连接 |

> BotSAI / BotDGT / TIGN 依赖 `TransformerConv`，不可用时自动跳过（warning）

## 三个研究问题

代码围绕三个问题组织，所有产出在 `artifacts/` 下：

| 问题 | 产出 |
|---|---|
| 哪类信息更有效 | `tables/family_summary.csv`、`tables/source_contribution_summary.csv`、`figures/information_effectiveness.png` |
| 方法差异在哪里 | `tables/source_contribution_details.csv`、`figures/method_differences.png` |
| 结果可解释吗 | `tables/*_signals.csv`、`figures/explainability_signals.png`、`*_signal_map.png` |

> `interpretation.py` 生成信息源增益和消融实验表；`visualization.py` 负责绑图；`reporting.py` 输出 `artifacts/report.md`

## 产出目录

```
artifacts/
├── cache/          # 预处理数据缓存（PreparedDataset）
├── models/         # 训练好的 GNN .pt 文件
├── tables/         # CSV 结果表
│   ├── experiment_metrics.csv     # 各方法性能指标
│   ├── experiment_predictions.csv # 预测结果
│   ├── family_summary.csv         # 方法族最佳表现
│   ├── source_contribution_summary.csv
│   ├── source_ablation.csv
│   └── *_signals.csv              # 特征/文本/图解释性信号
└── figures/        # 可视化图片
    ├── information_effectiveness.png
    ├── method_differences.png
    ├── explainability_signals.png
    └── *_signal_map.png
```

## 依赖与可选功能

- `torch-geometric`（可选）：缺失时 GNN 实验跳过，提示 WARNING 而非报错
- `sentence-transformers`（可选）：缺失时 Transformer 文本嵌入跳过
- `TransformerConv`（torch-geometric 版本相关）：缺失时 BotSAI / BotDGT / TIGN 跳过

## 配置参数

所有超参数集中在 `ProjectConfig`（`code/config.py`），可通过命令行覆盖：

- 数据：`--max-graph-users`、`--max-tweets-per-user`
- 文本：`--tfidf-max-features`、`--dense-text-svd-dim`、`--disable-transformer`
- 图：`--node2vec-dimensions`、`--skip-node2vec`
- GNN：`--gnn-hidden-dim`、`--gnn-epochs`、`--gnn-patience`、`--gnn-dropout`
- BotSAI：`--botsai-invariant-weight`、`--botsai-attention-heads`
- BotDGT：`--botdgt-snapshot-count`、`--botdgt-temporal-module`（attention/gru/lstm）、`--botdgt-temporal-smoothness-weight`
- TIGN：`--tign-num-age-buckets`（默认 3）、`--tign-intra-class-weight`（默认 0.02）
