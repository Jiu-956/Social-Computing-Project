# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
├── __main__.py   # CLI 入口
├── cli.py         # 命令行参数解析
├── config.py      # ProjectConfig 配置类
├── data.py        # TwiBot-20 数据加载（流式 JSON）
├── baselines/     # sklearn 基线实验
│   ├── run.py         # run_experiments() 调度
│   ├── specs.py       # ExperimentSpec + 9 个基线规格
│   └── embeddings/    # TF-IDF / Node2Vec / Transformer
├── gnn/           # GNN 模型（torch-geometric）
│   ├── run.py         # run_graph_neural_models() 编排
│   ├── train.py       # 训练循环 + GNNResult
│   ├── builders/      # 边索引构建
│   │   ├── edge_index.py        # 无向图边索引
│   │   ├── relation_graph.py    # follow/friend 关系图
│   │   ├── relation_age_graph.py # 年龄感知关系图（TIGN 用）
│   │   └── snapshot_bundle.py   # BotDGT 动态快照
│   └── models/        # 6 个 GNN 模型
│       ├── base.py   # _FeatureTextGraphBase 基类
│       ├── gcn.py    # FeatureTextGraphGCN
│       ├── gat.py    # FeatureTextGraphGAT
│       ├── botrgcn.py # FeatureTextGraphBotRGCN
│       ├── botsai.py # FeatureTextGraphBotSAI
│       ├── botdgt.py # FeatureTextGraphBotDGT
│       └── tign.py   # FeatureTextGraphTIGN（创新模型）
├── interpretation.py # 信息源增益分析、消融实验、解释性信号
├── visualization.py  # 围绕三个研究问题的图表生成
└── reporting.py   # 生成 artifacts/report.md
```

### 关键设计

**数据流**：raw files → `prepare_dataset()` → 用户级 DataFrame（包含 `combined_text`、数值属性、类别属性、图统计特征、`account_age_bucket`）

**实验规格**：`baselines/specs.py` 中 `ExperimentSpec` dataclass 描述每种方法所需的特征列、文本模式等信息

**方法族映射**（`baselines/specs.py` 中 `_make_specs()`）：

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

### GNN（torch-geometric）

| 模型 | 核心机制 |
|---|---|
| `feature_text_graph_gcn` | 基础图卷积，邻居信息均匀聚合 |
| `feature_text_graph_gat` | 注意力机制，不同邻居分配不同权重 |
| `feature_text_graph_botrgcn` | 关系感知卷积（`follow` vs `friend` 区分边类型） |
| `feature_text_graph_botsai` | 模态融合：每种模态分解为不变表示+特定表示，通道自注意力融合；图传播用 TransformerConv + 关系嵌入；训练时加不变性约束 |
| `feature_text_graph_botdgt` | 基于账号年龄构建多个图快照（默认 8 个），快照内用 TransformerConv，位置编码含聚类密度/双向链接比/保留比例；时间维用 attention/GRU/LSTM 聚合，加时序平滑与跨步一致性约束 |
| `feature_text_graph_tign` | **创新模型**：BotSAI + 年龄感知关系嵌入。边类型 = (follow/friend) × (src_age_bucket 0-2) × (tgt_age_bucket 0-2) = 18 种，通过 MLP 生成动态边嵌入。图层含残差连接。可通过 `--tign-num-age-buckets` 和 `--tign-intra-class-weight` 配置 |

**BotDGT 快照构建**：按账号年龄分位数从稀到密生成边，强制单调累积（后一快照包含前一快照的边）

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