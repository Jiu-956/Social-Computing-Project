# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 TwiBot-20 数据集的社交机器人检测研究项目，围绕三个研究问题组织：
1. 哪类信息（属性/文本/图结构）更有效？
2. 不同方法差异在哪里？
3. 结果是否可解释？

## 常用命令

```bash
# 完整流水线
python -m code run-all

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

# 使用 Hugging Face 镜像
$env:HF_ENDPOINT="https://hf-mirror.com"; python -m code run-all
```

## 数据集与路径

- 原始数据位置：`data/raw/`（包含 `edge.csv`, `label.csv`, `split.csv`, `node.json`）
- `--data-dir data` 时自动识别 `data/raw`；也可以显式指定 `--data-dir data/raw`
- `node.json` 流式解析，不一次性读入内存（参见 `code/data.py` 中的 `stream_json_array`）

## 架构概览

```
code/
├── __main__.py   # CLI 入口，解析命令行参数
├── config.py     # ProjectConfig 配置类，所有超参数汇总
├── data.py       # TwiBot-20 读取与用户级样本构建（流式 JSON）
├── experiments.py # 16 种方法的实验调度（非 GNN 部分）
├── graph_models.py # GNN 模型定义与训练（GCN/GAT/BotRGCN/BotSAI/BotDGT）
├── interpretation.py # 信息源增益分析、消融实验、解释性信号表
├── visualization.py  # 围绕三个研究问题的图表生成
└── reporting.py   # 生成 artifacts/report.md
```

### 关键设计

**数据流**：raw files → `prepare_dataset()` → 用户级 DataFrame（包含 `combined_text`、数值属性、类别属性、图统计特征）

**实验规格**：`experiments.py` 中 `ExperimentSpec` dataclass 描述每种方法所需的特征列、文本模式等信息

**方法族映射**（`experiments.py` 中 `FAMILY_SOURCES`）：

| Family | 来源 |
|---|---|
| `feature_only` | feature |
| `text_only` | text |
| `graph_only` | graph |
| `feature_text` | feature + text |
| `feature_graph` | feature + graph |
| `feature_text_graph` | feature + text + graph |

**GNN 模型输入**：description 嵌入 + tweet 嵌入 + 数值属性 + 类别属性 + 边索引，各自经独立 MLP 投影后拼接，再经图卷积层传播

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

**BotDGT 快照构建**：按账号年龄分位数从稀到密生成边，强制单调累积（后一快照包含前一快照的边）

## 产出目录

```
artifacts/
├── cache/          # 预处理数据缓存（PreparedDataset）
├── models/         # 训练好的 GNN .pt 文件
├── tables/         # CSV 结果表
│   ├── source_contribution_summary.csv  # 信息源平均增益
│   ├── source_contribution_details.csv  # 各方法间差异来源
│   ├── source_ablation.csv              # 融合模型消融
│   └── *_signals.csv                    # 特征/文本/图解释性信号
└── figures/        # 可视化图片
    ├── information_effectiveness.png     # Q1：哪类信息更有效
    ├── method_differences.png            # Q2：方法差异分析
    ├── explainability_signals.png        # Q3：可解释性概览
    └── *_signal_map.png 等              # 细粒度信号图
```

## 依赖与可选功能

- `torch-geometric`（可选）：缺失时 GNN 实验跳过，提示 WARNING 而非报错
- `sentence-transformers`（可选）：缺失时 Transformer 文本嵌入跳过
- `TransformerConv`（torch-geometric 版本相关）：缺失时 BotSAI / BotDGT 跳过

## 配置参数

所有超参数集中在 `ProjectConfig`（`code/config.py`），可通过命令行覆盖：

- 数据：`--max-graph-users`、`--max-tweets-per-user`
- 文本：`--tfidf-max-features`、`--dense-text-svd-dim`、`--disable-transformer`
- 图：`--node2vec-dimensions`、`--skip-node2vec`
- GNN：`--gnn-hidden-dim`、`--gnn-epochs`、`--gnn-patience`、`--gnn-dropout`
- BotSAI：`--botsai-invariant-weight`、`--botsai-attention-heads`
- BotDGT：`--botdgt-snapshot-count`、`--botdgt-temporal-module`（attention/gru/lstm）、`--botdgt-temporal-smoothness-weight`
