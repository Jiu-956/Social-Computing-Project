# Dynamic-Aware Adaptive Fusion Bot Detector

一个面向 **TwiBot-20** 的社交机器人检测项目实现，核心目标是将 **属性信息、文本信息、动态图结构信息** 统一建模，并通过 **自适应融合机制** 完成 bot / human 二分类。

项目当前已经完成：

- TwiBot-20 原始数据预处理
- 基于 `created_at` 的累积式动态图快照构建
- 属性分支、文本分支、动态图分支建模
- Temporal Self-Attention 时间聚合
- Adaptive Gating 融合
- baseline 与 ablation 实验
- 模态权重、时间权重与错误样本分析导出

## 项目特点

- `data/` 与 `code/` 分离，结构清晰，适合课程项目与论文复现
- 原始数据、处理中间结果、实验输出分别管理
- 提供单步脚本，也提供一键执行脚本 `run_all.py`
- 文本编码支持 `sentence-transformers`，不可用时自动回退到 `TF-IDF + SVD`
- 动态图不是按 tweet 时间，而是按用户 `created_at` 构造，符合本项目方法设定

## 项目结构

```text
Social-Computing-Project/
├─ README.md
├─ method.md
├─ requirements.txt
├─ data/
│  ├─ raw/
│  │  ├─ node.json
│  │  ├─ edge.csv
│  │  ├─ label.csv
│  │  └─ split.csv
│  └─ processed/
│     ├─ user_table.csv
│     ├─ attr_features.pt
│     ├─ text_features.pt
│     ├─ text_inputs.pkl
│     ├─ quality_features.pt
│     ├─ labels.pt
│     ├─ train_idx.pt
│     ├─ val_idx.pt
│     ├─ test_idx.pt
│     └─ dynamic_graph/
│        ├─ meta.pkl
│        └─ snapshots/
├─ code/
│  ├─ preprocess_twibot20.py
│  ├─ configs/
│  │  └─ default.yaml
│  ├─ scripts/
│  │  ├─ run_preprocess.py
│  │  ├─ run_train.py
│  │  ├─ run_eval.py
│  │  ├─ run_ablation.py
│  │  ├─ run_visualize.py
│  │  └─ run_all.py
│  └─ src/
│     └─ dafbot/
│        ├─ data/
│        ├─ models/
│        ├─ train/
│        ├─ config.py
│        ├─ pipeline.py
│        └─ utils.py
└─ outputs/
   ├─ checkpoints/
   ├─ logs/
   ├─ figures/
   └─ tables/
```

## 方法概览

本项目方法是一个三分支统一框架：

1. 属性分支：编码用户 profile 与统计特征
2. 文本分支：编码用户简介与 tweet 文本
3. 动态图分支：基于用户 `created_at` 构造多时间快照图，并提取时序结构表示

随后使用质量感知的自适应融合模块，为每个用户动态分配三种模态权重，最后通过分类头输出 bot / human 预测结果。

如果你需要看更正式的论文写法，请参考 [method.md](/D:/作业/社会计算/Social-Computing-Project/method.md)。

## 动态图构建方式

本项目的动态图不是按 tweet 发表时间构造，而是按 **用户账户创建时间 `created_at`** 构造累积式快照图。

对于时间点 `tau_t`，定义：

- 节点集合：`V^(t) = {u_i | created_at(u_i) < tau_t}`
- 边集合：`E^(t) = {(u, v) in E | u in V^(t) and v in V^(t)}`

这意味着：

- 某个用户只有在其创建时间之后才会出现在图中
- 后期 snapshot 会累积包含早期 snapshot 中的节点和边
- 最后一个 snapshot 可以视为完整静态图

每个 snapshot 都会保存：

- `edge_index`
- `edge_type`
- `exist_nodes`
- `global_index`
- `clustering_coefficient`
- `bidirectional_links_ratio`

对应实现：

- [code/preprocess_twibot20.py](/D:/作业/社会计算/Social-Computing-Project/code/preprocess_twibot20.py)
- [build_dynamic_snapshots.py](/D:/作业/社会计算/Social-Computing-Project/code/src/dafbot/data/build_dynamic_snapshots.py)

## 数据说明

原始数据默认放在 `data/raw/`：

- `node.json`：节点信息，包含用户节点和 tweet 节点
- `edge.csv`：边信息，包含 `follow`、`friend`、`post`
- `label.csv`：用户标签
- `split.csv`：训练/验证/测试划分

处理规则如下：

- `follow` 和 `friend` 用于构造用户关系图
- `post` 只用于找回 tweet 文本，不进入图结构建模
- `data/processed/` 保存预处理后的中间结果与张量文件

## 环境要求

建议使用：

- Python 3.10 或 3.11

安装依赖：

```bash
pip install -r requirements.txt
```

核心依赖包括：

- `torch`
- `sentence-transformers`
- `scikit-learn`
- `networkx`
- `matplotlib`
- `PyYAML`

说明：

- 如果 `sentence-transformers` 不能正常使用，代码会自动回退到 `TF-IDF + SVD`
- 如果你使用 GPU，可将 [default.yaml](/D:/作业/社会计算/Social-Computing-Project/code/configs/default.yaml) 中的 `device` 改为 `cuda`

## 快速开始

### 1. 仅做预处理

```bash
python code/preprocess_twibot20.py
```

或：

```bash
python code/scripts/run_preprocess.py
```

预处理完成后会在 `data/processed/` 中生成：

- 用户表 `user_table.csv`
- 属性特征 `attr_features.pt`
- 文本特征 `text_features.pt`
- 文本输入 `text_inputs.pkl`
- 质量特征 `quality_features.pt`
- 标签和数据划分索引
- 动态图元信息和 snapshot 文件

### 2. 训练完整模型

```bash
python code/scripts/run_train.py --experiment dynamic_graph_adaptive
```

默认 checkpoint 输出到：

- `outputs/checkpoints/dynamic_graph_adaptive.pt`

### 3. 评估模型并导出分析结果

```bash
python code/scripts/run_eval.py --checkpoint outputs/checkpoints/dynamic_graph_adaptive.pt
```

### 4. 跑 baseline 与 ablation

```bash
python code/scripts/run_ablation.py
```

### 5. 一键执行完整流程

```bash
python code/scripts/run_all.py
```

默认执行：

1. 预处理
2. 训练 `dynamic_graph_adaptive`
3. 评估训练得到的 checkpoint

如果希望连同 baseline / ablation 一起跑：

```bash
python code/scripts/run_all.py --with-ablation
```

如果只想跳过预处理：

```bash
python code/scripts/run_all.py --skip-preprocess
```

## `run_all.py` 参数说明

`run_all.py` 主要参数如下：

- `--config`：配置文件路径，默认 `code/configs/default.yaml`
- `--experiment`：主实验名称，默认 `dynamic_graph_adaptive`
- `--skip-preprocess`：跳过预处理
- `--skip-train`：跳过训练，直接使用 checkpoint 做评估
- `--skip-eval`：跳过评估
- `--checkpoint`：手动指定 checkpoint 路径
- `--with-ablation`：额外跑 baseline 与 ablation

## 支持的实验变体

当前训练器支持以下实验：

- `attr_only`
- `text_only`
- `static_graph_only`
- `attr_text_concat`
- `attr_text_static_graph_concat`
- `dynamic_graph_only`
- `dynamic_graph_concat`
- `dynamic_graph_adaptive`
- `ablation_no_temporal`
- `ablation_no_adaptive_fusion`
- `ablation_no_quality`

主实现位于：

- [trainer.py](/D:/作业/社会计算/Social-Computing-Project/code/src/dafbot/train/trainer.py)

## 输出说明

### `data/processed/`

保存预处理结果：

- 用户级标准表
- 属性/文本/质量特征
- 标签与划分索引
- 动态图 snapshot

### `outputs/checkpoints/`

保存训练好的模型参数：

- `*.pt`

### `outputs/logs/`

保存训练历史与中间日志：

- `*_history.json`

### `outputs/tables/`

保存实验结果表：

- `*_metrics.csv`
- `ablation_summary.csv`

### `outputs/<experiment_name>/`

在评估阶段额外生成：

- `metrics.csv`
- `test_predictions.csv`
- `val_predictions.csv`
- `modal_weight_statistics.csv`
- `temporal_weight_heatmap.csv`
- `misclassified_samples.csv`

## 关键配置项

配置文件位于：

- [code/configs/default.yaml](/D:/作业/社会计算/Social-Computing-Project/code/configs/default.yaml)

常用配置项：

- `paths.data_root`：原始数据目录
- `paths.processed_dir`：预处理结果目录
- `paths.output_dir`：实验输出目录
- `preprocess.max_tweets_per_user`：每个用户最多保留的 tweet 数
- `text.encoder_type`：文本编码方式
- `graph.granularity`：动态图时间粒度
- `graph.max_snapshots`：训练时最多使用的快照数
- `model.d_model`：统一隐层维度
- `training.epochs`：训练轮数
- `training.learning_rate`：学习率
- `training.patience`：early stopping patience

## 复现建议

建议按以下顺序进行实验：

1. 先完成预处理
2. 优先跑 `dynamic_graph_adaptive`
3. 再跑 `dynamic_graph_concat`
4. 再跑三个主要消融
5. 最后整理 `outputs/tables/` 和 `outputs/figures/`

如果是第一次跑项目，推荐直接：

```bash
python code/scripts/run_all.py
```

## 相关文件

- 方法章节说明：[method.md](/D:/作业/社会计算/Social-Computing-Project/method.md)
- 预处理入口：[code/preprocess_twibot20.py](/D:/作业/社会计算/Social-Computing-Project/code/preprocess_twibot20.py)
- 一键运行脚本：[run_all.py](/D:/作业/社会计算/Social-Computing-Project/code/scripts/run_all.py)

## 备注

当前版本优先保证：

- 方法流程完整
- 代码结构清晰
- 实验可复现
- baseline 与 full model 可公平比较

如果后续继续扩展，可以在此基础上增加：

- 更强的文本编码器
- 更复杂的图神经网络
- 更细粒度的时间采样策略
- 更完整的论文实验结果整理脚本
