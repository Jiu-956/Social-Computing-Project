# Multi-granularity BotDGTv1 实验文档

## 概述

Multi-granularity BotDGTv1 是对原始 BotDGT 的时序粒度增强，同时使用多个时间粒度（粗/中/细粒度）的 snapshot，通过 Gate Fusion 融合多粒度表示。

## 问题背景

原始 BotDGT 只使用单一时间间隔（如 year = 12个月）构建动态图快照：
- 时间粒度过粗可能丢失短期演化模式
- 时间粒度过细可能引入噪声

多粒度方法让模型同时学习：
- **粗粒度**（年）：长期行为模式
- **中粒度**（半年）：中期趋势
- **细粒度**（季度）：短期波动

## 方法说明

### 架构

```
                    Multi-Granularity Fusion
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
    year             six_months         three_months
        │                   │                   │
        ▼                   ▼                   ▼
  Structural + Temporal   Structural + Temporal   Structural + Temporal
        │                   │                   │
        ▼                   ▼                   ▼
      z_year            z_six_months        z_three_months
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                    GranularityFusion
                     (Gate/Mean/Concat)
                            │
                            ▼
                     fused_representation
                            │
                            ▼
                    Classifier → bot/human
```

### Fusion 方法

| 方法 | 说明 |
|------|------|
| **gate** | 学习权重 gates = softmax(MLP(concat([z_12, z_6, z_3]))) |
| **mean** | z = mean([z_12, z_6, z_3]) |
| **concat** | z = MLP(concat([z_12, z_6, z_3])) |

### 显存优化

- **共享编码器**：默认共享 structural encoder，减少参数量
- **顺序处理**：每个粒度顺序处理，最后融合，避免显存爆炸

## 实验命令

### 原始 BotDGT (baseline)

```bash
python -m code train --only-botdgt
```

### Multi-granularity BotDGTv1

```bash
# 三粒度 (year + six_months + three_months)
python -m code train --only-botdgt --use-multi-granularity \
    --granularities "year,six_months,three_months" \
    --granularity-fusion gate

# 两粒度 (year + three_months)
python -m code train --only-botdgt --use-multi-granularity \
    --granularities "year,three_months" \
    --granularity-fusion gate

# 使用 mean fusion
python -m code train --only-botdgt --use-multi-granularity \
    --granularities "year,six_months,three_months" \
    --granularity-fusion mean
```

### 消融实验

```bash
# 不同 fusion 方法对比
python -m code train --only-botdgt --use-multi-granularity \
    --granularities "year,six_months,three_months" \
    --granularity-fusion gate

python -m code train --only-botdgt --use-multi-granularity \
    --granularities "year,six_months,three_months" \
    --granularity-fusion mean

python -m code train --only-botdgt --use-multi-granularity \
    --granularities "year,six_months,three_months" \
    --granularity-fusion concat

# 不同粒度组合对比
python -m code train --only-botdgt --use-multi-granularity \
    --granularities "year,six_months" --granularity-fusion gate

python -m code train --only-botdgt --use-multi-granularity \
    --granularities "year,three_months" --granularity-fusion gate

python -m code train --only-botdgt --use-multi-granularity \
    --granularities "six_months,three_months" --granularity-fusion gate

# 独立编码器对比
python -m code train --only-botdgt --use-multi-granularity \
    --granularities "year,six_months,three_months" \
    --granularity-fusion gate \
    --no-share-structural-encoder \
    --no-share-temporal-encoder
```

## 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-multi-granularity` | flag | False | 启用多粒度模式 |
| `--granularities` | str | "year" | 逗号分隔的粒度列表 |
| `--granularity-fusion` | str | "gate" | 融合方式: gate/mean/concat |
| `--share-structural-encoder` | flag | True | 共享 structural 编码器 |
| `--share-temporal-encoder` | flag | True | 共享 temporal 编码器 |
| `--temporal-readout` | str | "last" | 时序表示读取方式: last/masked_mean |

## 代码结构

```
code/gnn/botdgt/
├── multi_granularity.py  # BotDGTv1 核心实现
│   ├── GranularityFusion - 融合模块
│   ├── MultiGranularityBotDyGNN - 多粒度模型
│   ├── MultiGranularityBotDGTTrainer - 训练器
│   └── run_botdgt_multi_granularity() - 运行函数
├── model.py - 原始 BotDGT 模型
├── train.py - 原始 BotDGT 训练
└── data.py - 数据加载
```

## 预期效果

Multi-granularity BotDGTv1 可能在以下场景获得提升：

1. **行为演化明显的机器人**：短期和长期行为模式不同
2. **时间跨度大的数据集**：需要融合多时间尺度的信息
3. **复杂行为模式**：单一粒度无法捕获所有信息

## 实验记录模板

```markdown
## 实验结果记录

### 实验配置
- 模型: Multi-granularity BotDGTv1
- 粒度: ___
- Fusion: ___
- Dataset: TwiBot-20

### 结果
- Val Accuracy: ___
- Val F1: ___
- Test Accuracy: ___
- Test F1: ___

### 对比 (vs Original BotDGT)
- Accuracy 提升: ___
- F1 提升: ___
```

## 注意事项

1. **默认行为不变**：不加 `--use-multi-granularity` 时，运行原始 BotDGT
2. **粒度必须存在于数据**：确保 `data/graph_data/graphs/` 下有对应间隔的 snapshot 文件
3. **显存**：多粒度会占用更多显存，使用共享编码器可缓解
