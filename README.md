# 基于 TwiBot-20 的模态可靠性感知动态融合方法

这个分支只保留一个方法，不再做多方法横向比较。

方法名称：

`Modality Reliability-Aware Dynamic Fusion`

核心思想：

不是把账号属性、文本、图结构三种信息固定拼接，而是让模型先判断当前账号“更应该信谁”，再做融合。

对应的直观理解是：

- 文本丰富的账号，更依赖 text branch；
- 图结构清晰的账号，更依赖 graph branch；
- 属性异常明显的账号，更依赖 feature branch。

## 代码入口

在仓库根目录运行：

```bash
python -m code prepare
python -m code train
python -m code visualize
python -m code report
python -m code run-all
```

## 当前分支保留的内容

- `code/data.py`
  负责把 TwiBot-20 原始文件整理成用户级样本。

- `code/experiments.py`
  只训练这一套动态融合方法。

- `code/graph_models.py`
  实现关系感知图传播、质量感知门控和动态融合。

- `code/visualization.py`
  只生成这一套方法的图，包括性能图和模态权重图。

- `code/reporting.py`
  只输出这套方法的报告。

- `README_METHOD.md`
  详细解释方法逻辑、输入、门控机制和论文表述方式。

## 方法概览

模型由三部分组成：

1. `Feature branch`
   编码账号的结构化属性，例如粉丝数、关注数、简介长度、默认头像、认证状态等。

2. `Text branch`
   编码简介和采样 tweet 的语义表示。

3. `Graph branch`
   在 `follow` / `friend` 关系图上进行关系感知传播，提取网络上下文表示。

之后，模型引入一组模态质量指标：

- `quality_text_richness`
- `quality_text_completeness`
- `quality_graph_connectivity`
- `quality_graph_reciprocity`
- `quality_feature_completeness`
- `quality_profile_anomaly`
- `quality_account_maturity`

这些指标不直接用于最终分类，而是进入一个 gating network，输出三路模态权重：

`w_feature, w_text, w_graph`

再通过 lightweight attention fusion 完成最终融合。

## 这个分支的创新点

可以用一句话概括：

`从固定融合，变成样本级自适应融合。`

相比“直接拼接三个分支”，这套方法的优势是：

- 更符合社交账号信息质量不一致的现实；
- 更容易解释为什么某些样本更依赖某一模态；
- 更适合在 TwiBot-20 上稳定落地。

## 训练后会生成什么

主要输出包括：

- `artifacts/models/modality_reliability_adaptive_fusion.pt`
  训练好的方法参数。

- `artifacts/tables/experiment_metrics.csv`
  验证集和测试集指标。

- `artifacts/tables/modality_reliability_adaptive_fusion_gate_diagnostics.csv`
  每个账号的预测概率、主导模态、三路权重和质量指标。

- `artifacts/figures/method_performance.png`
  单方法性能图。

- `artifacts/figures/modality_gate_profile.png`
  验证集/测试集平均模态权重图。

- `artifacts/figures/modality_reliability_profiles.png`
  不同账号条件下的模态偏好图。

## 环境建议

- Python：`3.10`
- 依赖安装：

```bash
pip install -r requirements.txt
```

如果缺少 `torch-geometric`，图模型部分将无法运行。

## 建议阅读顺序

1. 先看当前 `README.md`，了解这个分支只保留哪一个方法。
2. 再看 [README_METHOD.md](D:/作业/社会计算/Social-Computing-Project/README_METHOD.md)，理解方法逻辑和创新点。
3. 最后看 `artifacts/report.md` 和导出的图表，检查模型学到了怎样的模态偏好。
