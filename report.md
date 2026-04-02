# 报告导航

根目录这份文档不再重复维护一份容易过期的“静态实验报告”，而是作为整个项目的阅读入口，避免 `README`、方法说明和实验报告再次出现上下逻辑对不上的情况。

## 推荐阅读顺序

1. `README.md`
   先看项目现在围绕哪三个研究问题组织。
2. `README_METHOD.md`
   再看六类方法、解释性分析和可视化模块是怎么衔接起来的。
3. `artifacts/report.md`
   最后看当前缓存结果自动生成的实验报告。

## 当前主线

项目现在统一回答三个问题，而不是只做“谁分数最高”的排行榜：

1. 哪类信息更有效？
2. 不同方法差异在哪里？
3. 结果是否可解释？

## 对应产物

### 问题一：哪类信息更有效

- 表格：`artifacts/tables/source_contribution_summary.csv`
- 表格：`artifacts/tables/source_ablation.csv`
- 图片：`artifacts/figures/information_effectiveness.png`

### 问题二：不同方法差异在哪里

- 表格：`artifacts/tables/source_contribution_details.csv`
- 图片：`artifacts/figures/method_differences.png`

### 问题三：结果是否可解释

- 表格：`artifacts/tables/feature_signals.csv`
- 表格：`artifacts/tables/text_signals.csv`
- 表格：`artifacts/tables/graph_signals.csv`
- 图片：`artifacts/figures/explainability_signals.png`
- 图片：`artifacts/figures/feature_signal_map.png`
- 图片：`artifacts/figures/embedding_separation_map.png`
- 图片：`artifacts/figures/local_network_patterns.png`

## 如何刷新

如果已经有缓存和训练结果，只需要重新生成图表与报告：

```bash
python -m code visualize
python -m code report
```

如果要从头完整跑一遍：

```bash
python -m code run-all
```

## 说明

- `artifacts/report.md` 才是当前实验结果的最新自动化报告。
- 根目录这份 `report.md` 只负责给出统一入口与阅读路径，避免再次和代码实现脱节。
