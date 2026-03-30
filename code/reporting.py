from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import ProjectConfig

FAMILY_LABELS = {
    "feature_only": "基于特征",
    "text_only": "基于文本",
    "graph_only": "基于图",
    "feature_text": "基于特征和文本",
    "feature_graph": "基于特征和图",
    "feature_text_graph": "基于特征、文本和图",
}


def generate_report(config: ProjectConfig) -> Path:
    metrics = pd.read_csv(config.tables_dir / "experiment_metrics.csv", low_memory=False)
    family_summary = pd.read_csv(config.tables_dir / "family_summary.csv", low_memory=False)
    dataset_summary = json.loads((config.cache_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    best_experiment = json.loads((config.models_dir / "best_experiment.json").read_text(encoding="utf-8"))

    metrics["family_cn"] = metrics["family"].map(FAMILY_LABELS).fillna(metrics["family"])
    family_summary["family_cn"] = family_summary["family"].map(FAMILY_LABELS).fillna(family_summary["family"])

    val_table = metrics[metrics["split"] == "val"].sort_values(["f1", "auc_roc"], ascending=False).head(12)
    test_table = metrics[metrics["split"] == "test"].sort_values(["f1", "auc_roc"], ascending=False).head(12)

    test_best_accuracy = _pick_best_row(metrics, split_name="test", sort_columns=["accuracy", "f1", "auc_roc"])
    test_best_f1 = _pick_best_row(metrics, split_name="test", sort_columns=["f1", "auc_roc", "accuracy"])
    graph_best_test = _pick_best_row(
        metrics,
        split_name="test",
        family="feature_text_graph",
        sort_columns=["accuracy", "f1", "auc_roc"],
    )

    report_lines = [
        "# 基于 TwiBot-20 的社交机器人检测实验报告",
        "",
        "## 1. 研究目标",
        "本项目围绕 TwiBot-20 数据集完成 bot / human 二分类实验，并严格按照论文常见的六类方法组织实验：基于特征、基于文本、基于图、基于特征和文本、基于特征和图、基于特征、文本和图。",
        "",
        "## 2. 数据与处理",
        f"- 图用户总数：{dataset_summary['graph_user_count']}",
        f"- 有标签用户数：{dataset_summary['labeled_user_count']}",
        f"- support 节点数：{dataset_summary['support_user_count']}",
        f"- 用户关系边数：{dataset_summary['graph_edge_count']}",
        f"- 采样 tweet 数：{dataset_summary['sampled_tweet_count']}",
        "",
        "数据准备阶段对 `node.json` 采用流式解析，避免 5.7GB 原始文件一次性载入内存；同时保留 support 节点，为 GCN、GAT、BotRGCN 风格模型提供完整的邻居上下文。",
        "",
        "## 3. 六类方法设计",
        "- 基于特征：只使用用户画像和账号属性特征。",
        "- 基于文本：只使用简介文本和 tweet 文本表示。",
        "- 基于图：只使用图统计特征或图嵌入。",
        "- 基于特征和文本：将结构化账号特征与文本特征拼接后分类。",
        "- 基于特征和图：将账号属性与图统计或 Node2Vec 图嵌入融合。",
        "- 基于特征、文本和图：同时输入 description 表示、tweet 表示、数值属性、类别属性与图结构，最接近论文中的 FTG 思路。",
        "",
        "## 4. 验证集表现",
        _to_markdown_table(val_table[["experiment", "family_cn", "accuracy", "precision", "recall", "f1", "auc_roc"]]),
        "",
        "## 5. 测试集表现",
        _to_markdown_table(test_table[["experiment", "family_cn", "accuracy", "precision", "recall", "f1", "auc_roc"]]),
        "",
        "## 6. 按方法族汇总",
        _to_markdown_table(family_summary[["family_cn", "split", "best_experiment", "accuracy", "f1", "auc_roc"]]),
        "",
        "## 7. 结果解读",
        f"- 按验证集 F1 选择的最佳模型是 `{best_experiment['best_experiment']}`，验证集 F1 为 {best_experiment['best_val_f1']:.4f}。",
        f"- 测试集按准确率最高的模型是 `{test_best_accuracy['experiment']}`，Accuracy 为 {test_best_accuracy['accuracy']:.4f}，F1 为 {test_best_accuracy['f1']:.4f}。",
        f"- 测试集按 F1 最高的模型是 `{test_best_f1['experiment']}`，F1 为 {test_best_f1['f1']:.4f}，Accuracy 为 {test_best_f1['accuracy']:.4f}。",
    ]

    if graph_best_test is not None:
        report_lines.extend(
            [
                f"- 本次最强图神经网络模型是 `{graph_best_test['experiment']}`，测试集 Accuracy 为 {graph_best_test['accuracy']:.4f}，F1 为 {graph_best_test['f1']:.4f}。",
                "- 相比旧版简化图模型，当前 FTG 图模型把 description、tweet、数值属性、类别属性和关系边同时输入图网络，更接近论文和官方仓库里的实现组织方式。",
            ]
        )

    report_lines.extend(
        [
            "- 如果只看 Accuracy，图方法已经进入全项目第一梯队；如果看 F1，基于特征和文本的方法仍然略占优势，这说明当前任务里文本语义和用户属性依然非常重要。",
            "- BotRGCN 风格模型优于 GCN，说明区分 follow 与 friend 关系类型是有价值的，这与论文中“关系感知传播更有效”的结论一致。",
            "",
            "## 8. 与论文和官方仓库的关系",
            "本项目参考了论文 2206.04564v6 以及官方仓库 `LuoUndergradXJTU/TwiBot-22` 中对 F / T / G / FT / FG / FTG 六类方法的组织方式。",
            "图模型部分重点借鉴了官方仓库在 TwiBot-20 上的输入结构：description 表示、tweet 表示、数值属性、类别属性和关系边共同进入 GCN / GAT / BotRGCN 风格模型。",
            "需要说明的是，本项目仍然属于课程项目级的复现与重实现，而不是逐文件复刻官方仓库，因此结果可以用于趋势比较，但不应直接表述为严格复现论文原始成绩。",
            "",
            "## 9. 可视化与附件",
            "- `artifacts/figures/model_comparison.png`：测试集模型 F1 对比图",
            "- `artifacts/figures/node2vec_projection.png`：Node2Vec 二维投影",
            "- `artifacts/figures/feature_importance.png`：传统特征重要性",
        ]
    )

    report_path = config.output_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def _pick_best_row(
    metrics: pd.DataFrame,
    split_name: str,
    sort_columns: list[str],
    family: str | None = None,
) -> dict[str, object] | None:
    frame = metrics[metrics["split"] == split_name].copy()
    if family is not None:
        frame = frame[frame["family"] == family].copy()
    if frame.empty:
        return None
    best = frame.sort_values(sort_columns, ascending=False).iloc[0]
    return best.to_dict()


def _to_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_暂无结果_"
    columns = frame.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False):
        rendered = []
        for value in row:
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)
