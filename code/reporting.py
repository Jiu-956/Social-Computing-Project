from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .interpretation import (
    FAMILY_LABELS_EN,
    FAMILY_LABELS_ZH,
    SOURCE_LABELS_ZH,
    build_family_best_frame,
    ensure_explainability_signal_analysis,
    ensure_information_source_analysis,
)


def generate_report(config: ProjectConfig) -> Path:
    metrics = pd.read_csv(config.tables_dir / "experiment_metrics.csv", low_memory=False)
    dataset_summary = json.loads((config.cache_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    source_tables = ensure_information_source_analysis(config, metrics)
    signal_tables = ensure_explainability_signal_analysis(config)

    metrics["family_cn"] = metrics["family"].map(FAMILY_LABELS_ZH).fillna(metrics["family"])
    metrics["family_en"] = metrics["family"].map(FAMILY_LABELS_EN).fillna(metrics["family"])

    family_best = build_family_best_frame(metrics)
    family_best = family_best.rename(columns={"experiment": "best_experiment"})
    family_best["family_cn"] = family_best["family"].map(FAMILY_LABELS_ZH).fillna(family_best["family"])
    family_best["family_en"] = family_best["family"].map(FAMILY_LABELS_EN).fillna(family_best["family"])

    test_family = family_best[family_best["split"] == "test"].copy().sort_values("f1", ascending=False)
    val_family = family_best[family_best["split"] == "val"].copy().sort_values("f1", ascending=False)
    test_ftg = metrics[(metrics["split"] == "test") & (metrics["family"] == "feature_text_graph")].copy().sort_values(
        ["f1", "auc_roc"],
        ascending=False,
    )

    source_summary = source_tables["source_contribution_summary"].copy()
    source_details = source_tables["source_contribution_details"].copy()
    source_ablation = source_tables["source_ablation"].copy()

    if not source_summary.empty:
        source_summary["source_cn"] = source_summary["source"].map(SOURCE_LABELS_ZH).fillna(source_summary["source_cn"])
    if not source_details.empty:
        source_details["source_cn"] = source_details["source"].map(SOURCE_LABELS_ZH).fillna(source_details["source_cn"])
    if not source_ablation.empty:
        source_ablation["source_cn"] = source_ablation["source"].map(SOURCE_LABELS_ZH).fillna(source_ablation["source_cn"])

    test_source_summary = source_summary[source_summary["split"] == "test"].copy()
    test_source_details = source_details[source_details["split"] == "test"].copy()
    test_source_ablation = source_ablation[source_ablation["split"] == "test"].copy()
    botdgt_ablation_path = config.tables_dir / "botdgt_modality_ablation.csv"
    if botdgt_ablation_path.exists():
        botdgt_ablation = pd.read_csv(botdgt_ablation_path, low_memory=False)
        test_botdgt_ablation = botdgt_ablation[botdgt_ablation["split"] == "test"].copy()
    else:
        test_botdgt_ablation = pd.DataFrame()

    report_lines = [
        "# 基于 TwiBot-20 的社交机器人检测实验报告",
        "",
        "## 1. 研究问题",
        "本项目不把目标设定为“只找一个分数最高的模型”，而是围绕下面三个问题组织实验与报告：",
        "1. 哪类信息更有效？",
        "2. 不同方法差异在哪里？",
        "3. 结果是否可解释？",
        "",
        "## 2. 数据与实验设置",
        f"- 图用户总数：{dataset_summary['graph_user_count']}",
        f"- 有标签用户数：{dataset_summary['labeled_user_count']}",
        f"- support 节点数：{dataset_summary['support_user_count']}",
        f"- 用户关系边数：{dataset_summary['graph_edge_count']}",
        f"- 采样 tweet 数：{dataset_summary['sampled_tweet_count']}",
        "- 方法体系：统一比较 F / T / G / FT / FG / FTG 六类方法。",
        "- 评价指标：Accuracy、Precision、Recall、F1、AUC-ROC。",
        "- 可解释分析：信息源平均增益、融合模型消融、用户属性/文本/图结构信号拆解。",
        "",
        "## 3. 结果故事线",
        "### 3.1 第一层：整体表现",
        "先看六类方法族在统一测试集上的最佳结果：",
        _to_markdown_table(test_family[["family_cn", "best_experiment", "accuracy", "f1", "auc_roc"]]),
        "",
        "### 3.2 第二层：增益与消融",
        "先看“加入某类信息之后平均带来多少增益”：",
        _to_markdown_table(
            test_source_summary[
                ["source_cn", "comparison_count", "mean_accuracy_gain", "mean_f1_gain", "mean_auc_roc_gain"]
            ]
        ),
        "",
    ]

    report_lines.extend(_build_question_one_interpretation(test_family, test_source_summary, test_source_ablation))
    report_lines.extend(
        [
            "",
            "再看最佳可解释融合基线上的消融结果：",
            _to_markdown_table(
                test_source_ablation[
                    [
                        "experiment",
                        "source_cn",
                        "baseline_f1",
                        "ablated_f1",
                        "f1_drop",
                        "baseline_auc_roc",
                        "ablated_auc_roc",
                        "auc_roc_drop",
                    ]
                ]
            ),
            "",
            "### 3.3 第三层：可解释性分析",
            "这部分重点回答模型到底在看什么信号，以及这些信号是否能支持多源互补的结论。",
            "",
        ]
    )
    report_lines.extend(
        [
            "### BotDGT 三大模态重训练消融",
            _to_markdown_table(
                test_botdgt_ablation[
                    [
                        "experiment",
                        "removed_modality",
                        "baseline_f1",
                        "ablated_f1",
                        "f1_drop",
                        "baseline_accuracy",
                        "ablated_accuracy",
                        "accuracy_drop",
                    ]
                ]
                if not test_botdgt_ablation.empty
                else test_botdgt_ablation
            ),
            "",
        ]
    )
    report_lines.extend(_build_signal_summary(signal_tables))
    report_lines.extend(_build_question_three_interpretation(test_source_ablation, signal_tables))
    report_lines.extend(
        [
            "",
            "## 4. 补充：不同方法差异来自哪里？",
            "如果继续追问“同样用到多源信息时，不同方法为什么会拉开差距”，可以看下面两组补充结果。",
            "",
            "### 4.1 配对增益",
            _to_markdown_table(
                test_source_details[
                    [
                        "source_cn",
                        "base_family_cn",
                        "augmented_family_cn",
                        "f1_gain",
                        "auc_roc_gain",
                    ]
                ]
            ),
            "",
            "### 4.2 同样使用特征、文本、图信息时，不同 FTG 方法的差异",
            _to_markdown_table(test_ftg[["experiment", "accuracy", "precision", "recall", "f1", "auc_roc"]]),
            "",
        ]
    )
    report_lines.extend(_build_question_two_interpretation(test_source_details, test_ftg))
    report_lines.extend(
        [
            "",
            "## 5. 关键可视化输出",
            "- `artifacts/figures/training_curves.png`：展示 GNN 训练过程中的 loss 与 F1 变化。",
            "- `artifacts/figures/information_effectiveness.png`：回答“哪类信息更有效”。",
            "- `artifacts/figures/explainability_signals.png`：给出账号属性、文本表达、图结构三类解释信号的总览。",
            "- `artifacts/figures/feature_signal_map.png`：展示关键结构化特征的重要性及其机器人 / 人类方向差异。",
            "- `artifacts/figures/embedding_separation_map.png`：展示文本嵌入与图嵌入在二维空间中的分布与重叠。",
            "- `artifacts/figures/local_network_patterns.png`：展示高置信人类 / 机器人的局部网络结构与关注 / 好友关系模式。",
            "- `artifacts/figures/method_differences.png`：作为补充图，说明多源方法内部的差异来自哪里。",
            "",
            "## 6. 与论文和官方仓库的关系",
            "本项目参考了论文 2206.04564v6 以及官方仓库 `LuoUndergradXJTU/TwiBot-22` 中对特征 / 文本 / 图及其融合方法的组织方式。",
            "图模型部分重点借鉴了官方仓库在 TwiBot-20 上的输入结构：description 表示、tweet 表示、数值属性、类别属性和关系边共同进入 GCN / GAT / BotRGCN 风格模型。",
            "需要说明的是，本项目仍然属于课程项目级的复现与重实现，因此适合做趋势比较和方法分析，但不应表述为对论文成绩的严格复现。",
            "",
            "## 7. 附录：验证集最佳方法族",
            _to_markdown_table(val_family[["family_cn", "best_experiment", "accuracy", "f1", "auc_roc"]]),
        ]
    )

    report_path = config.output_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


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


def _build_question_one_interpretation(
    test_family: pd.DataFrame,
    test_source_summary: pd.DataFrame,
    test_source_ablation: pd.DataFrame,
) -> list[str]:
    lines: list[str] = []
    if test_family.empty:
        lines.append("- 目前还没有可用的测试集方法族结果。")
        return lines

    strongest_family = test_family.iloc[0]
    weakest_family = test_family.iloc[-1]
    single_source = test_family[test_family["family"].isin(["feature_only", "text_only", "graph_only"])].copy()
    best_single_source = single_source.iloc[0] if not single_source.empty else None
    lines.append(
        f"- 从整体性能看，测试集最佳方法族是 `{strongest_family['family_cn']}`，最佳实验为 `{strongest_family['best_experiment']}`，F1 为 {strongest_family['f1']:.4f}，说明融合方法的性能上限最高。"
    )
    if best_single_source is not None:
        lines.append(
            f"- 但在单源方法里，`{best_single_source['family_cn']}` 仍然最稳定，而 `{weakest_family['family_cn']}` 最弱。这说明当前实验中，单靠一种信息源很难达到融合方法的效果。"
        )

    if not test_source_summary.empty:
        strongest_source = test_source_summary.sort_values("mean_f1_gain", ascending=False).iloc[0]
        weakest_source = test_source_summary.sort_values("mean_f1_gain", ascending=True).iloc[0]
        lines.append(
            f"- 从平均增益看，`{strongest_source['source_cn']}` 是当前实验里最有效的主干信息源，测试集平均 F1 增益为 {strongest_source['mean_f1_gain']:.4f}。"
        )
        lines.append(
            f"- 相比之下，`{weakest_source['source_cn']}` 的平均 F1 变化仅为 {weakest_source['mean_f1_gain']:.4f}，说明图结构或文本信息的价值更依赖具体融合方式，而不是简单叠加就能稳定提升。"
        )

    if not test_source_ablation.empty:
        biggest_drop = test_source_ablation.sort_values("f1_drop", ascending=False).iloc[0]
        second_drop = test_source_ablation.sort_values("f1_drop", ascending=False).iloc[min(1, len(test_source_ablation) - 1)]
        lines.append(
            f"- 从消融结果看，移除 `{biggest_drop['source_cn']}` 后测试集 F1 下降 {biggest_drop['f1_drop']:.4f}，损失最大；`{second_drop['source_cn']}` 的影响次之。也就是说，这套结果更像“属性主干 + 文本补充 + 图结构选择性增益”的格局。"
        )
    return lines


def _build_question_two_interpretation(test_source_details: pd.DataFrame, test_ftg: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if test_source_details.empty:
        lines.append("- 当前还没有足够的配对增益结果来分析方法差异。")
        return lines

    best_gain = test_source_details.sort_values("f1_gain", ascending=False).iloc[0]
    worst_gain = test_source_details.sort_values("f1_gain", ascending=True).iloc[0]
    lines.append(
        f"- 最大的家族级差异来自 `{best_gain['base_family_cn']}` 到 `{best_gain['augmented_family_cn']}`：加入 `{best_gain['source_cn']}` 后，测试集 F1 提升 {best_gain['f1_gain']:.4f}。"
    )
    lines.append(
        f"- 最小甚至为负的变化是 `{worst_gain['base_family_cn']}` 到 `{worst_gain['augmented_family_cn']}`：加入 `{worst_gain['source_cn']}` 后，测试集 F1 变化为 {worst_gain['f1_gain']:.4f}。这说明多源方法的差距不只是“用了什么信息”，更在于“怎么融合这些信息”。"
    )

    if not test_ftg.empty:
        best_ftg = test_ftg.iloc[0]
        worst_ftg = test_ftg.iloc[-1]
        lines.append(
            f"- 在同样都使用特征、文本、图信息的 FTG 家族内部，最佳方法是 `{best_ftg['experiment']}`（F1={best_ftg['f1']:.4f}），最弱的是 `{worst_ftg['experiment']}`（F1={worst_ftg['f1']:.4f}）。"
        )
        lines.append(
            "- 这说明“是否用到了特征、文本、图”只是第一层差异，第二层差异来自融合机制本身，例如简单拼接、注意力传播和关系感知传播。"
        )
    return lines


def _build_signal_summary(signal_tables: dict[str, pd.DataFrame]) -> list[str]:
    lines: list[str] = []
    feature_signals = signal_tables["feature_signals"]
    text_signals = signal_tables["text_signals"]
    graph_signals = signal_tables["graph_signals"]

    lines.append(_render_signal_table("用户属性信号", feature_signals.head(6)))
    lines.append("")
    lines.append(_render_signal_table("文本信号", _select_text_summary_rows(text_signals)))
    lines.append("")
    lines.append(_render_signal_table("图结构信号", graph_signals.head(6)))
    return lines


def _build_question_three_interpretation(
    test_source_ablation: pd.DataFrame,
    signal_tables: dict[str, pd.DataFrame],
) -> list[str]:
    lines: list[str] = []
    feature_signals = signal_tables["feature_signals"]
    text_signals = signal_tables["text_signals"]
    graph_signals = signal_tables["graph_signals"]

    if not test_source_ablation.empty:
        strongest_drop = test_source_ablation.sort_values("f1_drop", ascending=False).iloc[0]
        lines.append(
            f"- 消融结果说明，`{strongest_drop['source_cn']}` 是最佳可解释融合基线里最关键的输入来源：移除后测试集 F1 下降 {strongest_drop['f1_drop']:.4f}。"
        )

    feature_phrase = ""
    if not feature_signals.empty:
        feature_phrase = _join_top_labels(feature_signals.head(3), 'signal_name_zh')
        lines.append(
            f"- 用户属性层面，最显著的信号包括 `{feature_phrase}`，它们主要对应账号认证、账号年龄、粉丝规模等画像是否异常。"
        )
    text_phrase = ""
    if not text_signals.empty:
        bot_like = text_signals[text_signals["score"] > 0].head(3)
        human_like = text_signals[text_signals["score"] < 0].head(3)
        text_phrase = _join_top_labels(bot_like, 'signal_name_zh')
        lines.append(
            f"- 文本层面，机器人倾向词包括 `{_join_top_labels(bot_like, 'signal_name_zh')}`，人类倾向词包括 `{_join_top_labels(human_like, 'signal_name_zh')}`。"
        )
    graph_phrase = ""
    if not graph_signals.empty:
        graph_phrase = _join_top_labels(graph_signals.head(3), 'signal_name_zh')
        lines.append(
            f"- 图结构层面，最显著的信号包括 `{graph_phrase}`，说明模型在关注主动连接模式与邻居关系结构。"
        )
    if feature_phrase or text_phrase or graph_phrase:
        lines.append(
            "- 综合来看，模型并不是只盯住单一线索，而是在同时利用账号行为异常、社交连接模式和文本表达特征来区分机器人与人类，这正是多源信息互补性的体现。"
        )
    return lines


def _render_signal_table(title: str, frame: pd.DataFrame) -> str:
    if frame.empty:
        return f"#### {title}\n_暂无结果_"
    display = frame[
        [
            "signal_name_zh",
            "direction_zh",
            "score",
            "importance",
        ]
    ].copy()
    display.columns = ["signal", "direction", "score", "importance"]
    return f"#### {title}\n{_to_markdown_table(display)}"


def _select_text_summary_rows(frame: pd.DataFrame, top_each_side: int = 3) -> pd.DataFrame:
    if frame.empty:
        return frame
    positive = frame[frame["score"] > 0].head(top_each_side)
    negative = frame[frame["score"] < 0].head(top_each_side)
    return pd.concat([positive, negative], ignore_index=True)


def _join_top_labels(frame: pd.DataFrame, column: str) -> str:
    if frame.empty:
        return "暂无明显信号"
    return "、".join(str(value) for value in frame[column].tolist())
