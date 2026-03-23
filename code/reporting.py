from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import PipelineConfig


def generate_report(config: PipelineConfig) -> Path:
    network_summary = load_json(config.cache_dir / "network_summary.json")
    best_info = load_json(config.models_dir / "best_experiment.json")
    comparison_df = pd.read_csv(config.tables_dir / "classification_comparison.csv")
    cluster_summary = safe_read_csv(config.tables_dir / "cluster_summary.csv")
    cluster_metrics = load_json(config.tables_dir / "cluster_metrics.json")
    community_summary = safe_read_csv(config.tables_dir / "community_summary.csv")
    community_metrics = load_json(config.tables_dir / "community_metrics.json")

    report_lines = [
        "# 实验报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 核心结论",
        "",
        build_story(best_info, comparison_df, cluster_metrics, community_summary),
        "",
        "## 数据与图概况",
        "",
        f"- 标注账号数：`{network_summary.get('labeled_user_count', 'N/A')}`",
        f"- 图中账号数：`{network_summary.get('graph_user_count', 'N/A')}`",
        f"- 有向关系边数：`{network_summary.get('directed_user_edges', 'N/A')}`",
        f"- 图密度：`{network_summary.get('density', 0.0):.4f}`",
        f"- 平均聚类系数：`{network_summary.get('average_clustering', 0.0):.4f}`",
        "",
        "## 模型比较",
        "",
        "测试集 Top 结果：",
        "",
        dataframe_to_markdown(
            rename_comparison_columns(
                comparison_df[
                    [
                        "rank",
                        "experiment",
                        "family",
                        "model_type",
                        "text_encoder",
                        "graph_encoder",
                        "f1",
                        "auc_roc",
                        "delta_vs_best_baseline_f1",
                    ]
                ].head(10)
            ),
            precision=4,
        ),
        "",
        "## 方法族解读",
        "",
        build_family_summary(comparison_df),
        "",
        "## 可疑群体发现",
        "",
        build_cluster_summary(cluster_metrics, cluster_summary),
        "",
        "## 社区与模块度分析",
        "",
        build_community_summary(community_summary, community_metrics),
        "",
        "## 结果文件",
        "",
        "- `result/tables/classification_metrics.csv`",
        "- `result/tables/classification_comparison.csv`",
        "- `result/tables/cluster_summary.csv`",
        "- `result/tables/community_summary.csv`",
        "- `result/report.md`",
    ]

    report_path = config.output_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def rename_comparison_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "rank": "排名",
            "experiment": "实验",
            "family": "方法族",
            "model_type": "模型类型",
            "text_encoder": "文本编码",
            "graph_encoder": "图编码",
            "f1": "F1",
            "auc_roc": "AUC-ROC",
            "delta_vs_best_baseline_f1": "相对最佳基线F1提升",
        }
    )


def rename_community_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "method": "方法",
            "community_count": "社区数量",
            "modularity": "模块度",
            "largest_community_size": "最大社区规模",
            "average_community_size": "平均社区规模",
            "average_bot_ratio": "平均机器人占比",
        }
    )


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_story(best_info: dict, comparison_df: pd.DataFrame, cluster_metrics: dict, community_summary: pd.DataFrame) -> str:
    if comparison_df.empty:
        return "当前没有找到实验指标，因此暂时无法生成结果叙事。"

    best_experiment = best_info.get("best_experiment")
    if best_experiment and "experiment" in comparison_df.columns:
        best_candidates = comparison_df[comparison_df["experiment"] == best_experiment]
        best_row = best_candidates.iloc[0] if not best_candidates.empty else comparison_df.iloc[0]
    else:
        best_row = comparison_df.iloc[0]

    baseline_rows = comparison_df[comparison_df["family"].astype(str).str.startswith("baseline")]
    baseline_row = baseline_rows.iloc[0] if not baseline_rows.empty else None
    baseline_f1 = float(baseline_row["f1"]) if baseline_row is not None else 0.0
    uplift = float(best_row["f1"]) - baseline_f1

    story_lines = [
        f"当前表现最好的实验是 `{best_row['experiment']}`，属于 `{best_row['family']}` 方法族。",
        f"它在测试集上的 F1 为 `{float(best_row['f1']):.4f}`，AUC-ROC 为 `{float(best_row['auc_roc']):.4f}`。",
    ]
    if baseline_row is not None:
        story_lines.append(
            f"相对于最强的原始 baseline `{baseline_row['experiment']}`，F1 提升了 `{uplift:.4f}`。"
        )

    if str(best_row["family"]).startswith("gnn"):
        story_lines.append("这说明最明显的性能收益来自图上传播机制与更丰富节点特征的联合使用。")
    elif "transformer" in str(best_row["text_encoder"]):
        story_lines.append("这说明 Transformer 文本语义表示补充了 TF-IDF 难以稳定捕获的判别信号。")
    elif "node2vec" in str(best_row["graph_encoder"]):
        story_lines.append("这说明性能提升主要来自更强的图结构表示，而不只是手工图统计特征。")
    else:
        story_lines.append("这说明原始机器学习基线仍然具备较强竞争力，适合作为后续增强模型的比较参照。")

    if cluster_metrics:
        story_lines.append(
            f"在可疑群体发现阶段，聚类方法从 `{cluster_metrics.get('candidate_count', 0)}` 个高风险账号中识别出了 `{cluster_metrics.get('cluster_count', 0)}` 个群体。"
        )
    if not community_summary.empty:
        best_method = community_summary.sort_values("modularity", ascending=False).iloc[0]
        story_lines.append(
            f"在社区发现阶段，`{best_method['method']}` 的模块度最高，达到 `{float(best_method['modularity']):.4f}`，说明高风险账号之间存在较明显的非随机组织结构。"
        )

    return " ".join(story_lines)


def build_family_summary(comparison_df: pd.DataFrame) -> str:
    if comparison_df.empty:
        return "当前没有模型对比结果。"

    best_per_family = comparison_df.sort_values(["family", "f1", "auc_roc"], ascending=[True, False, False]).groupby("family").head(1)
    lines = []
    for row in best_per_family.itertuples(index=False):
        lines.append(
            f"- `{row.family}` 方法族中表现最好的是 `{row.experiment}`，其 F1 为 `{float(row.f1):.4f}`，AUC-ROC 为 `{float(row.auc_roc):.4f}`。"
        )
    return "\n".join(lines)


def build_cluster_summary(cluster_metrics: dict, cluster_summary: pd.DataFrame) -> str:
    if not cluster_metrics:
        return "当前没有可疑群体发现结果。"

    lines = [
        f"- 候选高风险账号数：`{cluster_metrics.get('candidate_count', 0)}`",
        f"- 识别出的群体数量：`{cluster_metrics.get('cluster_count', 0)}`",
        f"- Purity：`{float(cluster_metrics.get('purity', 0.0)):.4f}`",
        f"- NMI：`{float(cluster_metrics.get('nmi', 0.0)):.4f}`",
        f"- 噪声点比例：`{float(cluster_metrics.get('noise_ratio', 0.0)):.4f}`",
    ]
    if not cluster_summary.empty:
        top_cluster = cluster_summary.iloc[0]
        lines.append(
            f"- 最可疑的群体规模为 `{int(top_cluster['size'])}`，机器人占比为 `{float(top_cluster['bot_ratio']):.4f}`，子图密度为 `{float(top_cluster['density']):.4f}`。"
        )
    return "\n".join(lines)


def build_community_summary(community_summary: pd.DataFrame, community_metrics: dict) -> str:
    if community_summary.empty:
        return "当前没有社区分析结果。"

    lines = []
    candidate_count = community_metrics.get("candidate_count")
    candidate_with_edges = community_metrics.get("candidate_with_edges")
    isolated_candidate_count = community_metrics.get("isolated_candidate_count")
    subgraph_edge_count = community_metrics.get("subgraph_edge_count")
    connected_component_count = community_metrics.get("connected_component_count")

    if candidate_count is not None:
        lines.append(f"- 高风险候选账号总数：`{int(candidate_count)}`")
    if candidate_with_edges is not None:
        lines.append(f"- 进入社区子图的候选账号数：`{int(candidate_with_edges)}`")
    if isolated_candidate_count is not None:
        lines.append(f"- 未进入社区子图的孤立候选账号数：`{int(isolated_candidate_count)}`")
    if subgraph_edge_count is not None:
        lines.append(f"- 候选子图中的边数：`{int(subgraph_edge_count)}`")
    if connected_component_count is not None:
        lines.append(f"- 候选子图的连通分量数：`{int(connected_component_count)}`")

    lines.append(
        dataframe_to_markdown(
            rename_community_columns(
                community_summary[
                    [
                        "method",
                        "community_count",
                        "modularity",
                        "largest_community_size",
                        "average_community_size",
                        "average_bot_ratio",
                    ]
                ]
            ),
            precision=4,
        )
    )
    best_method = community_summary.sort_values("modularity", ascending=False).iloc[0]
    lines.append(
        f"模块度最高的划分方法是 `{best_method['method']}`，其模块度为 `{float(best_method['modularity']):.4f}`。"
    )
    return "\n\n".join(lines)


def dataframe_to_markdown(df: pd.DataFrame, precision: int = 4) -> str:
    if df.empty:
        return "_暂无数据。_"

    display_df = df.copy()
    for column in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda value: f"{value:.{precision}f}")
    header = "| " + " | ".join(display_df.columns.astype(str).tolist()) + " |"
    separator = "| " + " | ".join(["---"] * len(display_df.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in display_df.to_numpy()]
    return "\n".join([header, separator, *rows])
