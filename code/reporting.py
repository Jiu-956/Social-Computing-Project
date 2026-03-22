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
        "# Experiment Report",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## The Story",
        "",
        build_story(best_info, comparison_df, cluster_metrics, community_summary),
        "",
        "## Data and Graph Overview",
        "",
        f"- Labeled users: {network_summary.get('labeled_user_count', 'N/A')}",
        f"- Graph users: {network_summary.get('graph_user_count', 'N/A')}",
        f"- Directed user edges: {network_summary.get('directed_user_edges', 'N/A')}",
        f"- Graph density: {network_summary.get('density', 0.0):.4f}",
        f"- Average clustering: {network_summary.get('average_clustering', 0.0):.4f}",
        "",
        "## Model Comparison",
        "",
        "Top test-set results:",
        "",
        dataframe_to_markdown(
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
            ].head(10),
            precision=4,
        ),
        "",
        "## Family-Level Reading",
        "",
        build_family_summary(comparison_df),
        "",
        "## Suspicious Group Discovery",
        "",
        build_cluster_summary(cluster_metrics, cluster_summary),
        "",
        "## Community and Modularity Analysis",
        "",
        build_community_summary(community_summary, community_metrics),
        "",
        "## Output Files",
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
        return "The report could not build a comparison story because no experiment metrics were found."

    best_row = comparison_df.iloc[0]
    baseline_rows = comparison_df[comparison_df["family"].astype(str).str.startswith("baseline")]
    baseline_row = baseline_rows.iloc[0] if not baseline_rows.empty else None
    baseline_f1 = float(baseline_row["f1"]) if baseline_row is not None else 0.0
    uplift = float(best_row["f1"]) - baseline_f1

    story_lines = [
        f"The current winner is `{best_row['experiment']}` from the `{best_row['family']}` family.",
        f"Its test F1 is `{float(best_row['f1']):.4f}` and AUC-ROC is `{float(best_row['auc_roc']):.4f}`.",
    ]
    if baseline_row is not None:
        story_lines.append(
            f"Compared with the strongest original baseline `{baseline_row['experiment']}`, the F1 uplift is `{uplift:.4f}`."
        )

    if str(best_row["family"]).startswith("gnn"):
        story_lines.append(
            "This suggests the best gains come from jointly using richer node features and message passing over the social graph."
        )
    elif "transformer" in str(best_row["text_encoder"]):
        story_lines.append(
            "This suggests semantic text encoding is adding signal that TF-IDF could not recover cleanly."
        )
    elif "node2vec" in str(best_row["graph_encoder"]):
        story_lines.append(
            "This suggests the structural gain mainly comes from a stronger graph embedding rather than only hand-crafted graph statistics."
        )
    else:
        story_lines.append(
            "This suggests the original machine learning pipeline remains a strong baseline and should stay in the comparison set."
        )

    if cluster_metrics:
        story_lines.append(
            f"On suspicious-group discovery, clustering found `{cluster_metrics.get('cluster_count', 0)}` clusters from `{cluster_metrics.get('candidate_count', 0)}` high-risk accounts."
        )
    if not community_summary.empty:
        best_method = community_summary.sort_values("modularity", ascending=False).iloc[0]
        story_lines.append(
            f"For community structure, `{best_method['method']}` achieved the highest modularity at `{float(best_method['modularity']):.4f}`, indicating non-random coordination structure among suspicious accounts."
        )

    return " ".join(story_lines)


def build_family_summary(comparison_df: pd.DataFrame) -> str:
    if comparison_df.empty:
        return "No model comparison table is available."

    best_per_family = comparison_df.sort_values(["family", "f1", "auc_roc"], ascending=[True, False, False]).groupby("family").head(1)
    lines = []
    for row in best_per_family.itertuples(index=False):
        lines.append(
            f"- `{row.family}` is led by `{row.experiment}` with F1 `{float(row.f1):.4f}` and AUC-ROC `{float(row.auc_roc):.4f}`."
        )
    return "\n".join(lines)


def build_cluster_summary(cluster_metrics: dict, cluster_summary: pd.DataFrame) -> str:
    if not cluster_metrics:
        return "Cluster outputs are not available."
    lines = [
        f"- Candidate accounts: `{cluster_metrics.get('candidate_count', 0)}`",
        f"- Discovered clusters: `{cluster_metrics.get('cluster_count', 0)}`",
        f"- Purity: `{float(cluster_metrics.get('purity', 0.0)):.4f}`",
        f"- NMI: `{float(cluster_metrics.get('nmi', 0.0)):.4f}`",
        f"- Noise ratio: `{float(cluster_metrics.get('noise_ratio', 0.0)):.4f}`",
    ]
    if not cluster_summary.empty:
        top_cluster = cluster_summary.iloc[0]
        lines.append(
            f"- The most suspicious cluster has size `{int(top_cluster['size'])}`, bot ratio `{float(top_cluster['bot_ratio']):.4f}`, and density `{float(top_cluster['density']):.4f}`."
        )
    return "\n".join(lines)


def build_community_summary(community_summary: pd.DataFrame, community_metrics: dict) -> str:
    if community_summary.empty:
        return "Community analysis outputs are not available."

    lines = [
        dataframe_to_markdown(
            community_summary[
                [
                    "method",
                    "community_count",
                    "modularity",
                    "largest_community_size",
                    "average_community_size",
                    "average_bot_ratio",
                ]
            ],
            precision=4,
        )
    ]
    best_method = community_summary.sort_values("modularity", ascending=False).iloc[0]
    lines.append(
        f"The highest-modularity partition is `{best_method['method']}` with modularity `{float(best_method['modularity']):.4f}`."
    )
    return "\n\n".join(lines)


def dataframe_to_markdown(df: pd.DataFrame, precision: int = 4) -> str:
    if df.empty:
        return "_No rows available._"

    display_df = df.copy()
    for column in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda value: f"{value:.{precision}f}")
    header = "| " + " | ".join(display_df.columns.astype(str).tolist()) + " |"
    separator = "| " + " | ".join(["---"] * len(display_df.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in display_df.to_numpy()]
    return "\n".join([header, separator, *rows])
