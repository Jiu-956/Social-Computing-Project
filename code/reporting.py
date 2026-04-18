from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .graph_models import PRIMARY_METHOD_NAME


def generate_report(config: ProjectConfig) -> Path:
    metrics = _safe_read_csv(config.tables_dir / "experiment_metrics.csv")
    method_summary = _safe_read_csv(config.tables_dir / "method_summary.csv")
    diagnostics = _safe_read_csv(config.tables_dir / f"{PRIMARY_METHOD_NAME}_gate_diagnostics.csv")
    dataset_summary = json.loads((config.cache_dir / "dataset_summary.json").read_text(encoding="utf-8"))

    report_lines = [
        "# Modality Reliability-Aware Dynamic Fusion",
        "",
        "## 1. Method Goal",
        "This branch keeps only one method for TwiBot-20 bot detection: `Modality Reliability-Aware Dynamic Fusion`.",
        "The model does not fuse profile, text, and graph features with a fixed rule.",
        "Instead, it learns which modality should be trusted more for each specific account.",
        "",
        "## 2. Core Logic",
        "1. Encode profile signals into a feature representation.",
        "2. Encode description and sampled tweets into a text representation.",
        "3. Propagate information over `follow` and `friend` edges to obtain a graph representation.",
        "4. Estimate modality reliability from quality indicators such as text richness, graph connectivity, and profile anomaly.",
        "5. Use a gating network plus lightweight attention fusion to dynamically assign feature/text/graph weights.",
        "",
        "## 3. Why This Method Is Different",
        "- It replaces fixed multimodal fusion with sample-level adaptive fusion.",
        "- It explicitly distinguishes modality quality from modality content.",
        "- It produces interpretable gate weights for each account instead of only a final label.",
        "",
        "## 4. Dataset Summary",
        f"- Graph users: {dataset_summary.get('graph_user_count', 0)}",
        f"- Labeled users: {dataset_summary.get('labeled_user_count', 0)}",
        f"- Support users: {dataset_summary.get('support_user_count', 0)}",
        f"- Relation edges: {dataset_summary.get('graph_edge_count', 0)}",
        f"- Sampled tweets: {dataset_summary.get('sampled_tweet_count', 0)}",
        "",
        "## 5. Metrics",
        _render_metrics(metrics, method_summary),
        "",
        "## 6. Learned Modality Preference",
        _render_gate_summary(diagnostics),
        "",
        "## 7. Output Files",
        f"- `artifacts/models/{PRIMARY_METHOD_NAME}.pt`: trained model weights.",
        "- `artifacts/tables/experiment_metrics.csv`: validation and test metrics.",
        "- `artifacts/tables/method_summary.csv`: compact method summary.",
        f"- `artifacts/tables/{PRIMARY_METHOD_NAME}_gate_diagnostics.csv`: per-user gate weights and quality indicators.",
    ]

    report_path = config.output_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _render_metrics(metrics: pd.DataFrame, method_summary: pd.DataFrame) -> str:
    if not metrics.empty:
        return _to_markdown_table(metrics[["experiment", "split", "accuracy", "precision", "recall", "f1", "auc_roc"]].copy())

    if method_summary.empty:
        return "_No metrics available yet._"

    row = method_summary.iloc[0].to_dict()
    frame = pd.DataFrame(
        [
            {
                "experiment": row.get("method", PRIMARY_METHOD_NAME),
                "split": "val",
                "accuracy": row.get("val_accuracy"),
                "precision": row.get("val_precision"),
                "recall": row.get("val_recall"),
                "f1": row.get("val_f1"),
                "auc_roc": row.get("val_auc_roc"),
            },
            {
                "experiment": row.get("method", PRIMARY_METHOD_NAME),
                "split": "test",
                "accuracy": row.get("test_accuracy"),
                "precision": row.get("test_precision"),
                "recall": row.get("test_recall"),
                "f1": row.get("test_f1"),
                "auc_roc": row.get("test_auc_roc"),
            },
        ]
    )
    return _to_markdown_table(frame)


def _render_gate_summary(diagnostics: pd.DataFrame) -> str:
    if diagnostics.empty:
        return "_No gate diagnostics available yet._"

    subset = diagnostics[diagnostics["split"].isin(["val", "test"])].copy()
    if subset.empty:
        return "_No gate diagnostics available yet._"

    rows = []
    for split_name, split_frame in subset.groupby("split", sort=False):
        row = {
            "split": split_name,
            "samples": int(len(split_frame)),
            "feature_weight": float(split_frame["feature_weight"].mean()),
            "text_weight": float(split_frame["text_weight"].mean()),
            "graph_weight": float(split_frame["graph_weight"].mean()),
        }
        if "dominant_modality" in split_frame.columns:
            dominant_share = split_frame["dominant_modality"].value_counts(normalize=True)
            row["dominant_feature_share"] = float(dominant_share.get("feature", 0.0))
            row["dominant_text_share"] = float(dominant_share.get("text", 0.0))
            row["dominant_graph_share"] = float(dominant_share.get("graph", 0.0))
        rows.append(row)
    return _to_markdown_table(pd.DataFrame(rows))


def _to_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No results available._"

    columns = frame.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
