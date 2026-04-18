from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib import font_manager

from .config import ProjectConfig
from .graph_models import PRIMARY_METHOD_NAME

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
_AVAILABLE_FONTS = {font.name for font in font_manager.fontManager.ttflist}
matplotlib.rcParams["font.sans-serif"] = [
    font_name for font_name in _CHINESE_FONT_CANDIDATES if font_name in _AVAILABLE_FONTS
] or ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

COLOR_VAL = "#8ecae6"
COLOR_TEST = "#219ebc"
COLOR_FEATURE = "#ed7d31"
COLOR_TEXT = "#5aa897"
COLOR_GRAPH = "#7b6cf6"


def generate_visualizations(config: ProjectConfig) -> None:
    metrics = _safe_read_csv(config.tables_dir / "experiment_metrics.csv")
    diagnostics = _safe_read_csv(config.tables_dir / f"{PRIMARY_METHOD_NAME}_gate_diagnostics.csv")

    _plot_method_performance(metrics, config)
    _plot_split_gate_profile(diagnostics, config)
    _plot_reliability_profiles(diagnostics, config)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _plot_method_performance(metrics: pd.DataFrame, config: ProjectConfig) -> None:
    if metrics.empty:
        return

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    colors = [COLOR_VAL if split == "val" else COLOR_TEST for split in metrics["split"].tolist()]
    for axis in axes:
        axis.set_facecolor("#faf7ef")
        axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)

    axes[0].bar(metrics["split"], metrics["f1"], color=colors)
    axes[0].set_ylim(0.0, max(0.9, float(metrics["f1"].max()) + 0.05))
    axes[0].set_title("F1 Score")
    for index, value in enumerate(metrics["f1"].tolist()):
        axes[0].text(index, value + 0.01, f"{value:.3f}", ha="center")

    axes[1].bar(metrics["split"], metrics["auc_roc"], color=colors)
    axes[1].set_ylim(0.0, max(0.95, float(metrics["auc_roc"].max()) + 0.05))
    axes[1].set_title("AUC-ROC")
    for index, value in enumerate(metrics["auc_roc"].tolist()):
        axes[1].text(index, value + 0.01, f"{value:.3f}", ha="center")

    figure.suptitle("Modality Reliability-Aware Dynamic Fusion", fontsize=15)
    figure.tight_layout()
    figure.savefig(config.figures_dir / "method_performance.png", dpi=220)
    plt.close(figure)


def _plot_split_gate_profile(diagnostics: pd.DataFrame, config: ProjectConfig) -> None:
    if diagnostics.empty:
        return

    subset = diagnostics[diagnostics["split"].isin(["val", "test"])].copy()
    if subset.empty:
        return

    summary = subset.groupby("split", sort=False)[["feature_weight", "text_weight", "graph_weight"]].mean().reset_index()
    positions = range(len(summary))
    width = 0.22
    figure, axis = plt.subplots(figsize=(10.5, 5.0))
    axis.set_facecolor("#faf7ef")
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)

    axis.bar([position - width for position in positions], summary["feature_weight"], width=width, color=COLOR_FEATURE, label="Feature")
    axis.bar(list(positions), summary["text_weight"], width=width, color=COLOR_TEXT, label="Text")
    axis.bar([position + width for position in positions], summary["graph_weight"], width=width, color=COLOR_GRAPH, label="Graph")

    axis.set_xticks(list(positions), summary["split"].tolist())
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Average gate weight")
    axis.set_title("Average modality reliance on validation and test splits")
    axis.legend(frameon=False, ncol=3, loc="upper center")
    figure.tight_layout()
    figure.savefig(config.figures_dir / "modality_gate_profile.png", dpi=220)
    plt.close(figure)


def _plot_reliability_profiles(diagnostics: pd.DataFrame, config: ProjectConfig) -> None:
    if diagnostics.empty:
        return

    subset = diagnostics[diagnostics["split"].isin(["val", "test"])].copy()
    if subset.empty:
        return

    text_q75 = subset["quality_text_richness"].quantile(0.75)
    text_q25 = subset["quality_text_richness"].quantile(0.25)
    graph_q75 = subset["quality_graph_connectivity"].quantile(0.75)
    graph_q25 = subset["quality_graph_connectivity"].quantile(0.25)
    anomaly_q75 = subset["quality_profile_anomaly"].quantile(0.75)

    profile_specs = [
        ("Text-rich / graph-sparse", (subset["quality_text_richness"] >= text_q75) & (subset["quality_graph_connectivity"] <= graph_q25)),
        ("Graph-rich / text-sparse", (subset["quality_graph_connectivity"] >= graph_q75) & (subset["quality_text_richness"] <= text_q25)),
        ("Feature-anomalous", subset["quality_profile_anomaly"] >= anomaly_q75),
    ]

    rows = []
    for profile_name, mask in profile_specs:
        profile_subset = subset[mask].copy()
        if profile_subset.empty:
            continue
        rows.append(
            {
                "profile": profile_name,
                "sample_count": int(len(profile_subset)),
                "feature_weight": float(profile_subset["feature_weight"].mean()),
                "text_weight": float(profile_subset["text_weight"].mean()),
                "graph_weight": float(profile_subset["graph_weight"].mean()),
            }
        )

    if not rows:
        return

    summary = pd.DataFrame(rows)
    summary.to_csv(config.tables_dir / f"{PRIMARY_METHOD_NAME}_profiles.csv", index=False)

    positions = range(len(summary))
    width = 0.22
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.set_facecolor("#faf7ef")
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)

    axis.bar([position - width for position in positions], summary["feature_weight"], width=width, color=COLOR_FEATURE, label="Feature")
    axis.bar(list(positions), summary["text_weight"], width=width, color=COLOR_TEXT, label="Text")
    axis.bar([position + width for position in positions], summary["graph_weight"], width=width, color=COLOR_GRAPH, label="Graph")

    for position, sample_count in zip(list(positions), summary["sample_count"].tolist(), strict=False):
        axis.text(position, 1.01, f"n={sample_count}", ha="center", va="bottom", fontsize=10)

    axis.set_xticks(list(positions), summary["profile"].tolist())
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel("Average gate weight")
    axis.set_title("Different account conditions trigger different modality preferences")
    axis.legend(frameon=False, ncol=3, loc="upper center")
    figure.tight_layout()
    figure.savefig(config.figures_dir / "modality_reliability_profiles.png", dpi=220)
    plt.close(figure)
