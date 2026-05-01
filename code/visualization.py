from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .config import ProjectConfig, safe_slug
from .interpretation import (
    FAMILY_LABELS_EN,
    FAMILY_LABELS_ZH,
    SOURCE_LABELS_EN,
    SOURCE_LABELS_ZH,
    build_family_best_frame,
    ensure_explainability_signal_analysis,
    ensure_information_source_analysis,
)

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

COLOR_PROFILE = "#ed7d31"
COLOR_TEXT = "#5aa897"
COLOR_GRAPH = "#7b6cf6"
COLOR_NEUTRAL = "#b9b2a6"
COLOR_HUMAN = "#4c78a8"
COLOR_BOT = "#f28e2b"
COLOR_SUPPORT = "#c7c7c7"
COLOR_GRID = "#d9d9d9"
COLOR_FOLLOW = "#8ecae6"
COLOR_FRIEND = "#ffb703"
COLOR_ACCENT = "#2f4858"
COLOR_PANEL = "#fbf7ef"

SOURCE_COLORS = {
    "feature": COLOR_PROFILE,
    "text": COLOR_TEXT,
    "graph": COLOR_GRAPH,
}


def generate_visualizations(config: ProjectConfig) -> None:
    metrics_path = config.tables_dir / "experiment_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("experiment_metrics.csv does not exist. Run the training step first.")

    metrics = pd.read_csv(metrics_path, low_memory=False)
    users = pd.read_csv(config.cache_dir / "users.csv", low_memory=False)
    training_history_path = config.tables_dir / "training_history.csv"
    if training_history_path.exists():
        training_history = pd.read_csv(training_history_path, low_memory=False)
        _plot_training_history(training_history, config)
    analysis_tables = ensure_information_source_analysis(config, metrics)
    signal_tables = ensure_explainability_signal_analysis(config)

    _plot_feature_importance(signal_tables, config)
    _plot_information_effectiveness(metrics, analysis_tables, config)
    _plot_method_differences(metrics, analysis_tables["source_contribution_details"], config)
    _plot_explainability_signals(signal_tables, config)
    _plot_embedding_separation_map(users, config)
    _plot_local_network_patterns(users, metrics, config)
    _cleanup_unused_figures(config)


def _plot_model_comparison(metrics: pd.DataFrame, config: ProjectConfig) -> None:
    test_metrics = metrics[metrics["split"] == "test"].sort_values("f1", ascending=True)
    if test_metrics.empty:
        return
    plt.figure(figsize=(11, 7))
    plt.barh(test_metrics["experiment"], test_metrics["f1"], color="#4472c4")
    plt.xlabel("F1-score")
    plt.title("TwiBot-20 Test F1 Comparison")
    plt.tight_layout()
    plt.savefig(config.figures_dir / "model_comparison.png", dpi=200)
    plt.close()


def _plot_training_history(history: pd.DataFrame, config: ProjectConfig) -> None:
    if history.empty or "experiment" not in history.columns:
        return

    experiments = [experiment for experiment in history["experiment"].dropna().astype(str).unique().tolist()]
    if not experiments:
        return

    ncols = 2
    nrows = int(np.ceil(len(experiments) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(14, 4.8 * nrows), squeeze=False)

    for index, experiment in enumerate(experiments):
        row = index // ncols
        col = index % ncols
        axis = axes[row][col]
        subset = history[history["experiment"] == experiment].sort_values("epoch")
        if subset.empty:
            axis.axis("off")
            continue

        loss_axis = axis
        f1_axis = axis.twinx()
        loss_axis.plot(subset["epoch"], subset["train_loss"], label="train loss", color="#4472c4", linewidth=1.6)
        loss_axis.plot(subset["epoch"], subset["val_loss"], label="val loss", color="#ed7d31", linewidth=1.6)
        f1_axis.plot(subset["epoch"], subset["val_f1"], label="val F1", color="#2f855a", linewidth=1.8)
        loss_axis.set_title(experiment)
        loss_axis.set_xlabel("Epoch")
        loss_axis.set_ylabel("Loss")
        f1_axis.set_ylabel("Val F1")
        loss_axis.grid(True, alpha=0.25)
        loss_axis.set_ylim(bottom=0.0)
        f1_axis.set_ylim(0.0, 1.0)

        if index == 0:
            loss_handles, loss_labels = loss_axis.get_legend_handles_labels()
            f1_handles, f1_labels = f1_axis.get_legend_handles_labels()
            loss_axis.legend(loss_handles + f1_handles, loss_labels + f1_labels, frameon=False, loc="upper right")

    for index in range(len(experiments), nrows * ncols):
        row = index // ncols
        col = index % ncols
        axes[row][col].axis("off")

    figure.suptitle("Training History Overview")
    figure.tight_layout()
    figure.savefig(config.figures_dir / "training_curves.png", dpi=220)
    plt.close(figure)


def _plot_node2vec_projection(users: pd.DataFrame, config: ProjectConfig) -> None:
    projection = _get_graph_embedding_projection(users, config)
    if projection.empty:
        return
    fig, axis = plt.subplots(figsize=(8, 6))
    _plot_embedding_panel(
        axis,
        projection,
        title="Node2Vec 邻域投影",
        subtitle="平衡抽样后的有标签节点，使用 t-SNE 投影",
    )
    fig.tight_layout()
    fig.savefig(config.figures_dir / "node2vec_projection.png", dpi=220)
    plt.close(fig)


def _plot_feature_importance(signal_tables: dict[str, pd.DataFrame], config: ProjectConfig) -> None:
    feature_signals = signal_tables["feature_signals"]
    graph_signals = signal_tables["graph_signals"]
    if feature_signals.empty and graph_signals.empty:
        return

    feature_top = (
        feature_signals.sort_values(["importance", "ranking_score"], ascending=False).head(10).copy()
        if not feature_signals.empty
        else pd.DataFrame()
    )
    graph_top = (
        graph_signals.sort_values(["importance", "ranking_score"], ascending=False).head(8).copy()
        if not graph_signals.empty
        else pd.DataFrame()
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    _plot_importance_panel(
        axes[0, 0],
        feature_top,
        color=COLOR_PROFILE,
        title="模型最依赖的账号属性特征",
        xlabel="随机森林重要性",
    )
    _plot_direction_panel(
        axes[0, 1],
        feature_top,
        title="账号属性在人类与机器人间的方向差异",
        xlabel="更像人类  ←  标准化均值差  →  更像机器人",
    )
    _plot_importance_panel(
        axes[1, 0],
        graph_top,
        color=COLOR_GRAPH,
        title="模型最依赖的图结构特征",
        xlabel="随机森林重要性",
    )
    _plot_direction_panel(
        axes[1, 1],
        graph_top,
        title="哪些局部图模式更像机器人",
        xlabel="更像人类  ←  标准化均值差  →  更像机器人",
    )
    fig.suptitle("关键特征与方向差异", fontsize=17)
    fig.tight_layout()
    fig.savefig(config.figures_dir / "feature_signal_map.png", dpi=220)
    plt.close(fig)


def _plot_source_contribution(summary: pd.DataFrame, config: ProjectConfig) -> None:
    if summary.empty:
        return

    ordered_sources = [source for source in ("feature", "text", "graph") if source in set(summary["source"].astype(str))]
    if not ordered_sources:
        return

    f1_pivot = summary.pivot(index="source", columns="split", values="mean_f1_gain").reindex(ordered_sources).fillna(0.0)
    auc_pivot = summary.pivot(index="source", columns="split", values="mean_auc_roc_gain").reindex(ordered_sources).fillna(0.0)
    split_order = [split for split in ("val", "test") if split in f1_pivot.columns]
    if not split_order:
        return

    positions = np.arange(len(ordered_sources))
    width = 0.35 if len(split_order) > 1 else 0.5
    offsets = np.linspace(-(len(split_order) - 1) * width / 2, (len(split_order) - 1) * width / 2, len(split_order))
    source_labels = [SOURCE_LABELS_EN.get(source, source) for source in ordered_sources]
    colors = {"val": "#8ecae6", "test": "#219ebc"}

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for axis, pivot, metric_name in (
        (axes[0], f1_pivot, "Mean F1 gain"),
        (axes[1], auc_pivot, "Mean AUC gain"),
    ):
        for offset, split_name in zip(offsets, split_order):
            values = pivot[split_name].to_numpy(dtype=float)
            axis.bar(
                positions + offset,
                values,
                width=width,
                label=split_name.upper(),
                color=colors.get(split_name, "#4472c4"),
            )
        axis.axhline(0.0, color="#666666", linewidth=0.8)
        axis.set_xticks(positions, source_labels)
        axis.set_ylabel(metric_name)
        axis.set_title(metric_name)

    axes[1].legend(frameon=False)
    figure.suptitle("Information Source Contribution Gains")
    figure.tight_layout()
    figure.savefig(config.figures_dir / "source_contribution.png", dpi=200)
    plt.close(figure)


def _plot_source_ablation(ablation: pd.DataFrame, config: ProjectConfig) -> None:
    if ablation.empty:
        return

    ordered_sources = [source for source in ("feature", "text", "graph") if source in set(ablation["source"].astype(str))]
    if not ordered_sources:
        return

    f1_pivot = ablation.pivot(index="source", columns="split", values="f1_drop").reindex(ordered_sources).fillna(0.0)
    auc_pivot = ablation.pivot(index="source", columns="split", values="auc_roc_drop").reindex(ordered_sources).fillna(0.0)
    split_order = [split for split in ("val", "test") if split in f1_pivot.columns]
    if not split_order:
        return

    positions = np.arange(len(ordered_sources))
    width = 0.35 if len(split_order) > 1 else 0.5
    offsets = np.linspace(-(len(split_order) - 1) * width / 2, (len(split_order) - 1) * width / 2, len(split_order))
    source_labels = [SOURCE_LABELS_EN.get(source, source) for source in ordered_sources]
    colors = {"val": "#ffb703", "test": "#fb8500"}
    experiment_names = sorted(set(ablation["experiment"].astype(str)))
    title_suffix = experiment_names[0] if len(experiment_names) == 1 else "best interpretable fusion model"

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for axis, pivot, metric_name in (
        (axes[0], f1_pivot, "F1 drop after source removal"),
        (axes[1], auc_pivot, "AUC drop after source removal"),
    ):
        for offset, split_name in zip(offsets, split_order):
            values = pivot[split_name].to_numpy(dtype=float)
            axis.bar(
                positions + offset,
                values,
                width=width,
                label=split_name.upper(),
                color=colors.get(split_name, "#ed7d31"),
            )
        axis.axhline(0.0, color="#666666", linewidth=0.8)
        axis.set_xticks(positions, source_labels)
        axis.set_ylabel(metric_name)
        axis.set_title(metric_name)

    axes[1].legend(frameon=False)
    figure.suptitle(f"Source Ablation on {title_suffix}")
    figure.tight_layout()
    figure.savefig(config.figures_dir / "source_ablation.png", dpi=200)
    plt.close(figure)


def _plot_information_effectiveness(
    metrics: pd.DataFrame,
    analysis_tables: dict[str, pd.DataFrame],
    config: ProjectConfig,
) -> None:
    family_best = build_family_best_frame(metrics)
    test_family = family_best[family_best["split"] == "test"].copy()
    if test_family.empty:
        return
    test_family["family_label"] = test_family["family"].map(FAMILY_LABELS_ZH).fillna(test_family["family"])
    test_family = test_family.sort_values("f1", ascending=True)

    summary = analysis_tables["source_contribution_summary"]
    ablation = analysis_tables["source_ablation"]
    test_summary = summary[summary["split"] == "test"].copy()
    test_ablation = ablation[ablation["split"] == "test"].copy()
    source_order = [source for source in ("feature", "text", "graph") if source in set(test_summary["source"].astype(str))]
    if not source_order:
        source_order = [source for source in ("feature", "text", "graph") if source in set(test_ablation["source"].astype(str))]
    if not source_order:
        return

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.5, 5.8),
        gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]},
    )
    _apply_axis_style(axes[0])
    _apply_axis_style(axes[1])
    _apply_axis_style(axes[2])

    family_colors = [COLOR_NEUTRAL] * len(test_family)
    family_colors[-1] = COLOR_PROFILE
    axes[0].barh(test_family["family_label"], test_family["f1"], color=family_colors)
    axes[0].set_title("整体表现：测试集最佳方法族")
    axes[0].set_xlabel("F1")
    axes[0].set_xlim(0.60, max(0.88, float(test_family["f1"].max()) + 0.02))
    for index, value in enumerate(test_family["f1"]):
        axes[0].text(value + 0.004, index, f"{value:.3f}", va="center", fontsize=10)

    if not test_summary.empty:
        test_summary = test_summary.set_index("source").reindex(source_order).reset_index()
        axes[1].bar(
            [SOURCE_LABELS_ZH.get(source, source) for source in test_summary["source"]],
            test_summary["mean_f1_gain"],
            color=[SOURCE_COLORS.get(source, COLOR_NEUTRAL) for source in test_summary["source"]],
        )
        axes[1].set_title("增益分析：加入哪类信息更有效")
        axes[1].set_ylabel("平均 F1 增益")
        axes[1].axhline(0.0, color="#666666", linewidth=0.8)
        for idx, value in enumerate(test_summary["mean_f1_gain"]):
            axes[1].text(idx, value + (0.005 if value >= 0 else -0.01), f"{value:.3f}", ha="center", fontsize=10)

    if not test_ablation.empty:
        test_ablation = test_ablation.set_index("source").reindex(source_order).reset_index()
        axes[2].bar(
            [SOURCE_LABELS_ZH.get(source, source) for source in test_ablation["source"]],
            test_ablation["f1_drop"],
            color=[SOURCE_COLORS.get(source, COLOR_NEUTRAL) for source in test_ablation["source"]],
        )
        axes[2].set_title("消融分析：移除哪类信息损失最大")
        axes[2].set_ylabel("F1 下降值")
        axes[2].axhline(0.0, color="#666666", linewidth=0.8)
        for idx, value in enumerate(test_ablation["f1_drop"]):
            axes[2].text(idx, value + (0.01 if value >= 0 else -0.02), f"{value:.3f}", ha="center", fontsize=10)

    fig.suptitle("结果主线一：整体表现与信息贡献", fontsize=16)
    fig.tight_layout()
    fig.savefig(config.figures_dir / "information_effectiveness.png", dpi=220)
    plt.close(fig)


def _plot_method_differences(metrics: pd.DataFrame, source_details: pd.DataFrame, config: ProjectConfig) -> None:
    test_details = source_details[source_details["split"] == "test"].copy()
    ftg_metrics = metrics[(metrics["split"] == "test") & (metrics["family"] == "feature_text_graph")].copy()
    if test_details.empty and ftg_metrics.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), gridspec_kw={"width_ratios": [1.2, 1.0]})
    _apply_axis_style(axes[0])
    _apply_axis_style(axes[1])

    if not test_details.empty:
        test_details["comparison"] = test_details.apply(
            lambda row: f"{row['base_family_cn']} -> {row['augmented_family_cn']}",
            axis=1,
        )
        test_details = test_details.sort_values("f1_gain", ascending=True)
        min_gain = float(test_details["f1_gain"].min())
        max_gain = float(test_details["f1_gain"].max())
        gain_span = max_gain - min_gain
        gain_padding = max(0.01, gain_span * 0.12)
        axes[0].barh(
            test_details["comparison"],
            test_details["f1_gain"],
            color=[SOURCE_COLORS.get(source, COLOR_NEUTRAL) for source in test_details["source"]],
        )
        axes[0].set_title("方法差异来自哪里")
        axes[0].set_xlabel("测试集 F1 增益")
        axes[0].axvline(0.0, color="#666666", linewidth=0.8)
        axes[0].set_xlim(min(0.0, min_gain) - gain_padding, max_gain + gain_padding)
        for index, value in enumerate(test_details["f1_gain"]):
            x_pos = value + gain_padding * 0.18 if value >= 0 else value - gain_padding * 0.18
            ha = "left" if value >= 0 else "right"
            axes[0].text(x_pos, index, f"{value:.3f}", va="center", ha=ha, fontsize=10)

    if not ftg_metrics.empty:
        ftg_metrics["display_name"] = ftg_metrics["experiment"].map(_format_ftg_experiment_name)
        ftg_metrics = ftg_metrics.sort_values("f1", ascending=True)
        axes[1].barh(
            ftg_metrics["display_name"],
            ftg_metrics["f1"],
            color=[COLOR_PROFILE if "BotRGCN" in name else COLOR_NEUTRAL for name in ftg_metrics["display_name"]],
        )
        axes[1].set_title("同样使用特征+文本+图时，不同模型设计的差异")
        axes[1].set_xlabel("测试集 F1")
        axes[1].set_xlim(min(0.78, float(ftg_metrics["f1"].min()) - 0.01), float(ftg_metrics["f1"].max()) + 0.03)
        for index, value in enumerate(ftg_metrics["f1"]):
            axes[1].text(value + 0.003, index, f"{value:.3f}", va="center", fontsize=10)

    fig.suptitle("结果补充：不同方法的差异来自何处", fontsize=16)
    fig.tight_layout()
    fig.savefig(config.figures_dir / "method_differences.png", dpi=220)
    plt.close(fig)


def _plot_explainability_signals(signal_tables: dict[str, pd.DataFrame], config: ProjectConfig) -> None:
    feature_signals = signal_tables["feature_signals"]
    text_signals = signal_tables["text_signals"]
    graph_signals = signal_tables["graph_signals"]
    if feature_signals.empty and text_signals.empty and graph_signals.empty:
        return

    feature_top = feature_signals.head(6).sort_values("score", ascending=True)
    graph_top = graph_signals.head(6).sort_values("score", ascending=True)
    text_top = _select_text_plot_rows(text_signals)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.3))
    _plot_signed_signal_panel(
        axes[0],
        feature_top,
        title="账号属性信号",
        xlabel="更像人类  ←  标准化均值差  →  更像机器人",
    )
    _plot_signed_signal_panel(
        axes[1],
        text_top,
        title="文本表达信号",
        xlabel="更像人类  ←  逻辑回归系数  →  更像机器人",
    )
    _plot_signed_signal_panel(
        axes[2],
        graph_top,
        title="图结构信号",
        xlabel="更像人类  ←  标准化均值差  →  更像机器人",
    )
    fig.suptitle("结果主线三：模型到底在看什么信号", fontsize=16)
    fig.tight_layout()
    fig.savefig(config.figures_dir / "explainability_signals.png", dpi=220)
    plt.close(fig)


def _plot_signed_signal_panel(axis: plt.Axes, frame: pd.DataFrame, title: str, xlabel: str) -> None:
    _apply_axis_style(axis)
    if frame.empty:
        axis.set_title(title)
        axis.text(0.5, 0.5, "暂无可解释信号", ha="center", va="center")
        return

    labels = frame["signal_name_zh"].fillna(frame["signal_name_en"])
    values = frame["score"]
    colors = [COLOR_BOT if value >= 0 else COLOR_HUMAN for value in values]
    axis.barh(labels, values, color=colors)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.axvline(0.0, color="#666666", linewidth=0.8)
    max_abs = max(abs(float(values.min())), abs(float(values.max())))
    axis.set_xlim(-max_abs * 1.25, max_abs * 1.25)
    for index, value in enumerate(values):
        x_pos = value + max_abs * 0.04 if value >= 0 else value - max_abs * 0.04
        ha = "left" if value >= 0 else "right"
        axis.text(x_pos, index, f"{value:.2f}", va="center", ha=ha, fontsize=9.5)


def _select_text_plot_rows(text_signals: pd.DataFrame, top_each_side: int = 6) -> pd.DataFrame:
    if text_signals.empty:
        return text_signals
    bot_like = text_signals[text_signals["score"] > 0].head(top_each_side)
    human_like = text_signals[text_signals["score"] < 0].head(top_each_side)
    combined = pd.concat([human_like, bot_like], ignore_index=True)
    return combined.sort_values("score", ascending=True).reset_index(drop=True)


def _apply_axis_style(axis: plt.Axes) -> None:
    axis.grid(axis="x", color=COLOR_GRID, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _format_ftg_experiment_name(experiment_name: str) -> str:
    mapping = {
        "feature_text_graph_tfidf_node2vec_logistic_regression": "TF-IDF + Node2Vec + LR",
        "feature_text_graph_gcn": "GCN",
        "feature_text_graph_gat": "GAT",
        "feature_text_graph_botrgcn": "BotRGCN",
        "feature_text_graph_botsai": "BotSAI",
        "feature_text_graph_botdgt": "BotDGT",
    }
    return mapping.get(experiment_name, experiment_name)


def _plot_embedding_separation_map(users: pd.DataFrame, config: ProjectConfig) -> None:
    panels: list[tuple[str, str, pd.DataFrame]] = []
    text_projection = _get_text_embedding_projection(users, config)
    graph_projection = _get_graph_embedding_projection(users, config)

    if not text_projection.empty:
        panels.append(("文本嵌入空间", "SentenceTransformer 表示 + t-SNE", text_projection))
    if not graph_projection.empty:
        panels.append(("图嵌入空间", "Node2Vec 表示 + t-SNE", graph_projection))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(7.6 * len(panels), 6.4), squeeze=False)
    for axis, (title, subtitle, frame) in zip(axes[0], panels, strict=False):
        _plot_embedding_panel(axis, frame, title=title, subtitle=subtitle)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_HUMAN, markersize=8, label="人类用户"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_BOT, markersize=8, label="机器人用户"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("嵌入空间中的机器人 / 人类分布", fontsize=16)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(config.figures_dir / "embedding_separation_map.png", dpi=220)
    plt.close(fig)


def _plot_local_network_patterns(users: pd.DataFrame, metrics: pd.DataFrame, config: ProjectConfig) -> None:
    edges_path = config.cache_dir / "graph_edges.csv"
    predictions_path = config.tables_dir / "experiment_predictions.csv"
    if not edges_path.exists() or not predictions_path.exists():
        return

    graph_models = metrics[
        (metrics["split"] == "test") & (metrics["family"].isin(["graph_only", "feature_graph", "feature_text_graph"]))
    ].copy()
    if graph_models.empty:
        return
    best_graph_experiment = str(
        graph_models.sort_values(["f1", "auc_roc", "accuracy"], ascending=False).iloc[0]["experiment"]
    )

    predictions = pd.read_csv(
        predictions_path,
        usecols=["experiment", "split", "user_id", "true_label", "pred_label", "bot_probability"],
        low_memory=False,
    )
    predictions = predictions[
        (predictions["experiment"] == best_graph_experiment) & (predictions["split"] == "test")
    ].copy()
    if predictions.empty:
        return

    edges = pd.read_csv(edges_path, low_memory=False)
    user_summary = users[["user_id", "label_id", "split", "total_in_degree", "total_out_degree"]].copy()
    labeled_lookup = users.set_index("user_id")["label_id"].to_dict()
    labeled_users = {user_id for user_id, label in labeled_lookup.items() if int(label) >= 0}
    adjacency = _build_neighbor_lookup(edges)

    human_anchor = _select_representative_anchor(
        predictions=predictions,
        user_summary=user_summary,
        adjacency=adjacency,
        labeled_users=labeled_users,
        label_id=0,
    )
    bot_anchor = _select_representative_anchor(
        predictions=predictions,
        user_summary=user_summary,
        adjacency=adjacency,
        labeled_users=labeled_users,
        label_id=1,
    )
    if human_anchor is None and bot_anchor is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), squeeze=False)
    panels = [
        ("高置信人类用户的局部邻域", human_anchor),
        ("高置信机器人用户的局部邻域", bot_anchor),
    ]
    for axis, (title, anchor) in zip(axes[0], panels, strict=False):
        if anchor is None:
            axis.axis("off")
            axis.text(0.5, 0.5, "暂无可展示的代表性邻域", ha="center", va="center")
            continue
        local_graph, summary = _build_local_relation_graph(
            edges=edges,
            anchor_id=str(anchor["user_id"]),
            labeled_lookup=labeled_lookup,
            max_neighbors=18,
        )
        _plot_local_network_panel(
            axis,
            graph=local_graph,
            summary=summary,
            title=title,
            anchor_probability=float(anchor["bot_probability"]),
        )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_HUMAN, markersize=8, label="人类节点"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_BOT, markersize=8, label="机器人节点"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SUPPORT, markersize=8, label="辅助节点 / 未标注"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COLOR_ACCENT, markersize=8, label="中心账号"),
        Line2D([0], [0], color=COLOR_FOLLOW, linewidth=2.2, label="关注关系"),
        Line2D([0], [0], color=COLOR_FRIEND, linewidth=2.2, label="好友关系"),
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle("局部网络结构：高置信样本的关系模式", fontsize=16)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.savefig(config.figures_dir / "local_network_patterns.png", dpi=220)
    plt.close(fig)


def _plot_importance_panel(axis: plt.Axes, frame: pd.DataFrame, color: str, title: str, xlabel: str) -> None:
    _apply_axis_style(axis)
    if frame.empty:
        axis.set_title(title)
        axis.text(0.5, 0.5, "暂无特征重要性结果", ha="center", va="center")
        return

    ordered = frame.sort_values("importance", ascending=True)
    labels = ordered["signal_name_zh"].fillna(ordered["signal_name_en"])
    axis.barh(labels, ordered["importance"], color=color, alpha=0.9)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    max_value = float(ordered["importance"].max())
    for index, value in enumerate(ordered["importance"]):
        axis.text(value + max(max_value * 0.03, 0.003), index, f"{value:.3f}", va="center", fontsize=9)


def _plot_direction_panel(axis: plt.Axes, frame: pd.DataFrame, title: str, xlabel: str) -> None:
    _apply_axis_style(axis)
    if frame.empty:
        axis.set_title(title)
        axis.text(0.5, 0.5, "暂无方向差异结果", ha="center", va="center")
        return

    ordered = frame.sort_values("importance", ascending=True)
    colors = [COLOR_BOT if value >= 0 else COLOR_HUMAN for value in ordered["score"]]
    labels = ordered["signal_name_zh"].fillna(ordered["signal_name_en"])
    axis.barh(labels, ordered["score"], color=colors)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.axvline(0.0, color="#666666", linewidth=0.8)
    max_abs = max(abs(float(ordered["score"].min())), abs(float(ordered["score"].max())))
    axis.set_xlim(-max_abs * 1.2, max_abs * 1.2)
    for index, value in enumerate(ordered["score"]):
        x_pos = value + max_abs * 0.04 if value >= 0 else value - max_abs * 0.04
        ha = "left" if value >= 0 else "right"
        axis.text(x_pos, index, f"{value:.2f}", va="center", ha=ha, fontsize=9)


def _plot_embedding_panel(axis: plt.Axes, frame: pd.DataFrame, title: str, subtitle: str) -> None:
    axis.set_facecolor(COLOR_PANEL)
    axis.grid(color=COLOR_GRID, linewidth=0.8, alpha=0.45)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.set_xticks([])
    axis.set_yticks([])

    for label_id, color in ((0, COLOR_HUMAN), (1, COLOR_BOT)):
        subset = frame[frame["label_id"] == label_id]
        if subset.empty:
            continue
        axis.scatter(subset["x"], subset["y"], s=18, alpha=0.34, color=color, edgecolors="none")
        _draw_covariance_ellipse(axis, subset["x"].to_numpy(dtype=float), subset["y"].to_numpy(dtype=float), color=color)
        centroid = subset[["x", "y"]].mean().to_numpy(dtype=float)
        axis.scatter(
            [centroid[0]],
            [centroid[1]],
            s=170,
            color=color,
            edgecolors="white",
            linewidths=1.8,
            zorder=5,
        )

    axis.set_title(title)
    axis.text(0.02, 0.98, subtitle, transform=axis.transAxes, va="top", fontsize=10, color="#444444")
    class_counts = frame["label_id"].value_counts().to_dict()
    axis.text(
        0.02,
        0.06,
        f"样本数：{len(frame)}\n人类：{class_counts.get(0, 0)}  机器人：{class_counts.get(1, 0)}",
        transform=axis.transAxes,
        va="bottom",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#dddddd"},
    )


def _draw_covariance_ellipse(axis: plt.Axes, x: np.ndarray, y: np.ndarray, color: str, n_std: float = 1.8) -> None:
    if len(x) < 3 or len(y) < 3:
        return
    covariance = np.cov(x, y)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues <= 0):
        return
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(
        xy=(float(np.mean(x)), float(np.mean(y))),
        width=float(width),
        height=float(height),
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=0.10,
        linewidth=2,
    )
    axis.add_patch(ellipse)


def _get_text_embedding_projection(users: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    labeled_users = users.loc[users["label_id"] >= 0, ["user_id", "label_id"]].copy()
    preferred = config.cache_dir / f"combined_transformer_embeddings_{safe_slug(config.transformer_model_name)}.joblib"
    candidates = [preferred] if preferred.exists() else []
    candidates.extend(sorted(path for path in config.cache_dir.glob("combined_transformer_embeddings_*.joblib") if path not in candidates))
    for candidate in candidates:
        loaded = joblib.load(candidate)
        if not isinstance(loaded, pd.DataFrame) or "user_id" not in loaded.columns:
            continue
        frame = labeled_users.merge(loaded, on="user_id", how="inner")
        embedding_columns = [column for column in frame.columns if str(column).startswith("combined_text_emb_")]
        if not embedding_columns:
            continue
        projection_path = config.tables_dir / "text_embedding_projection.csv"
        return _load_or_compute_projection(
            frame=frame,
            feature_columns=embedding_columns,
            projection_path=projection_path,
            input_paths=[candidate, Path(__file__)],
            random_state=config.random_state,
        )
    return pd.DataFrame(columns=["user_id", "label_id", "x", "y"])


def _get_graph_embedding_projection(users: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    embedding_path = config.cache_dir / "node2vec_embeddings.csv"
    if not embedding_path.exists():
        return pd.DataFrame(columns=["user_id", "label_id", "x", "y"])

    labeled_users = users.loc[users["label_id"] >= 0, ["user_id", "label_id"]].copy()
    embeddings = pd.read_csv(embedding_path, low_memory=False)
    frame = labeled_users.merge(embeddings, on="user_id", how="inner")
    embedding_columns = [column for column in frame.columns if str(column).startswith("n2v_")]
    if not embedding_columns:
        return pd.DataFrame(columns=["user_id", "label_id", "x", "y"])
    projection_path = config.tables_dir / "graph_embedding_projection.csv"
    return _load_or_compute_projection(
        frame=frame,
        feature_columns=embedding_columns,
        projection_path=projection_path,
        input_paths=[embedding_path, Path(__file__)],
        random_state=config.random_state,
    )


def _load_or_compute_projection(
    frame: pd.DataFrame,
    feature_columns: list[str],
    projection_path: Path,
    input_paths: list[Path],
    random_state: int,
    max_points: int = 1600,
) -> pd.DataFrame:
    if _output_is_fresh(projection_path, input_paths):
        cached = pd.read_csv(projection_path, low_memory=False)
        if {"user_id", "label_id", "x", "y"}.issubset(cached.columns):
            return cached

    sampled = _balanced_sample(frame[["user_id", "label_id", *feature_columns]], max_points=max_points, random_state=random_state)
    if sampled.empty:
        return pd.DataFrame(columns=["user_id", "label_id", "x", "y"])

    matrix = sampled[feature_columns].to_numpy(dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.shape[1] > 32 and matrix.shape[0] > 32:
        matrix = PCA(n_components=32, random_state=random_state).fit_transform(matrix)

    if matrix.shape[0] >= 6:
        perplexity = float(max(5, min(35, (matrix.shape[0] - 1) // 3)))
        coordinates = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=random_state,
            max_iter=1000,
        ).fit_transform(matrix)
    elif min(matrix.shape[0], matrix.shape[1]) >= 2:
        coordinates = PCA(n_components=2, random_state=random_state).fit_transform(matrix)
    else:
        coordinates = np.zeros((matrix.shape[0], 2), dtype=np.float32)

    projection = sampled[["user_id", "label_id"]].copy()
    projection["x"] = coordinates[:, 0]
    projection["y"] = coordinates[:, 1]
    projection.to_csv(projection_path, index=False)
    return projection


def _balanced_sample(frame: pd.DataFrame, max_points: int, random_state: int) -> pd.DataFrame:
    if frame.empty or len(frame) <= max_points:
        return frame.reset_index(drop=True)

    sampled_groups = []
    class_count = max(1, frame["label_id"].nunique())
    per_class = max(1, max_points // class_count)
    for _, group in frame.groupby("label_id", sort=False):
        sampled_groups.append(group.sample(n=min(len(group), per_class), random_state=random_state))
    sampled = pd.concat(sampled_groups, ignore_index=True)
    if len(sampled) > max_points:
        sampled = sampled.sample(n=max_points, random_state=random_state)
    return sampled.reset_index(drop=True)


def _output_is_fresh(output_path: Path, input_paths: list[Path]) -> bool:
    if not output_path.exists():
        return False
    existing_inputs = [path for path in input_paths if path.exists()]
    if not existing_inputs:
        return False
    newest_input_time = max(path.stat().st_mtime for path in existing_inputs)
    return output_path.stat().st_mtime >= newest_input_time


def _build_neighbor_lookup(edges: pd.DataFrame) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in edges.itertuples(index=False):
        source_id = str(row.source_id)
        target_id = str(row.target_id)
        adjacency[source_id].add(target_id)
        adjacency[target_id].add(source_id)
    return adjacency


def _select_representative_anchor(
    predictions: pd.DataFrame,
    user_summary: pd.DataFrame,
    adjacency: dict[str, set[str]],
    labeled_users: set[str],
    label_id: int,
) -> pd.Series | None:
    candidates = predictions.merge(user_summary, on="user_id", how="left")
    candidates = candidates[
        (candidates["true_label"] == label_id) & (candidates["pred_label"] == label_id)
    ].copy()
    if candidates.empty:
        return None

    candidates["total_degree"] = (
        candidates["total_in_degree"].fillna(0.0) + candidates["total_out_degree"].fillna(0.0)
    )
    candidates["confidence"] = (
        candidates["bot_probability"] if label_id == 1 else 1.0 - candidates["bot_probability"]
    )
    candidates = candidates[candidates["total_degree"] >= 6].sort_values("confidence", ascending=False).head(250).copy()
    if candidates.empty:
        return None

    candidates["labeled_neighbor_count"] = candidates["user_id"].map(
        lambda user_id: sum(1 for neighbor in adjacency.get(str(user_id), set()) if neighbor in labeled_users)
    )
    target_degree = float(candidates["total_degree"].median())
    candidates["degree_gap"] = (candidates["total_degree"] - target_degree).abs()
    candidates["selection_score"] = (
        candidates["confidence"] * 6.0
        + candidates["labeled_neighbor_count"] * 0.6
        - candidates["degree_gap"] * 0.08
    )
    return candidates.sort_values(["selection_score", "confidence"], ascending=False).iloc[0]


def _build_local_relation_graph(
    edges: pd.DataFrame,
    anchor_id: str,
    labeled_lookup: dict[str, int],
    max_neighbors: int,
) -> tuple[nx.Graph, dict[str, float]]:
    graph = nx.Graph()
    graph.add_node(anchor_id, label_id=int(labeled_lookup.get(anchor_id, -1)))

    incident = edges[(edges["source_id"] == anchor_id) | (edges["target_id"] == anchor_id)].copy()
    if incident.empty:
        return graph, {
            "anchor_id": anchor_id,
            "node_count": 1,
            "human_count": 0,
            "bot_count": 0,
            "support_count": 0,
            "follow_edges": 0,
            "friend_edges": 0,
        }

    incident["neighbor_id"] = np.where(incident["source_id"] == anchor_id, incident["target_id"], incident["source_id"])
    neighbor_priority = (
        incident.assign(is_friend=(incident["relation"] == "friend").astype(int))
        .groupby("neighbor_id", sort=False)
        .agg(edge_count=("relation", "size"), friend_count=("is_friend", "sum"))
        .reset_index()
    )
    neighbor_priority["is_labeled"] = neighbor_priority["neighbor_id"].map(
        lambda node_id: int(labeled_lookup.get(str(node_id), -1) >= 0)
    )
    neighbor_priority = neighbor_priority.sort_values(
        ["is_labeled", "friend_count", "edge_count"],
        ascending=[False, False, False],
    )
    direct_limit = min(10, max_neighbors)
    direct_neighbors = neighbor_priority["neighbor_id"].astype(str).head(direct_limit).tolist()

    neighbor_set = set(direct_neighbors)
    two_hop = edges[(edges["source_id"].isin(neighbor_set)) | (edges["target_id"].isin(neighbor_set))].copy()
    if not two_hop.empty:
        two_hop = two_hop[(two_hop["source_id"] != anchor_id) & (two_hop["target_id"] != anchor_id)].copy()
        two_hop["candidate_id"] = np.where(
            two_hop["source_id"].isin(neighbor_set) & ~two_hop["target_id"].isin(neighbor_set),
            two_hop["target_id"],
            np.where(
                two_hop["target_id"].isin(neighbor_set) & ~two_hop["source_id"].isin(neighbor_set),
                two_hop["source_id"],
                "",
            ),
        )
        two_hop["via_neighbor"] = np.where(two_hop["source_id"].isin(neighbor_set), two_hop["source_id"], two_hop["target_id"])
        two_hop = two_hop[(two_hop["candidate_id"] != "") & (two_hop["candidate_id"] != anchor_id)].copy()
        if not two_hop.empty:
            two_hop_priority = (
                two_hop.assign(is_friend=(two_hop["relation"] == "friend").astype(int))
                .groupby("candidate_id", sort=False)
                .agg(
                    shared_direct_neighbors=("via_neighbor", "nunique"),
                    edge_count=("relation", "size"),
                    friend_count=("is_friend", "sum"),
                )
                .reset_index()
            )
            two_hop_priority["is_labeled"] = two_hop_priority["candidate_id"].map(
                lambda node_id: int(labeled_lookup.get(str(node_id), -1) >= 0)
            )
            preferred_two_hop = two_hop_priority[
                (two_hop_priority["shared_direct_neighbors"] >= 2) | (two_hop_priority["is_labeled"] == 1)
            ].copy()
            if preferred_two_hop.empty:
                preferred_two_hop = two_hop_priority
            remaining_slots = max(0, max_neighbors - len(direct_neighbors))
            second_hop_neighbors = (
                preferred_two_hop.sort_values(
                    ["shared_direct_neighbors", "is_labeled", "friend_count", "edge_count"],
                    ascending=[False, False, False, False],
                )["candidate_id"]
                .astype(str)
                .head(remaining_slots)
                .tolist()
            )
        else:
            second_hop_neighbors = []
    else:
        second_hop_neighbors = []

    selected_nodes = {anchor_id, *direct_neighbors, *second_hop_neighbors}

    local_edges = edges[edges["source_id"].isin(selected_nodes) & (edges["target_id"].isin(selected_nodes))].copy()
    if local_edges.empty:
        local_edges = incident[incident["neighbor_id"].astype(str).isin(selected_neighbors)].copy()

    pair_relations: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in local_edges.itertuples(index=False):
        source_id = str(row.source_id)
        target_id = str(row.target_id)
        if source_id == target_id:
            continue
        pair_relations[tuple(sorted((source_id, target_id)))].add(str(row.relation))

    for node_id in selected_nodes:
        graph.add_node(str(node_id), label_id=int(labeled_lookup.get(str(node_id), -1)))
    for (left, right), relations in pair_relations.items():
        relation = "friend" if "friend" in relations else "follow"
        graph.add_edge(left, right, relation=relation)

    labels = [int(data.get("label_id", -1)) for _, data in graph.nodes(data=True)]
    summary = {
        "anchor_id": anchor_id,
        "node_count": graph.number_of_nodes(),
        "human_count": sum(1 for label in labels if label == 0),
        "bot_count": sum(1 for label in labels if label == 1),
        "support_count": sum(1 for label in labels if label < 0),
        "follow_edges": sum(1 for _, _, data in graph.edges(data=True) if data.get("relation") == "follow"),
        "friend_edges": sum(1 for _, _, data in graph.edges(data=True) if data.get("relation") == "friend"),
    }
    return graph, summary


def _plot_local_network_panel(
    axis: plt.Axes,
    graph: nx.Graph,
    summary: dict[str, float],
    title: str,
    anchor_probability: float,
) -> None:
    axis.set_facecolor(COLOR_PANEL)
    axis.axis("off")
    axis.set_title(title)

    if graph.number_of_nodes() <= 1:
        axis.text(0.5, 0.5, "邻域过小，暂不绘制", ha="center", va="center")
        return

    anchor_id = str(summary["anchor_id"])
    positions = nx.spring_layout(
        graph,
        seed=42,
        pos={anchor_id: np.array([0.0, 0.0])},
        fixed=[anchor_id],
        iterations=250,
    )

    follow_edges = [(left, right) for left, right, data in graph.edges(data=True) if data.get("relation") == "follow"]
    friend_edges = [(left, right) for left, right, data in graph.edges(data=True) if data.get("relation") == "friend"]
    if follow_edges:
        nx.draw_networkx_edges(graph, positions, edgelist=follow_edges, edge_color=COLOR_FOLLOW, width=1.8, alpha=0.55, ax=axis)
    if friend_edges:
        nx.draw_networkx_edges(graph, positions, edgelist=friend_edges, edge_color=COLOR_FRIEND, width=2.2, alpha=0.65, ax=axis)

    for label_id, color in ((-1, COLOR_SUPPORT), (0, COLOR_HUMAN), (1, COLOR_BOT)):
        nodelist = [
            node_id
            for node_id, data in graph.nodes(data=True)
            if node_id != anchor_id and int(data.get("label_id", -1)) == label_id
        ]
        if nodelist:
            nx.draw_networkx_nodes(
                graph,
                positions,
                nodelist=nodelist,
                node_color=color,
                node_size=180,
                linewidths=0.8,
                edgecolors="white",
                ax=axis,
            )

    anchor_label_id = int(graph.nodes[anchor_id].get("label_id", -1))
    if anchor_label_id == 1:
        anchor_color = COLOR_BOT
    elif anchor_label_id == 0:
        anchor_color = COLOR_HUMAN
    else:
        anchor_color = COLOR_ACCENT

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[anchor_id],
        node_color=anchor_color,
        node_size=420,
        node_shape="D",
        linewidths=2.2,
        edgecolors=COLOR_ACCENT,
        ax=axis,
    )
    axis.text(
        positions[anchor_id][0],
        positions[anchor_id][1],
        "中心",
        ha="center",
        va="center",
        fontsize=8.5,
        color="white",
        fontweight="bold",
    )
    axis.text(
        0.03,
        0.03,
        (
            f"节点数：{summary['node_count']}\n"
            f"人类：{summary['human_count']}  机器人：{summary['bot_count']}  辅助节点：{summary['support_count']}\n"
            f"关注边：{summary['follow_edges']}  好友边：{summary['friend_edges']}\n"
            f"机器人概率：{anchor_probability:.3f}"
        ),
        transform=axis.transAxes,
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#dddddd"},
    )


def _cleanup_unused_figures(config: ProjectConfig) -> None:
    obsolete = [
        "model_comparison.png",
        "node2vec_projection.png",
        "source_contribution.png",
        "source_ablation.png",
        "feature_importance.png",
        "performance_story_cn.png",
        "detection_signals_cn.png",
    ]
    for filename in obsolete:
        path = config.figures_dir / filename
        if path.exists():
            path.unlink()
