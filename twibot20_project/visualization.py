from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig
from .experiments import load_cached_artifacts

LOGGER = logging.getLogger(__name__)


def generate_visualizations(config: PipelineConfig) -> None:
    users, _, embeddings = load_cached_artifacts(config)
    users = users.merge(embeddings, on="user_id", how="left").fillna(0.0)

    prediction_path = config.tables_dir / "classification_predictions.csv"
    cluster_path = config.tables_dir / "cluster_assignments.csv"
    edge_path = config.cache_dir / "graph_edges.csv"

    if prediction_path.exists():
        predictions = pd.read_csv(prediction_path)
        best_info = json.loads((config.models_dir / "best_experiment.json").read_text(encoding="utf-8"))
        predictions = predictions[predictions["experiment"] == best_info["best_experiment"]]
        users = users.merge(
            predictions[["user_id", "bot_probability", "prediction"]],
            on="user_id",
            how="left",
        )

    if cluster_path.exists():
        clusters = pd.read_csv(cluster_path)
        users = users.merge(clusters[["user_id", "cluster_id"]], on="user_id", how="left")

    plot_degree_distribution(users, config)
    plot_embedding_map(users, config)

    if edge_path.exists() and cluster_path.exists():
        plot_suspicious_cluster(users, pd.read_csv(edge_path), config)


def plot_degree_distribution(users: pd.DataFrame, config: PipelineConfig) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=users,
        x="graph_total_degree",
        hue="label",
        bins=40,
        stat="density",
        common_norm=False,
        alpha=0.4,
        ax=axis,
    )
    axis.set_title("Degree Distribution by Label")
    axis.set_xlabel("Graph total degree")
    figure.tight_layout()
    figure.savefig(config.figures_dir / "degree_distribution.png", dpi=180)
    plt.close(figure)


def plot_embedding_map(users: pd.DataFrame, config: PipelineConfig) -> None:
    embedding_columns = [column for column in users.columns if column.startswith("dw_")]
    if not embedding_columns:
        LOGGER.warning("No DeepWalk columns found; skipping embedding plot.")
        return

    sample_size = min(config.tsne_sample_size, len(users))
    sample = users.groupby("label", group_keys=False).apply(
        lambda frame: frame.sample(
            n=min(max(1, sample_size // max(1, users["label"].nunique())), len(frame)),
            random_state=config.random_state,
        )
    )
    sample = sample.drop_duplicates("user_id").reset_index(drop=True)

    scaler = StandardScaler()
    matrix = scaler.fit_transform(sample[embedding_columns].to_numpy(dtype=float))
    pca_components = min(32, matrix.shape[0] - 1, matrix.shape[1])
    if pca_components > 1:
        matrix = PCA(n_components=pca_components, random_state=config.random_state).fit_transform(matrix)

    perplexity = min(30, max(5, len(sample) // 20))
    projection = TSNE(
        n_components=2,
        learning_rate="auto",
        init="pca",
        perplexity=perplexity,
        random_state=config.random_state,
    ).fit_transform(matrix)

    plot_df = sample[["user_id", "label"]].copy()
    plot_df["x"] = projection[:, 0]
    plot_df["y"] = projection[:, 1]
    if "cluster_id" in sample:
        plot_df["cluster_id"] = sample["cluster_id"]

    figure, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="label",
        style="label",
        alpha=0.8,
        s=35,
        ax=axis,
    )
    axis.set_title("t-SNE of DeepWalk Embeddings")
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    figure.tight_layout()
    figure.savefig(config.figures_dir / "embedding_tsne.png", dpi=180)
    plt.close(figure)


def plot_suspicious_cluster(users: pd.DataFrame, edges: pd.DataFrame, config: PipelineConfig) -> None:
    if "cluster_id" not in users:
        return

    clustered = users.dropna(subset=["cluster_id"]).copy()
    clustered = clustered[clustered["cluster_id"] != -1]
    if clustered.empty:
        return

    ranked = (
        clustered.groupby("cluster_id")
        .agg(size=("user_id", "size"), bot_ratio=("label_id", "mean"))
        .sort_values(["bot_ratio", "size"], ascending=False)
    )
    chosen_cluster_id = ranked.index[0]
    cluster_users = clustered[clustered["cluster_id"] == chosen_cluster_id].copy()
    cluster_ids = cluster_users["user_id"].tolist()

    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        if row.source_id in cluster_ids and row.target_id in cluster_ids:
            graph.add_edge(row.source_id, row.target_id, weight=float(row.weight))

    if graph.number_of_nodes() == 0:
        return

    if graph.number_of_nodes() > 80:
        ranked_nodes = sorted(graph.degree, key=lambda item: item[1], reverse=True)[:80]
        graph = graph.subgraph([node for node, _ in ranked_nodes]).copy()

    node_df = cluster_users.set_index("user_id").loc[list(graph.nodes())]
    node_colors = node_df["label"].map({"bot": "#d95f02", "human": "#1b9e77"}).tolist()

    figure, axis = plt.subplots(figsize=(9, 7))
    position = nx.spring_layout(graph, seed=config.random_state)
    nx.draw_networkx(
        graph,
        pos=position,
        node_size=180,
        node_color=node_colors,
        edge_color="#b3b3b3",
        with_labels=False,
        alpha=0.9,
        ax=axis,
    )
    axis.set_title(f"Top Suspicious Cluster ({int(chosen_cluster_id)})")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(config.figures_dir / "top_suspicious_cluster.png", dpi=180)
    plt.close(figure)
