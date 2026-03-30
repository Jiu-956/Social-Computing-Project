from __future__ import annotations

import joblib
import matplotlib
import pandas as pd
from sklearn.decomposition import PCA

from .config import ProjectConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_visualizations(config: ProjectConfig) -> None:
    metrics_path = config.tables_dir / "experiment_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("experiment_metrics.csv does not exist. Run the training step first.")

    metrics = pd.read_csv(metrics_path, low_memory=False)
    users = pd.read_csv(config.cache_dir / "users.csv", low_memory=False)
    _plot_model_comparison(metrics, config)
    _plot_node2vec_projection(users, config)
    _plot_feature_importance(config)


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


def _plot_node2vec_projection(users: pd.DataFrame, config: ProjectConfig) -> None:
    embedding_path = config.cache_dir / "node2vec_embeddings.csv"
    if not embedding_path.exists():
        return
    embeddings = pd.read_csv(embedding_path, low_memory=False)
    labeled = users[users["label_id"] >= 0].merge(embeddings, on="user_id", how="inner")
    if labeled.empty:
        return

    if len(labeled) > config.visualization_sample_size:
        per_class = max(1, config.visualization_sample_size // max(1, labeled["label_id"].nunique()))
        sampled_groups = []
        for _, group in labeled.groupby("label_id", sort=False):
            sampled_groups.append(group.sample(n=min(len(group), per_class), random_state=config.random_state))
        labeled = pd.concat(sampled_groups, ignore_index=True)

    embedding_columns = [column for column in labeled.columns if column.startswith("n2v_")]
    if not embedding_columns:
        return
    projected = PCA(n_components=2, random_state=config.random_state).fit_transform(labeled[embedding_columns])
    colors = labeled["label_id"].map({0: "#2ca02c", 1: "#d62728"}).to_numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(projected[:, 0], projected[:, 1], c=colors, alpha=0.7, s=12)
    plt.title("Node2Vec 2D Projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(config.figures_dir / "node2vec_projection.png", dpi=200)
    plt.close()


def _plot_feature_importance(config: ProjectConfig) -> None:
    artifact_path = config.models_dir / "feature_only_random_forest.joblib"
    if not artifact_path.exists():
        return
    artifact = joblib.load(artifact_path)
    model = artifact.get("model")
    columns = artifact.get("numeric_columns", [])
    if not hasattr(model, "feature_importances_"):
        return
    importance = pd.DataFrame({"feature": columns, "importance": model.feature_importances_}).sort_values(
        "importance",
        ascending=False,
    )
    importance = importance.head(12).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(importance["feature"], importance["importance"], color="#ed7d31")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(config.figures_dir / "feature_importance.png", dpi=200)
    plt.close()
