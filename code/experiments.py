from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig
from .gnn import run_gnn_experiment
from .text_embeddings import load_or_compute_text_embeddings

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ExperimentOutputs:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    best_experiment: str


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    family: str
    estimator_kind: str
    numeric_columns: list[str]
    use_tfidf: bool = False
    use_transformer: bool = False
    graph_encoder: str = "none"
    text_encoder: str = "none"


def run_classification_experiments(config: PipelineConfig) -> ExperimentOutputs:
    users, manifest, embeddings = load_cached_artifacts(config)
    graph_edges = pd.read_csv(config.cache_dir / "graph_edges.csv")
    users = users.merge(embeddings, on="user_id", how="left").fillna(0.0)

    transformer_embeddings = None
    transformer_columns: list[str] = []
    try:
        transformer_embeddings = load_or_compute_text_embeddings(config, users[["user_id", "combined_text"]].copy())
        transformer_columns = [column for column in transformer_embeddings.columns if column != "user_id"]
        users = users.merge(transformer_embeddings, on="user_id", how="left").fillna(0.0)
    except Exception as exc:  # pragma: no cover - external model availability varies
        LOGGER.warning("Transformer text embeddings are unavailable and will be skipped: %s", exc)

    profile_columns = manifest["profile_numeric_columns"]
    graph_columns = manifest.get("graph_numeric_columns", [])
    deepwalk_columns = manifest.get("deepwalk_embedding_columns", [column for column in embeddings.columns if column.startswith("dw_")])
    node2vec_columns = manifest.get("node2vec_embedding_columns", [column for column in embeddings.columns if column.startswith("n2v_")])

    experiment_specs = [
        ExperimentSpec(
            name="profile_text_logreg",
            family="baseline_feature_text",
            estimator_kind="logreg",
            numeric_columns=profile_columns,
            use_tfidf=True,
            graph_encoder="none",
            text_encoder="tfidf",
        ),
        ExperimentSpec(
            name="graph_profile_rf",
            family="baseline_graph_deepwalk",
            estimator_kind="rf",
            numeric_columns=profile_columns + graph_columns + deepwalk_columns,
            graph_encoder="deepwalk",
        ),
        ExperimentSpec(
            name="full_logreg",
            family="baseline_hybrid_deepwalk",
            estimator_kind="logreg",
            numeric_columns=profile_columns + graph_columns + deepwalk_columns,
            use_tfidf=True,
            graph_encoder="deepwalk",
            text_encoder="tfidf",
        ),
        ExperimentSpec(
            name="graph_node2vec_rf",
            family="enhanced_graph_node2vec",
            estimator_kind="rf",
            numeric_columns=profile_columns + graph_columns + node2vec_columns,
            graph_encoder="node2vec",
        ),
        ExperimentSpec(
            name="full_node2vec_logreg",
            family="enhanced_hybrid_node2vec",
            estimator_kind="logreg",
            numeric_columns=profile_columns + graph_columns + node2vec_columns,
            use_tfidf=True,
            graph_encoder="node2vec",
            text_encoder="tfidf",
        ),
    ]
    if transformer_columns:
        experiment_specs.extend(
            [
                ExperimentSpec(
                    name="transformer_profile_logreg",
                    family="enhanced_text_transformer",
                    estimator_kind="logreg",
                    numeric_columns=profile_columns + transformer_columns,
                    use_transformer=True,
                    text_encoder="transformer",
                ),
                ExperimentSpec(
                    name="transformer_graph_logreg",
                    family="enhanced_text_graph",
                    estimator_kind="logreg",
                    numeric_columns=profile_columns + graph_columns + node2vec_columns + transformer_columns,
                    use_transformer=True,
                    graph_encoder="node2vec",
                    text_encoder="transformer",
                ),
            ]
        )

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    best_experiment = ""
    best_val_f1 = -1.0
    best_artifact: dict[str, Any] = {}

    train_df = users[users["split"] == "train"].copy()
    val_df = users[users["split"] == "val"].copy()
    test_df = users[users["split"] == "test"].copy()
    all_df = users.copy()

    for spec in experiment_specs:
        LOGGER.info("Running experiment: %s", spec.name)
        output = run_sklearn_experiment(config, spec, train_df, val_df, test_df, all_df, manifest)
        metrics_rows.extend(output["metrics_rows"])
        prediction_frames.append(output["predictions"])

        if output["best_val_f1"] > best_val_f1:
            best_val_f1 = float(output["best_val_f1"])
            best_experiment = spec.name
            best_artifact = artifact_metadata(output["artifact"])
            joblib.dump(output["artifact"], config.models_dir / "best_model.joblib")
            if output["artifact"].get("model_object") is not None:
                joblib.dump(output["artifact"], config.models_dir / "best_classifier.joblib")

    if transformer_columns:
        gnn_feature_columns = profile_columns + graph_columns + node2vec_columns + transformer_columns
        for name, family, model_type in (
            ("gcn_transformer", "gnn_gcn", "gcn"),
            ("botrgcn_transformer", "gnn_botrgcn", "botrgcn"),
        ):
            LOGGER.info("Running experiment: %s", name)
            gnn_output = run_gnn_experiment(
                config=config,
                name=name,
                family=family,
                users=all_df,
                feature_frame=all_df[gnn_feature_columns].astype(np.float32),
                graph_edges=graph_edges,
                model_type=model_type,
            )
            metrics_rows.extend(gnn_output.metrics_rows)
            prediction_frames.append(gnn_output.predictions)
            if gnn_output.best_val_f1 > best_val_f1:
                best_val_f1 = float(gnn_output.best_val_f1)
                best_experiment = name
                best_artifact = {
                    "experiment": name,
                    "family": family,
                    "model_type": model_type,
                    "graph_encoder": model_type,
                    "text_encoder": "transformer",
                    "artifact_path": gnn_output.artifact_path,
                }
                joblib.dump(best_artifact, config.models_dir / "best_model.joblib")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(["split", "f1", "auc_roc"], ascending=[True, False, False]).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    comparison_df = build_experiment_comparison(metrics_df)

    metrics_df.to_csv(config.tables_dir / "classification_metrics.csv", index=False)
    predictions_df.to_csv(config.tables_dir / "classification_predictions.csv", index=False)
    comparison_df.to_csv(config.tables_dir / "classification_comparison.csv", index=False)

    with (config.models_dir / "best_experiment.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_experiment": best_experiment,
                "best_val_f1": best_val_f1,
                "artifact": best_artifact,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    return ExperimentOutputs(metrics=metrics_df, predictions=predictions_df, best_experiment=best_experiment)


def run_sklearn_experiment(
    config: PipelineConfig,
    spec: ExperimentSpec,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_df: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    matrices = build_feature_matrices(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        all_df=all_df,
        numeric_columns=spec.numeric_columns,
        use_text=spec.use_tfidf,
        text_column=manifest["text_column"],
        tfidf_max_features=config.tfidf_max_features,
        tfidf_min_df=config.tfidf_min_df,
    )

    if spec.estimator_kind == "logreg":
        model = LogisticRegression(
            max_iter=1200,
            solver="saga",
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.random_state,
        )
        X_train = matrices["train_sparse"]
        X_val = matrices["val_sparse"]
        X_test = matrices["test_sparse"]
        X_all = matrices["all_sparse"]
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=config.random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        X_train = matrices["train_dense"]
        X_val = matrices["val_dense"]
        X_test = matrices["test_dense"]
        X_all = matrices["all_dense"]

    y_train = train_df["label_id"].to_numpy(dtype=int)
    y_val = val_df["label_id"].to_numpy(dtype=int)
    y_test = test_df["label_id"].to_numpy(dtype=int)

    model.fit(X_train, y_train)

    pred_val, proba_val = predict_with_scores(model, X_val)
    pred_test, proba_test = predict_with_scores(model, X_test)
    pred_all, proba_all = predict_with_scores(model, X_all)

    val_metrics = evaluate_predictions(y_val, pred_val, proba_val)
    test_metrics = evaluate_predictions(y_test, pred_test, proba_test)

    metrics_rows = [
        {
            "experiment": spec.name,
            "family": spec.family,
            "model_type": spec.estimator_kind,
            "text_encoder": spec.text_encoder,
            "graph_encoder": spec.graph_encoder,
            "split": "val",
            **val_metrics,
        },
        {
            "experiment": spec.name,
            "family": spec.family,
            "model_type": spec.estimator_kind,
            "text_encoder": spec.text_encoder,
            "graph_encoder": spec.graph_encoder,
            "split": "test",
            **test_metrics,
        },
    ]

    predictions = all_df[["user_id", "split", "label", "label_id"]].copy()
    predictions["experiment"] = spec.name
    predictions["family"] = spec.family
    predictions["model_type"] = spec.estimator_kind
    predictions["text_encoder"] = spec.text_encoder
    predictions["graph_encoder"] = spec.graph_encoder
    predictions["prediction"] = pred_all
    predictions["bot_probability"] = proba_all

    artifact = {
        "experiment": spec.name,
        "family": spec.family,
        "model_type": spec.estimator_kind,
        "numeric_columns": spec.numeric_columns,
        "use_text": spec.use_tfidf,
        "text_encoder": spec.text_encoder,
        "graph_encoder": spec.graph_encoder,
        "vectorizer": matrices["vectorizer"],
        "scaler": matrices["scaler"],
        "model_object": model,
    }
    return {
        "metrics_rows": metrics_rows,
        "predictions": predictions,
        "artifact": artifact,
        "best_val_f1": float(val_metrics["f1"]),
    }


def build_experiment_comparison(metrics_df: pd.DataFrame) -> pd.DataFrame:
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    if test_df.empty:
        return pd.DataFrame()

    baseline_best = test_df[test_df["family"].str.startswith("baseline")]["f1"].max()
    baseline_best = float(baseline_best) if pd.notna(baseline_best) else 0.0
    best_idx = test_df["f1"].idxmax()
    best_row = test_df.loc[best_idx]

    comparison = test_df.sort_values(["f1", "auc_roc"], ascending=False).reset_index(drop=True)
    comparison["rank"] = np.arange(1, len(comparison) + 1)
    comparison["delta_vs_best_baseline_f1"] = comparison["f1"] - baseline_best
    comparison["delta_vs_top_model_f1"] = comparison["f1"] - float(best_row["f1"])
    return comparison


def artifact_metadata(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": artifact.get("experiment"),
        "family": artifact.get("family"),
        "model_type": artifact.get("model_type"),
        "text_encoder": artifact.get("text_encoder"),
        "graph_encoder": artifact.get("graph_encoder"),
        "numeric_columns": artifact.get("numeric_columns", []),
        "use_text": bool(artifact.get("use_text", False)),
    }


def run_group_detection(
    config: PipelineConfig,
    method: str = "dbscan",
    split: str = "test",
    threshold: float = 0.8,
    use_ground_truth: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    users, manifest, embeddings = load_cached_artifacts(config)
    users = users.merge(embeddings, on="user_id", how="left").fillna(0.0)

    predictions = pd.read_csv(config.tables_dir / "classification_predictions.csv")
    best_info = json.loads((config.models_dir / "best_experiment.json").read_text(encoding="utf-8"))
    predictions = predictions[predictions["experiment"] == best_info["best_experiment"]]

    split_df = users[users["split"] == split].copy()
    split_predictions = predictions[predictions["split"] == split].copy()
    split_df = split_df.merge(
        split_predictions[["user_id", "bot_probability", "prediction"]],
        on="user_id",
        how="left",
    )

    if use_ground_truth:
        candidate_df = split_df[split_df["label_id"] == 1].copy()
    else:
        candidate_df = split_df[split_df["bot_probability"].fillna(0.0) >= threshold].copy()
        if len(candidate_df) < 10:
            candidate_df = split_df.nlargest(min(200, len(split_df)), "bot_probability").copy()

    if candidate_df.empty:
        raise ValueError("No candidate nodes available for clustering.")

    dense_columns = manifest["profile_numeric_columns"] + manifest.get("graph_numeric_columns", []) + manifest.get(
        "embedding_columns",
        [column for column in embeddings.columns if column != "user_id"],
    )
    candidate_matrix = candidate_df[dense_columns].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    candidate_matrix = scaler.fit_transform(candidate_matrix)

    pca_components = int(min(16, candidate_matrix.shape[0] - 1, candidate_matrix.shape[1]))
    if pca_components > 1:
        reducer = PCA(n_components=pca_components, random_state=config.random_state)
        reduced = reducer.fit_transform(candidate_matrix)
    else:
        reduced = candidate_matrix

    if method == "spectral":
        from sklearn.cluster import SpectralClustering

        n_clusters = max(2, min(8, len(candidate_df) // 20))
        cluster_labels = SpectralClustering(
            n_clusters=n_clusters,
            random_state=config.random_state,
            affinity="nearest_neighbors",
            assign_labels="kmeans",
        ).fit_predict(reduced)
    else:
        from sklearn.cluster import DBSCAN

        cluster_labels = DBSCAN(eps=1.6, min_samples=5).fit_predict(reduced)

    candidate_df["cluster_id"] = cluster_labels

    graph_edges = pd.read_csv(config.cache_dir / "graph_edges.csv")
    graph = nx_from_edges(graph_edges)

    summary_rows = []
    valid_mask = candidate_df["cluster_id"] != -1
    valid_clusters = sorted(candidate_df.loc[valid_mask, "cluster_id"].unique().tolist())
    for cluster_id in valid_clusters:
        members = candidate_df[candidate_df["cluster_id"] == cluster_id].copy()
        member_ids = members["user_id"].tolist()
        subgraph = graph.subgraph(member_ids).copy()
        bot_ratio = float(members["label_id"].mean()) if len(members) else 0.0
        density = float(nx_density_safe(subgraph))
        summary_rows.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(len(members)),
                "bot_ratio": bot_ratio,
                "human_ratio": 1.0 - bot_ratio,
                "density": density,
                "average_probability": float(members["bot_probability"].fillna(0.0).mean()),
            }
        )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(["bot_ratio", "density", "size"], ascending=False)
    else:
        summary_df = pd.DataFrame(columns=["cluster_id", "size", "bot_ratio", "human_ratio", "density", "average_probability"])
    purity = cluster_purity(candidate_df)
    nmi = normalized_mutual_info_score(
        candidate_df.loc[valid_mask, "label_id"],
        candidate_df.loc[valid_mask, "cluster_id"],
    ) if valid_mask.any() else 0.0

    metrics = {
        "method": method,
        "split": split,
        "candidate_count": int(len(candidate_df)),
        "cluster_count": int(len(valid_clusters)),
        "purity": float(purity),
        "nmi": float(nmi),
        "noise_ratio": float((candidate_df["cluster_id"] == -1).mean()),
    }

    candidate_df.to_csv(config.tables_dir / "cluster_assignments.csv", index=False)
    summary_df.to_csv(config.tables_dir / "cluster_summary.csv", index=False)
    with (config.tables_dir / "cluster_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    return candidate_df, summary_df


def build_feature_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_df: pd.DataFrame,
    numeric_columns: list[str],
    use_text: bool,
    text_column: str,
    tfidf_max_features: int,
    tfidf_min_df: int,
) -> dict[str, Any]:
    scaler = StandardScaler()
    train_numeric = scaler.fit_transform(train_df[numeric_columns].to_numpy(dtype=np.float32))
    val_numeric = scaler.transform(val_df[numeric_columns].to_numpy(dtype=np.float32))
    test_numeric = scaler.transform(test_df[numeric_columns].to_numpy(dtype=np.float32))
    all_numeric = scaler.transform(all_df[numeric_columns].to_numpy(dtype=np.float32))

    train_dense = train_numeric.astype(np.float32)
    val_dense = val_numeric.astype(np.float32)
    test_dense = test_numeric.astype(np.float32)
    all_dense = all_numeric.astype(np.float32)

    vectorizer = None
    train_sparse = sparse.csr_matrix(train_dense)
    val_sparse = sparse.csr_matrix(val_dense)
    test_sparse = sparse.csr_matrix(test_dense)
    all_sparse = sparse.csr_matrix(all_dense)

    if use_text:
        train_text = train_df[text_column].fillna("").astype(str)
        val_text = val_df[text_column].fillna("").astype(str)
        test_text = test_df[text_column].fillna("").astype(str)
        all_text = all_df[text_column].fillna("").astype(str)
        effective_min_df = min(tfidf_min_df, max(1, len(train_text)))
        vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            min_df=effective_min_df,
            ngram_range=(1, 2),
            stop_words="english",
        )
        try:
            X_train_text = vectorizer.fit_transform(train_text)
        except ValueError:
            vectorizer = TfidfVectorizer(
                max_features=tfidf_max_features,
                min_df=1,
                ngram_range=(1, 2),
                stop_words=None,
            )
            X_train_text = vectorizer.fit_transform(train_text)
        X_val_text = vectorizer.transform(val_text)
        X_test_text = vectorizer.transform(test_text)
        X_all_text = vectorizer.transform(all_text)

        train_sparse = sparse.hstack([train_sparse, X_train_text], format="csr")
        val_sparse = sparse.hstack([val_sparse, X_val_text], format="csr")
        test_sparse = sparse.hstack([test_sparse, X_test_text], format="csr")
        all_sparse = sparse.hstack([all_sparse, X_all_text], format="csr")

    return {
        "scaler": scaler,
        "vectorizer": vectorizer,
        "train_dense": train_dense,
        "val_dense": val_dense,
        "test_dense": test_dense,
        "all_dense": all_dense,
        "train_sparse": train_sparse,
        "val_sparse": val_sparse,
        "test_sparse": test_sparse,
        "all_sparse": all_sparse,
    }


def load_cached_artifacts(config: PipelineConfig) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    users = pd.read_csv(config.cache_dir / "users.csv")
    with (config.cache_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for column in manifest.get("profile_numeric_columns", []):
        if column not in users.columns:
            users[column] = 0.0
    for column in manifest.get("graph_numeric_columns", []):
        if column not in users.columns:
            users[column] = 0.0

    embedding_path = config.cache_dir / "graph_embeddings.joblib"
    if not embedding_path.exists():
        embedding_path = config.cache_dir / "deepwalk_embeddings.joblib"
    embedding_payload = joblib.load(embedding_path)
    embeddings = pd.DataFrame(embedding_payload["embeddings"], columns=embedding_payload["columns"])
    embeddings.insert(0, "user_id", embedding_payload["user_id"])
    return users, manifest, embeddings


def predict_with_scores(model: Any, features: Any) -> tuple[np.ndarray, np.ndarray]:
    predictions = model.predict(features)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(features)
        probabilities = 1.0 / (1.0 + np.exp(-decision))
    else:
        probabilities = predictions.astype(np.float32)
    return predictions.astype(int), probabilities.astype(np.float32)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0,
    }
    return metrics


def cluster_purity(candidate_df: pd.DataFrame) -> float:
    clustered = candidate_df[candidate_df["cluster_id"] != -1]
    if clustered.empty:
        return 0.0
    purity_numerator = 0
    for _, group in clustered.groupby("cluster_id"):
        purity_numerator += int(group["label_id"].value_counts().max())
    return float(purity_numerator / len(clustered))


def nx_from_edges(edge_df: pd.DataFrame):
    import networkx as nx

    graph = nx.Graph()
    for row in edge_df.itertuples(index=False):
        graph.add_edge(row.source_id, row.target_id, weight=float(row.weight))
    return graph


def nx_density_safe(graph) -> float:
    import networkx as nx

    if graph.number_of_nodes() <= 1:
        return 0.0
    return float(nx.density(graph))
