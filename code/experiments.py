from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from typing import Any

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .config import ProjectConfig, safe_slug
from .data import PreparedDataset, load_prepared_dataset
from .graph_models import run_graph_neural_models

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    family: str
    estimator: str
    numeric_columns: list[str]
    text_mode: str = "none"
    text_column: str = "combined_text"


def run_experiments(config: ProjectConfig) -> dict[str, Any]:
    config.ensure_directories()
    prepared = load_prepared_dataset(config)
    users = prepared.users.copy()
    labeled = users[users["label_id"] >= 0].copy()
    manifest = prepared.manifest

    feature_columns = manifest["feature_numeric_columns"] + manifest["feature_categorical_columns"]
    graph_columns = manifest["graph_structural_columns"]

    node2vec_frame = pd.DataFrame({"user_id": users["user_id"]})
    node2vec_columns: list[str] = []
    if config.run_node2vec:
        node2vec_frame = compute_node2vec_embeddings(config, prepared)
        node2vec_columns = [column for column in node2vec_frame.columns if column != "user_id"]

    transformer_frame = None
    transformer_columns: list[str] = []
    if config.use_transformer:
        transformer_frame = compute_transformer_embeddings(
            config=config,
            text_df=labeled[["user_id", manifest["combined_text_column"]]].rename(columns={manifest["combined_text_column"]: "text"}),
            cache_name="combined_transformer_embeddings",
            prefix="combined_text_emb_",
        )
        if transformer_frame is not None:
            transformer_columns = [column for column in transformer_frame.columns if column != "user_id"]

    description_dense = compute_dense_text_embeddings(
        config=config,
        text_df=users[["user_id", manifest["description_text_column"]]].rename(columns={manifest["description_text_column"]: "text"}),
        cache_name="description_dense_embeddings",
        prefix="des_svd_",
    )
    tweet_dense = compute_dense_text_embeddings(
        config=config,
        text_df=users[["user_id", manifest["tweet_text_column"]]].rename(columns={manifest["tweet_text_column"]: "text"}),
        cache_name="tweet_dense_embeddings",
        prefix="tweet_svd_",
    )

    specs = [
        ExperimentSpec(
            name="feature_only_logistic_regression",
            family="feature_only",
            estimator="logreg",
            numeric_columns=feature_columns,
        ),
        ExperimentSpec(
            name="feature_only_random_forest",
            family="feature_only",
            estimator="rf",
            numeric_columns=feature_columns,
        ),
        ExperimentSpec(
            name="text_only_tfidf_logistic_regression",
            family="text_only",
            estimator="logreg",
            numeric_columns=[],
            text_mode="tfidf",
        ),
        ExperimentSpec(
            name="graph_only_structure_random_forest",
            family="graph_only",
            estimator="rf",
            numeric_columns=graph_columns,
        ),
        ExperimentSpec(
            name="graph_only_node2vec_logistic_regression",
            family="graph_only",
            estimator="logreg",
            numeric_columns=node2vec_columns,
        ),
        ExperimentSpec(
            name="feature_text_tfidf_logistic_regression",
            family="feature_text",
            estimator="logreg",
            numeric_columns=feature_columns,
            text_mode="tfidf",
        ),
        ExperimentSpec(
            name="feature_graph_random_forest",
            family="feature_graph",
            estimator="rf",
            numeric_columns=feature_columns + graph_columns,
        ),
        ExperimentSpec(
            name="feature_graph_node2vec_logistic_regression",
            family="feature_graph",
            estimator="logreg",
            numeric_columns=feature_columns + graph_columns + node2vec_columns,
        ),
        ExperimentSpec(
            name="feature_text_graph_tfidf_node2vec_logistic_regression",
            family="feature_text_graph",
            estimator="logreg",
            numeric_columns=feature_columns + graph_columns + node2vec_columns,
            text_mode="tfidf",
        ),
    ]
    if transformer_columns:
        specs.extend(
            [
                ExperimentSpec(
                    name="text_only_transformer_logistic_regression",
                    family="text_only",
                    estimator="logreg",
                    numeric_columns=transformer_columns,
                ),
                ExperimentSpec(
                    name="feature_text_transformer_logistic_regression",
                    family="feature_text",
                    estimator="logreg",
                    numeric_columns=feature_columns + transformer_columns,
                ),
            ]
        )

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    best_experiment = ""
    best_val_f1 = -1.0

    merged_labeled = labeled.merge(node2vec_frame, on="user_id", how="left")
    if transformer_frame is not None:
        merged_labeled = merged_labeled.merge(transformer_frame, on="user_id", how="left")

    for spec in specs:
        LOGGER.info("Running experiment: %s", spec.name)
        result = _run_sklearn_experiment(config, spec, merged_labeled)
        metrics_rows.extend(result["metrics_rows"])
        prediction_frames.append(result["predictions"])
        if result["best_val_f1"] > best_val_f1:
            best_val_f1 = float(result["best_val_f1"])
            best_experiment = spec.name

    if config.run_gnn:
        gnn_users = users.merge(description_dense, on="user_id", how="left").merge(tweet_dense, on="user_id", how="left")
        if transformer_frame is not None:
            gnn_users = gnn_users.merge(transformer_frame, on="user_id", how="left")
        gnn_users = gnn_users.fillna(0.0)

        for output in run_graph_neural_models(
            config=config,
            users=gnn_users,
            graph_edges=prepared.graph_edges,
            description_columns=[column for column in description_dense.columns if column != "user_id"],
            tweet_columns=[column for column in tweet_dense.columns if column != "user_id"],
            num_property_columns=manifest["gnn_num_property_columns"],
            cat_property_columns=manifest["gnn_cat_property_columns"],
        ):
            metrics_rows.extend(output.metrics_rows)
            prediction_frames.append(output.predictions)
            if output.best_val_f1 > best_val_f1:
                best_val_f1 = output.best_val_f1
                best_experiment = output.artifact_path.stem

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(["split", "f1", "auc_roc"], ascending=[True, False, False]).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    family_summary = _build_family_summary(metrics_df)

    metrics_df.to_csv(config.tables_dir / "experiment_metrics.csv", index=False)
    predictions_df.to_csv(config.tables_dir / "experiment_predictions.csv", index=False)
    family_summary.to_csv(config.tables_dir / "family_summary.csv", index=False)
    with (config.models_dir / "best_experiment.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_experiment": best_experiment, "best_val_f1": best_val_f1}, handle, ensure_ascii=False, indent=2)
    return {
        "metrics": metrics_df,
        "predictions": predictions_df,
        "family_summary": family_summary,
        "best_experiment": best_experiment,
    }


def compute_node2vec_embeddings(config: ProjectConfig, prepared: PreparedDataset) -> pd.DataFrame:
    cache_path = config.cache_dir / "node2vec_embeddings.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path, low_memory=False)
        if _is_valid_embedding_cache(cached, prepared.users["user_id"], prefix="n2v_", expected_dim=config.node2vec_dimensions):
            return cached
        LOGGER.warning("Existing Node2Vec cache does not match the current prepared dataset. Recomputing embeddings.")

    try:
        from gensim.models import Word2Vec
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Skipping Node2Vec because gensim is unavailable: %s", exc)
        return pd.DataFrame({"user_id": prepared.users["user_id"]})

    graph = nx.Graph()
    graph.add_nodes_from(prepared.users["user_id"].tolist())
    for row in prepared.graph_edges.itertuples(index=False):
        weight = 1.0
        if graph.has_edge(row.source_id, row.target_id):
            graph[row.source_id][row.target_id]["weight"] += weight
        else:
            graph.add_edge(row.source_id, row.target_id, weight=weight)

    corpus = Node2VecCorpus(
        graph=graph,
        walk_length=config.node2vec_walk_length,
        num_walks=config.node2vec_num_walks,
        seed=config.random_state,
        return_p=config.node2vec_return_p,
        inout_q=config.node2vec_inout_q,
    )
    model = Word2Vec(
        vector_size=config.node2vec_dimensions,
        window=config.node2vec_window,
        min_count=0,
        sg=1,
        workers=max(1, config.node2vec_workers),
        epochs=config.node2vec_epochs,
        seed=config.random_state,
    )
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

    rows = []
    for user_id in prepared.users["user_id"].tolist():
        if user_id in model.wv:
            embedding = model.wv[user_id]
        else:
            embedding = np.zeros(config.node2vec_dimensions, dtype=np.float32)
        row = {"user_id": user_id}
        for index, value in enumerate(embedding.tolist()):
            row[f"n2v_{index}"] = float(value)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(cache_path, index=False)
    return frame


def compute_transformer_embeddings(
    config: ProjectConfig,
    text_df: pd.DataFrame,
    cache_name: str,
    prefix: str,
) -> pd.DataFrame | None:
    cache_path = config.cache_dir / f"{cache_name}_{safe_slug(config.transformer_model_name)}.joblib"
    if cache_path.exists():
        cached = joblib.load(cache_path)
        if isinstance(cached, pd.DataFrame) and _is_valid_embedding_cache(cached, text_df["user_id"], prefix=prefix):
            return cached
        LOGGER.warning("Existing transformer cache does not match the current prepared dataset. Recomputing embeddings.")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("SentenceTransformer is unavailable: %s", exc)
        return None

    try:
        model = SentenceTransformer(config.transformer_model_name)
        embeddings = model.encode(
            text_df["text"].fillna("").tolist(),
            batch_size=config.transformer_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Skipping transformer experiment because the model could not be loaded: %s", exc)
        return None

    frame = pd.DataFrame(embeddings, columns=[f"{prefix}{index}" for index in range(embeddings.shape[1])])
    frame.insert(0, "user_id", text_df["user_id"].to_numpy())
    joblib.dump(frame, cache_path)
    return frame


def compute_dense_text_embeddings(
    config: ProjectConfig,
    text_df: pd.DataFrame,
    cache_name: str,
    prefix: str,
) -> pd.DataFrame:
    cache_path = config.cache_dir / f"{cache_name}.joblib"
    if cache_path.exists():
        cached = joblib.load(cache_path)
        if isinstance(cached, pd.DataFrame) and _is_valid_embedding_cache(cached, text_df["user_id"], prefix=prefix):
            return cached
        LOGGER.warning("Existing dense text cache does not match the current prepared dataset. Recomputing embeddings.")

    vectorizer = TfidfVectorizer(
        max_features=config.dense_text_max_features,
        min_df=2 if len(text_df) > 100 else 1,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(text_df["text"].fillna(""))
    if matrix.shape[1] == 0:
        dense = np.zeros((len(text_df), 1), dtype=np.float32)
    else:
        n_components = min(config.dense_text_svd_dim, max(1, matrix.shape[1] - 1))
        if n_components <= 1:
            dense = matrix[:, :1].toarray().astype(np.float32)
        else:
            dense = TruncatedSVD(n_components=n_components, random_state=config.random_state).fit_transform(matrix).astype(np.float32)

    frame = pd.DataFrame(dense, columns=[f"{prefix}{index}" for index in range(dense.shape[1])])
    frame.insert(0, "user_id", text_df["user_id"].to_numpy())
    joblib.dump(frame, cache_path)
    return frame


def _run_sklearn_experiment(config: ProjectConfig, spec: ExperimentSpec, dataset: pd.DataFrame) -> dict[str, Any]:
    train_df = dataset[dataset["split"] == "train"].copy()
    val_df = dataset[dataset["split"] == "val"].copy()
    test_df = dataset[dataset["split"] == "test"].copy()

    train_numeric_frame = _prepare_numeric_frame(train_df, spec.numeric_columns)
    val_numeric_frame = _prepare_numeric_frame(val_df, spec.numeric_columns)
    test_numeric_frame = _prepare_numeric_frame(test_df, spec.numeric_columns)

    scaler = None
    if spec.numeric_columns:
        scaler = StandardScaler()
        train_numeric = scaler.fit_transform(train_numeric_frame)
        val_numeric = scaler.transform(val_numeric_frame)
        test_numeric = scaler.transform(test_numeric_frame)
    else:
        train_numeric = np.zeros((len(train_df), 0), dtype=np.float32)
        val_numeric = np.zeros((len(val_df), 0), dtype=np.float32)
        test_numeric = np.zeros((len(test_df), 0), dtype=np.float32)

    vectorizer = None
    if spec.text_mode == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features=config.tfidf_max_features,
            min_df=config.tfidf_min_df,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        train_text = vectorizer.fit_transform(train_df[spec.text_column].fillna(""))
        val_text = vectorizer.transform(val_df[spec.text_column].fillna(""))
        test_text = vectorizer.transform(test_df[spec.text_column].fillna(""))
        numeric_train_sparse = sparse.csr_matrix(train_numeric)
        numeric_val_sparse = sparse.csr_matrix(val_numeric)
        numeric_test_sparse = sparse.csr_matrix(test_numeric)
        train_matrix = sparse.hstack([numeric_train_sparse, train_text], format="csr")
        val_matrix = sparse.hstack([numeric_val_sparse, val_text], format="csr")
        test_matrix = sparse.hstack([numeric_test_sparse, test_text], format="csr")
    else:
        train_matrix = train_numeric
        val_matrix = val_numeric
        test_matrix = test_numeric

    if spec.estimator == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=config.random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    else:
        model = LogisticRegression(
            max_iter=4000,
            solver="saga",
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.random_state,
        )

    y_train = train_df["label_id"].to_numpy(dtype=int)
    y_val = val_df["label_id"].to_numpy(dtype=int)
    y_test = test_df["label_id"].to_numpy(dtype=int)
    model.fit(train_matrix, y_train)

    val_pred, val_prob = _predict_with_probabilities(model, val_matrix)
    test_pred, test_prob = _predict_with_probabilities(model, test_matrix)

    metrics_rows = [
        {"experiment": spec.name, "family": spec.family, "split": "val", **_compute_metrics(y_val, val_pred, val_prob)},
        {"experiment": spec.name, "family": spec.family, "split": "test", **_compute_metrics(y_test, test_pred, test_prob)},
    ]

    predictions = pd.concat(
        [
            _prediction_frame(spec, "val", val_df["user_id"], y_val, val_pred, val_prob),
            _prediction_frame(spec, "test", test_df["user_id"], y_test, test_pred, test_prob),
        ],
        ignore_index=True,
    )

    artifact = {
        "name": spec.name,
        "family": spec.family,
        "model": model,
        "scaler": scaler,
        "vectorizer": vectorizer,
        "numeric_columns": spec.numeric_columns,
        "text_mode": spec.text_mode,
        "text_column": spec.text_column,
    }
    joblib.dump(artifact, config.models_dir / f"{spec.name}.joblib")
    return {"metrics_rows": metrics_rows, "predictions": predictions, "best_val_f1": metrics_rows[0]["f1"]}


def _predict_with_probabilities(model: Any, matrix: Any) -> tuple[np.ndarray, np.ndarray]:
    probabilities = model.predict_proba(matrix)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return predictions, probabilities


def _prediction_frame(
    spec: ExperimentSpec,
    split_name: str,
    user_ids: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment": spec.name,
            "family": spec.family,
            "split": split_name,
            "user_id": user_ids.to_numpy(),
            "true_label": y_true,
            "pred_label": y_pred,
            "bot_probability": probabilities,
        }
    )


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc_roc = roc_auc_score(y_true, probabilities)
    except ValueError:
        auc_roc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(auc_roc),
    }


def _build_family_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(columns=["family", "split", "best_experiment", "f1", "auc_roc"])
    best_rows = []
    for (family, split_name), group in metrics_df.groupby(["family", "split"], sort=False):
        best = group.sort_values(["f1", "auc_roc"], ascending=False).iloc[0]
        best_rows.append(
            {
                "family": family,
                "split": split_name,
                "best_experiment": best["experiment"],
                "accuracy": best["accuracy"],
                "precision": best["precision"],
                "recall": best["recall"],
                "f1": best["f1"],
                "auc_roc": best["auc_roc"],
            }
        )
    return pd.DataFrame(best_rows).sort_values(["split", "f1"], ascending=[True, False]).reset_index(drop=True)


class Node2VecCorpus:
    def __init__(
        self,
        graph: nx.Graph,
        walk_length: int,
        num_walks: int,
        seed: int,
        return_p: float,
        inout_q: float,
    ) -> None:
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.seed = seed
        self.return_p = max(return_p, 1e-6)
        self.inout_q = max(inout_q, 1e-6)

    def __iter__(self):
        rng = random.Random(self.seed)
        nodes = list(self.graph.nodes())
        for _ in range(self.num_walks):
            rng.shuffle(nodes)
            for node in nodes:
                yield self._walk(node, rng)

    def _walk(self, start: str, rng: random.Random) -> list[str]:
        walk = [start]
        while len(walk) < self.walk_length:
            current = walk[-1]
            neighbors = list(self.graph.neighbors(current))
            if not neighbors:
                break
            if len(walk) == 1:
                walk.append(rng.choice(neighbors))
                continue
            previous = walk[-2]
            weights = []
            for neighbor in neighbors:
                base_weight = float(self.graph[current][neighbor].get("weight", 1.0))
                if neighbor == previous:
                    bias = 1.0 / self.return_p
                elif self.graph.has_edge(previous, neighbor):
                    bias = 1.0
                else:
                    bias = 1.0 / self.inout_q
                weights.append(base_weight * bias)
            walk.append(_weighted_choice(neighbors, weights, rng))
        return walk


def _weighted_choice(items: list[str], weights: list[float], rng: random.Random) -> str:
    total = sum(weights)
    if total <= 0:
        return rng.choice(items)
    threshold = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if cumulative >= threshold:
            return item
    return items[-1]


def _prepare_numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=frame.index)
    numeric = frame.reindex(columns=columns, fill_value=0.0).copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric.fillna(0.0)


def _is_valid_embedding_cache(
    frame: pd.DataFrame,
    expected_user_ids: pd.Series,
    prefix: str,
    expected_dim: int | None = None,
) -> bool:
    if "user_id" not in frame.columns:
        return False
    embedding_columns = [column for column in frame.columns if column.startswith(prefix)]
    if expected_dim is not None and len(embedding_columns) != expected_dim:
        return False
    expected_ids = set(expected_user_ids.astype(str).tolist())
    cached_ids = set(frame["user_id"].astype(str).tolist())
    if expected_ids != cached_ids:
        return False
    if frame["user_id"].duplicated().any():
        return False
    if embedding_columns and frame[embedding_columns].isna().any().any():
        return False
    return True
