from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ..config import ProjectConfig
from ..data import load_prepared_dataset
from ..gnn import run_graph_neural_models
from .embeddings import compute_dense_text_embeddings, compute_node2vec_embeddings, compute_transformer_embeddings
from .specs import FEATURE_CAT_COLUMNS, ExperimentSpec, _make_specs

LOGGER = logging.getLogger(__name__)


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

    specs = _make_specs(
        feature_cols=tuple(feature_columns),
        graph_cols=tuple(graph_columns),
        node2vec_cols=tuple(node2vec_columns),
        transformer_cols=tuple(transformer_columns),
    )

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    training_history_frames: list[pd.DataFrame] = []
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
        LOGGER.info(
            "[%s] val_f1=%.4f test_f1=%.4f val_auc=%.4f test_auc=%.4f",
            spec.name,
            result["val_metrics"]["f1"],
            result["test_metrics"]["f1"],
            result["val_metrics"]["auc_roc"],
            result["test_metrics"]["auc_roc"],
        )
        if result["best_val_f1"] > best_val_f1:
            best_val_f1 = float(result["best_val_f1"])
            best_experiment = spec.name

        training_history = result.get("training_history")
        if isinstance(training_history, pd.DataFrame) and not training_history.empty:
            training_history_frames.append(training_history)

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
            if not output.training_history.empty:
                training_history_frames.append(output.training_history)
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
    if training_history_frames:
        pd.concat(training_history_frames, ignore_index=True).to_csv(config.tables_dir / "training_history.csv", index=False)
    with (config.models_dir / "best_experiment.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_experiment": best_experiment, "best_val_f1": best_val_f1}, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Best experiment: %s (val_f1=%.4f)", best_experiment, best_val_f1)
    return {
        "metrics": metrics_df,
        "predictions": predictions_df,
        "family_summary": family_summary,
        "best_experiment": best_experiment,
    }


def _run_sklearn_experiment(config: ProjectConfig, spec: ExperimentSpec, dataset: pd.DataFrame) -> dict[str, Any]:
    train_df = dataset[dataset["split"] == "train"].copy()
    val_df = dataset[dataset["split"] == "val"].copy()
    test_df = dataset[dataset["split"] == "test"].copy()

    train_numeric_frame = _prepare_numeric_frame(train_df, list(spec.numeric_columns))
    val_numeric_frame = _prepare_numeric_frame(val_df, list(spec.numeric_columns))
    test_numeric_frame = _prepare_numeric_frame(test_df, list(spec.numeric_columns))

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
    val_metrics = _compute_metrics(y_val, val_pred, val_prob)
    test_metrics = _compute_metrics(y_test, test_pred, test_prob)

    predictions = pd.concat(
        [
            _prediction_frame(spec, "val", val_df["user_id"], y_val, val_pred, val_prob),
            _prediction_frame(spec, "test", test_df["user_id"], y_test, test_pred, test_prob),
        ],
        ignore_index=True,
    )

    import joblib
    artifact = {
        "name": spec.name,
        "family": spec.family,
        "model": model,
        "scaler": scaler,
        "vectorizer": vectorizer,
        "numeric_columns": list(spec.numeric_columns),
        "text_mode": spec.text_mode,
        "text_column": spec.text_column,
    }
    joblib.dump(artifact, config.models_dir / f"{spec.name}.joblib")
    return {
        "metrics_rows": metrics_rows,
        "predictions": predictions,
        "best_val_f1": val_metrics["f1"],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


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


def _prepare_numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=frame.index)
    numeric = frame.reindex(columns=columns, fill_value=0.0).copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric.fillna(0.0)
