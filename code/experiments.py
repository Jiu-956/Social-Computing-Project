from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import ProjectConfig
from .data import load_prepared_dataset
from .graph_models import PRIMARY_METHOD_NAME, run_graph_neural_models

LOGGER = logging.getLogger(__name__)


def run_experiments(config: ProjectConfig) -> dict[str, Any]:
    config.ensure_directories()
    prepared = load_prepared_dataset(config)
    users = prepared.users.copy()
    manifest = prepared.manifest

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

    gnn_users = users.merge(description_dense, on="user_id", how="left").merge(tweet_dense, on="user_id", how="left")
    gnn_users = gnn_users.fillna(0.0)

    outputs = run_graph_neural_models(
        config=config,
        users=gnn_users,
        graph_edges=prepared.graph_edges,
        description_columns=[column for column in description_dense.columns if column != "user_id"],
        tweet_columns=[column for column in tweet_dense.columns if column != "user_id"],
        num_property_columns=manifest["gnn_num_property_columns"],
        cat_property_columns=manifest["gnn_cat_property_columns"],
    )

    metrics_rows = [row for output in outputs for row in output.metrics_rows]
    prediction_frames = [output.predictions for output in outputs if not output.predictions.empty]

    metrics_df = pd.DataFrame(metrics_rows)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(["split", "f1", "auc_roc"], ascending=[True, False, False]).reset_index(drop=True)

    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    method_summary = _build_method_summary(metrics_df)
    best_val_f1 = float(method_summary.get("best_val_f1", np.nan))

    metrics_df.to_csv(config.tables_dir / "experiment_metrics.csv", index=False)
    predictions_df.to_csv(config.tables_dir / "experiment_predictions.csv", index=False)
    pd.DataFrame([method_summary]).to_csv(config.tables_dir / "method_summary.csv", index=False)
    with (config.models_dir / "best_experiment.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_experiment": PRIMARY_METHOD_NAME,
                "best_val_f1": best_val_f1,
                "method_summary_path": str(config.tables_dir / "method_summary.csv"),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "metrics": metrics_df,
        "predictions": predictions_df,
        "method_summary": method_summary,
        "best_experiment": PRIMARY_METHOD_NAME,
    }


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


def _build_method_summary(metrics_df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"method": PRIMARY_METHOD_NAME, "best_val_f1": float("nan")}
    if metrics_df.empty:
        return summary

    for split_name in ("val", "test"):
        split_frame = metrics_df[metrics_df["split"] == split_name].copy()
        if split_frame.empty:
            continue
        best_row = split_frame.sort_values(["f1", "auc_roc", "accuracy"], ascending=False).iloc[0]
        summary[f"{split_name}_accuracy"] = float(best_row["accuracy"])
        summary[f"{split_name}_precision"] = float(best_row["precision"])
        summary[f"{split_name}_recall"] = float(best_row["recall"])
        summary[f"{split_name}_f1"] = float(best_row["f1"])
        summary[f"{split_name}_auc_roc"] = float(best_row["auc_roc"])
    summary["best_val_f1"] = float(summary.get("val_f1", float("nan")))
    return summary


def _is_valid_embedding_cache(
    frame: pd.DataFrame,
    expected_user_ids: pd.Series,
    prefix: str,
) -> bool:
    if "user_id" not in frame.columns:
        return False
    embedding_columns = [column for column in frame.columns if column.startswith(prefix)]
    expected_ids = set(expected_user_ids.astype(str).tolist())
    cached_ids = set(frame["user_id"].astype(str).tolist())
    if expected_ids != cached_ids:
        return False
    if frame["user_id"].duplicated().any():
        return False
    if embedding_columns and frame[embedding_columns].isna().any().any():
        return False
    return True
