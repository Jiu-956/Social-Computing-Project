from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from .config import ProjectConfig, safe_slug

FAMILY_SOURCES: dict[str, tuple[str, ...]] = {
    "feature_only": ("feature",),
    "text_only": ("text",),
    "graph_only": ("graph",),
    "time_only": ("time",),
    "feature_text": ("feature", "text"),
    "feature_time": ("feature", "time"),
    "feature_graph": ("feature", "graph"),
    "feature_text_graph": ("feature", "text", "graph"),
    "feature_text_graph_time": ("feature", "text", "graph", "time"),
}

SOURCE_LABELS_ZH = {
    "feature": "用户属性",
    "text": "文本语义",
    "graph": "图结构",
    "time": "时间动态",
}

SOURCE_LABELS_EN = {
    "feature": "Profile",
    "text": "Text",
    "graph": "Graph",
    "time": "Temporal",
}

FAMILY_LABELS_ZH = {
    "feature_only": "基于特征",
    "text_only": "基于文本",
    "graph_only": "基于图",
    "time_only": "基于时间动态",
    "feature_text": "基于特征和文本",
    "feature_time": "基于特征和时间",
    "feature_graph": "基于特征和图",
    "feature_text_graph": "基于特征、文本和图",
    "feature_text_graph_time": "基于特征、文本、图和时间",
}

FAMILY_LABELS_EN = {
    "feature_only": "Feature only",
    "text_only": "Text only",
    "graph_only": "Graph only",
    "time_only": "Temporal only",
    "feature_text": "Feature + Text",
    "feature_time": "Feature + Temporal",
    "feature_graph": "Feature + Graph",
    "feature_text_graph": "Feature + Text + Graph",
    "feature_text_graph_time": "Feature + Text + Graph + Temporal",
}

SOURCE_GAIN_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("feature_only", "feature_text", "text"),
    ("feature_only", "feature_graph", "graph"),
    ("feature_only", "feature_time", "time"),
    ("text_only", "feature_text", "feature"),
    ("graph_only", "feature_graph", "feature"),
    ("time_only", "feature_time", "feature"),
    ("feature_text", "feature_text_graph", "graph"),
    ("feature_graph", "feature_text_graph", "text"),
    ("feature_text_graph", "feature_text_graph_time", "time"),
)

SOURCE_ABLATION_COLUMNS = [
    "experiment",
    "family",
    "split",
    "source",
    "source_cn",
    "source_en",
    "baseline_accuracy",
    "ablated_accuracy",
    "accuracy_drop",
    "baseline_precision",
    "ablated_precision",
    "precision_drop",
    "baseline_recall",
    "ablated_recall",
    "recall_drop",
    "baseline_f1",
    "ablated_f1",
    "f1_drop",
    "baseline_auc_roc",
    "ablated_auc_roc",
    "auc_roc_drop",
]

SIGNAL_TABLE_COLUMNS = [
    "modality",
    "feature",
    "signal_name_en",
    "signal_name_zh",
    "score",
    "score_kind",
    "direction_en",
    "direction_zh",
    "importance",
    "ranking_score",
    "human_mean",
    "bot_mean",
]

PROFILE_SIGNAL_LABELS = {
    "is_verified": ("Verified status", "认证状态"),
    "followers_following_ratio": ("Followers / following ratio", "粉丝关注比"),
    "followers_count": ("Followers count", "粉丝数"),
    "listed_count": ("Listed count", "被列入列表次数"),
    "account_age_days": ("Account age (days)", "账号年龄"),
    "default_profile_image": ("Default profile image", "默认头像"),
    "tweets_per_day": ("Tweets per day", "日均发文数"),
    "tweet_count": ("Tweet count", "发文总数"),
    "description_length": ("Description length", "简介长度"),
    "following_count": ("Following count", "关注数"),
    "log_followers": ("Log followers", "对数粉丝数"),
    "log_following": ("Log following", "对数关注数"),
    "log_tweet_count": ("Log tweet count", "对数发文数"),
}

GRAPH_SIGNAL_LABELS = {
    "follow_in_count": ("Follow in-count", "被 follow 次数"),
    "follow_out_count": ("Follow out-count", "主动 follow 次数"),
    "friend_in_count": ("Friend in-count", "被互关次数"),
    "friend_out_count": ("Friend out-count", "互关出边数"),
    "total_in_degree": ("Total in-degree", "总入度"),
    "total_out_degree": ("Total out-degree", "总出度"),
    "in_out_ratio": ("In / out ratio", "入出度比"),
    "friend_share_out": ("Friend share in outgoing edges", "出边中互关占比"),
    "follow_share_out": ("Follow share in outgoing edges", "出边中 follow 占比"),
    "post_count": ("Post count", "节点推文数"),
    "sampled_tweet_count": ("Sampled tweet count", "采样 tweet 数"),
}

TOKEN_EXCLUDE = {
    "rt",
    "http",
    "https",
    "amp",
    "co",
    "com",
    "org",
    "net",
    "www",
    "que",
}


def ensure_information_source_analysis(
    config: ProjectConfig,
    metrics: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    summary_path = config.tables_dir / "source_contribution_summary.csv"
    details_path = config.tables_dir / "source_contribution_details.csv"
    ablation_path = config.tables_dir / "source_ablation.csv"
    input_paths = [
        config.tables_dir / "experiment_metrics.csv",
        config.cache_dir / "users.csv",
        config.cache_dir / "manifest.json",
        Path(__file__),
    ]

    if _analysis_outputs_are_fresh([summary_path, details_path, ablation_path], input_paths):
        return {
            "source_contribution_summary": pd.read_csv(summary_path, low_memory=False),
            "source_contribution_details": pd.read_csv(details_path, low_memory=False),
            "source_ablation": pd.read_csv(ablation_path, low_memory=False),
        }

    source_summary, source_details = compute_source_contribution_tables(metrics)
    source_ablation = compute_best_model_source_ablation(config, metrics)

    source_summary.to_csv(summary_path, index=False)
    source_details.to_csv(details_path, index=False)
    source_ablation.to_csv(ablation_path, index=False)

    return {
        "source_contribution_summary": source_summary,
        "source_contribution_details": source_details,
        "source_ablation": source_ablation,
    }


def ensure_explainability_signal_analysis(
    config: ProjectConfig,
) -> dict[str, pd.DataFrame]:
    profile_path = config.tables_dir / "feature_signals.csv"
    text_path = config.tables_dir / "text_signals.csv"
    graph_path = config.tables_dir / "graph_signals.csv"
    input_paths = [
        config.cache_dir / "users.csv",
        config.cache_dir / "manifest.json",
        config.models_dir / "feature_only_random_forest.joblib",
        config.models_dir / "text_only_tfidf_logistic_regression.joblib",
        config.models_dir / "graph_only_structure_random_forest.joblib",
        Path(__file__),
    ]

    if _analysis_outputs_are_fresh([profile_path, text_path, graph_path], input_paths):
        return {
            "feature_signals": pd.read_csv(profile_path, low_memory=False),
            "text_signals": pd.read_csv(text_path, low_memory=False),
            "graph_signals": pd.read_csv(graph_path, low_memory=False),
        }

    users = pd.read_csv(config.cache_dir / "users.csv", low_memory=False)
    users = users[users["label_id"] >= 0].copy()
    manifest = json.loads((config.cache_dir / "manifest.json").read_text(encoding="utf-8"))

    feature_signals = compute_numeric_signal_table(
        users=users,
        artifact_path=config.models_dir / "feature_only_random_forest.joblib",
        columns=list(manifest.get("feature_numeric_columns", [])) + list(manifest.get("feature_categorical_columns", [])),
        labels=PROFILE_SIGNAL_LABELS,
        modality="feature",
    )
    text_signals = compute_text_signal_table(config)
    graph_signals = compute_numeric_signal_table(
        users=users,
        artifact_path=config.models_dir / "graph_only_structure_random_forest.joblib",
        columns=list(manifest.get("graph_structural_columns", [])),
        labels=GRAPH_SIGNAL_LABELS,
        modality="graph",
    )

    feature_signals.to_csv(profile_path, index=False)
    text_signals.to_csv(text_path, index=False)
    graph_signals.to_csv(graph_path, index=False)

    return {
        "feature_signals": feature_signals,
        "text_signals": text_signals,
        "graph_signals": graph_signals,
    }


def compute_source_contribution_tables(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    family_best = build_family_best_frame(metrics)

    detail_rows: list[dict[str, Any]] = []
    for split_name, split_frame in family_best.groupby("split", sort=False):
        family_rows = {str(row["family"]): row for _, row in split_frame.iterrows()}
        for base_family, augmented_family, added_source in SOURCE_GAIN_COMPARISONS:
            base_row = family_rows.get(base_family)
            augmented_row = family_rows.get(augmented_family)
            if base_row is None or augmented_row is None:
                continue

            detail_rows.append(
                {
                    "split": split_name,
                    "source": added_source,
                    "source_cn": SOURCE_LABELS_ZH[added_source],
                    "source_en": SOURCE_LABELS_EN[added_source],
                    "base_family": base_family,
                    "base_family_cn": FAMILY_LABELS_ZH.get(base_family, base_family),
                    "base_family_en": FAMILY_LABELS_EN.get(base_family, base_family),
                    "base_experiment": base_row["experiment"],
                    "augmented_family": augmented_family,
                    "augmented_family_cn": FAMILY_LABELS_ZH.get(augmented_family, augmented_family),
                    "augmented_family_en": FAMILY_LABELS_EN.get(augmented_family, augmented_family),
                    "augmented_experiment": augmented_row["experiment"],
                    "base_accuracy": float(base_row["accuracy"]),
                    "base_f1": float(base_row["f1"]),
                    "base_auc_roc": float(base_row["auc_roc"]),
                    "augmented_accuracy": float(augmented_row["accuracy"]),
                    "augmented_f1": float(augmented_row["f1"]),
                    "augmented_auc_roc": float(augmented_row["auc_roc"]),
                    "accuracy_gain": float(augmented_row["accuracy"] - base_row["accuracy"]),
                    "f1_gain": float(augmented_row["f1"] - base_row["f1"]),
                    "auc_roc_gain": float(augmented_row["auc_roc"] - base_row["auc_roc"]),
                }
            )

    detail_columns = [
        "split",
        "source",
        "source_cn",
        "source_en",
        "base_family",
        "base_family_cn",
        "base_family_en",
        "base_experiment",
        "augmented_family",
        "augmented_family_cn",
        "augmented_family_en",
        "augmented_experiment",
        "base_accuracy",
        "base_f1",
        "base_auc_roc",
        "augmented_accuracy",
        "augmented_f1",
        "augmented_auc_roc",
        "accuracy_gain",
        "f1_gain",
        "auc_roc_gain",
    ]
    details = pd.DataFrame(detail_rows, columns=detail_columns)
    if details.empty:
        summary = pd.DataFrame(
            columns=[
                "split",
                "source",
                "source_cn",
                "source_en",
                "comparison_count",
                "mean_accuracy_gain",
                "mean_f1_gain",
                "max_f1_gain",
                "min_f1_gain",
                "mean_auc_roc_gain",
            ]
        )
        return summary, details

    summary = (
        details.groupby(["split", "source", "source_cn", "source_en"], sort=False)
        .agg(
            comparison_count=("f1_gain", "size"),
            mean_accuracy_gain=("accuracy_gain", "mean"),
            mean_f1_gain=("f1_gain", "mean"),
            max_f1_gain=("f1_gain", "max"),
            min_f1_gain=("f1_gain", "min"),
            mean_auc_roc_gain=("auc_roc_gain", "mean"),
        )
        .reset_index()
    )
    summary = summary.sort_values(["split", "mean_f1_gain"], ascending=[True, False]).reset_index(drop=True)
    details = details.sort_values(["split", "f1_gain"], ascending=[True, False]).reset_index(drop=True)
    return summary, details


def compute_best_model_source_ablation(config: ProjectConfig, metrics: pd.DataFrame) -> pd.DataFrame:
    experiment_name = _select_ablation_experiment(config, metrics)
    if experiment_name is None:
        return pd.DataFrame(columns=SOURCE_ABLATION_COLUMNS)

    artifact_path = config.models_dir / f"{experiment_name}.joblib"
    if not artifact_path.exists():
        return pd.DataFrame(columns=SOURCE_ABLATION_COLUMNS)

    artifact = joblib.load(artifact_path)
    dataset = _load_artifact_dataset(config, artifact)
    if dataset.empty:
        return pd.DataFrame(columns=SOURCE_ABLATION_COLUMNS)

    manifest = json.loads((config.cache_dir / "manifest.json").read_text(encoding="utf-8"))
    source_positions = _resolve_numeric_source_positions(artifact, manifest)

    rows: list[dict[str, Any]] = []
    for split_name in ("val", "test"):
        split_df = dataset[dataset["split"] == split_name].copy()
        if split_df.empty:
            continue

        baseline_matrix, numeric_matrix, text_matrix = _build_model_matrix_components(artifact, split_df)
        baseline_pred, baseline_prob = _predict_with_probabilities(artifact["model"], baseline_matrix)
        y_true = split_df["label_id"].to_numpy(dtype=int)
        baseline_metrics = _compute_metrics(y_true, baseline_pred, baseline_prob)

        for source in ("feature", "text", "graph", "time"):
            ablated_matrix = _build_ablated_matrix(
                numeric_matrix=numeric_matrix,
                text_matrix=text_matrix,
                source=source,
                numeric_source_positions=source_positions,
            )
            if ablated_matrix is None:
                continue

            ablated_pred, ablated_prob = _predict_with_probabilities(artifact["model"], ablated_matrix)
            ablated_metrics = _compute_metrics(y_true, ablated_pred, ablated_prob)
            rows.append(
                {
                    "experiment": experiment_name,
                    "family": artifact.get("family", ""),
                    "split": split_name,
                    "source": source,
                    "source_cn": SOURCE_LABELS_ZH[source],
                    "source_en": SOURCE_LABELS_EN[source],
                    "baseline_accuracy": baseline_metrics["accuracy"],
                    "ablated_accuracy": ablated_metrics["accuracy"],
                    "accuracy_drop": baseline_metrics["accuracy"] - ablated_metrics["accuracy"],
                    "baseline_precision": baseline_metrics["precision"],
                    "ablated_precision": ablated_metrics["precision"],
                    "precision_drop": baseline_metrics["precision"] - ablated_metrics["precision"],
                    "baseline_recall": baseline_metrics["recall"],
                    "ablated_recall": ablated_metrics["recall"],
                    "recall_drop": baseline_metrics["recall"] - ablated_metrics["recall"],
                    "baseline_f1": baseline_metrics["f1"],
                    "ablated_f1": ablated_metrics["f1"],
                    "f1_drop": baseline_metrics["f1"] - ablated_metrics["f1"],
                    "baseline_auc_roc": baseline_metrics["auc_roc"],
                    "ablated_auc_roc": ablated_metrics["auc_roc"],
                    "auc_roc_drop": baseline_metrics["auc_roc"] - ablated_metrics["auc_roc"],
                }
            )

    return pd.DataFrame(rows, columns=SOURCE_ABLATION_COLUMNS)


def build_family_best_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=["family", "split", "experiment", "accuracy", "f1", "auc_roc"])

    best_rows = []
    for (family, split_name), group in metrics.groupby(["family", "split"], sort=False):
        best = group.sort_values(["f1", "auc_roc", "accuracy"], ascending=False).iloc[0]
        best_rows.append(
            {
                "family": family,
                "split": split_name,
                "experiment": best["experiment"],
                "accuracy": float(best["accuracy"]),
                "f1": float(best["f1"]),
                "auc_roc": float(best["auc_roc"]),
            }
        )
    return pd.DataFrame(best_rows)


def compute_numeric_signal_table(
    users: pd.DataFrame,
    artifact_path: Path,
    columns: list[str],
    labels: dict[str, tuple[str, str]],
    modality: str,
) -> pd.DataFrame:
    if users.empty or not columns:
        return pd.DataFrame(columns=SIGNAL_TABLE_COLUMNS)

    artifact = _load_artifact_if_exists(artifact_path)
    importance_map = _extract_importance_map(artifact, columns)
    rows: list[dict[str, Any]] = []
    human_users = users[users["label_id"] == 0]
    bot_users = users[users["label_id"] == 1]

    for column in columns:
        if column not in users.columns:
            continue
        human_values = pd.to_numeric(human_users[column], errors="coerce").fillna(0.0)
        bot_values = pd.to_numeric(bot_users[column], errors="coerce").fillna(0.0)
        effect_size = _compute_standardized_mean_difference(human_values, bot_values)
        signal_name_en, signal_name_zh = labels.get(column, (column, column))
        importance = importance_map.get(column, 0.0)
        rows.append(
            {
                "modality": modality,
                "feature": column,
                "signal_name_en": signal_name_en,
                "signal_name_zh": signal_name_zh,
                "score": effect_size,
                "score_kind": "standardized_mean_difference",
                "direction_en": "more_bot_like" if effect_size >= 0 else "more_human_like",
                "direction_zh": "更像机器人" if effect_size >= 0 else "更像人类",
                "importance": importance,
                "ranking_score": abs(effect_size) * (1.0 + importance),
                "human_mean": float(human_values.mean()),
                "bot_mean": float(bot_values.mean()),
            }
        )

    frame = pd.DataFrame(rows, columns=SIGNAL_TABLE_COLUMNS)
    if frame.empty:
        return frame
    return frame.sort_values(["ranking_score", "importance", "score"], ascending=[False, False, False]).reset_index(drop=True)


def compute_text_signal_table(config: ProjectConfig) -> pd.DataFrame:
    artifact_path = config.models_dir / "text_only_tfidf_logistic_regression.joblib"
    artifact = _load_artifact_if_exists(artifact_path)
    if artifact is None:
        return pd.DataFrame(columns=SIGNAL_TABLE_COLUMNS)

    model = artifact.get("model")
    vectorizer = artifact.get("vectorizer")
    if model is None or vectorizer is None or not hasattr(model, "coef_"):
        return pd.DataFrame(columns=SIGNAL_TABLE_COLUMNS)

    features = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    rows: list[dict[str, Any]] = []
    for token, coefficient in zip(features, coefficients, strict=False):
        if not _is_interpretable_token(str(token)):
            continue
        value = float(coefficient)
        rows.append(
            {
                "modality": "text",
                "feature": str(token),
                "signal_name_en": str(token),
                "signal_name_zh": str(token),
                "score": value,
                "score_kind": "logistic_coefficient",
                "direction_en": "more_bot_like" if value >= 0 else "more_human_like",
                "direction_zh": "更像机器人" if value >= 0 else "更像人类",
                "importance": abs(value),
                "ranking_score": abs(value),
                "human_mean": np.nan,
                "bot_mean": np.nan,
            }
        )

    frame = pd.DataFrame(rows, columns=SIGNAL_TABLE_COLUMNS)
    if frame.empty:
        return frame
    return frame.sort_values(["ranking_score", "score"], ascending=[False, False]).reset_index(drop=True)


def _analysis_outputs_are_fresh(output_paths: list[Path], input_paths: list[Path]) -> bool:
    if not all(path.exists() for path in output_paths):
        return False
    existing_inputs = [path for path in input_paths if path.exists()]
    if not existing_inputs:
        return False
    newest_input_time = max(path.stat().st_mtime for path in existing_inputs)
    oldest_output_time = min(path.stat().st_mtime for path in output_paths)
    return oldest_output_time >= newest_input_time


def _select_ablation_experiment(config: ProjectConfig, metrics: pd.DataFrame) -> str | None:
    if metrics.empty:
        return None

    candidates = metrics[metrics["split"] == "val"].copy()
    if candidates.empty:
        return None

    candidates["artifact_exists"] = candidates["experiment"].map(
        lambda name: (config.models_dir / f"{name}.joblib").exists()
    )
    candidates["source_count"] = candidates["family"].map(lambda value: len(FAMILY_SOURCES.get(str(value), ())))
    candidates = candidates[(candidates["artifact_exists"]) & (candidates["source_count"] >= 2)].copy()
    if candidates.empty:
        return None

    candidates = candidates.sort_values(
        ["source_count", "f1", "auc_roc", "accuracy"],
        ascending=[False, False, False, False],
    )
    return str(candidates.iloc[0]["experiment"])


def _load_artifact_dataset(config: ProjectConfig, artifact: dict[str, Any]) -> pd.DataFrame:
    users = pd.read_csv(config.cache_dir / "users.csv", low_memory=False)
    dataset = users[users["label_id"] >= 0].copy()
    numeric_columns = list(artifact.get("numeric_columns", []))

    if any(str(column).startswith("n2v_") for column in numeric_columns):
        embedding_path = config.cache_dir / "node2vec_embeddings.csv"
        if embedding_path.exists():
            dataset = dataset.merge(pd.read_csv(embedding_path, low_memory=False), on="user_id", how="left")

    if any(str(column).startswith("combined_text_emb_") for column in numeric_columns):
        transformer_frame = _load_transformer_embeddings(config)
        if transformer_frame is not None:
            dataset = dataset.merge(transformer_frame, on="user_id", how="left")

    return dataset


def _load_transformer_embeddings(config: ProjectConfig) -> pd.DataFrame | None:
    preferred = config.cache_dir / f"combined_transformer_embeddings_{safe_slug(config.transformer_model_name)}.joblib"
    candidates = [preferred] if preferred.exists() else []
    candidates.extend(
        sorted(path for path in config.cache_dir.glob("combined_transformer_embeddings_*.joblib") if path not in candidates)
    )
    for candidate in candidates:
        loaded = joblib.load(candidate)
        if isinstance(loaded, pd.DataFrame) and "user_id" in loaded.columns:
            return loaded
    return None


def _resolve_numeric_source_positions(artifact: dict[str, Any], manifest: dict[str, Any]) -> dict[str, list[int]]:
    feature_columns = set(manifest.get("feature_numeric_columns", [])) | set(manifest.get("feature_categorical_columns", []))
    graph_columns = set(manifest.get("graph_structural_columns", []))
    time_columns = set(manifest.get("time_proxy_columns", []))
    numeric_positions = {"feature": [], "text": [], "graph": [], "time": []}

    for index, column in enumerate(artifact.get("numeric_columns", [])):
        column = str(column)
        if column in feature_columns:
            numeric_positions["feature"].append(index)
        elif column in graph_columns or column.startswith("n2v_"):
            numeric_positions["graph"].append(index)
        elif column in time_columns:
            numeric_positions["time"].append(index)
        elif column.startswith("combined_text_emb_"):
            numeric_positions["text"].append(index)

    return numeric_positions


def _build_model_matrix_components(
    artifact: dict[str, Any],
    split_df: pd.DataFrame,
) -> tuple[Any, np.ndarray, sparse.csr_matrix | None]:
    numeric_columns = list(artifact.get("numeric_columns", []))
    numeric_frame = _prepare_numeric_frame(split_df, numeric_columns)
    numeric_matrix = (
        numeric_frame.to_numpy(dtype=np.float32) if numeric_columns else np.zeros((len(split_df), 0), dtype=np.float32)
    )

    scaler = artifact.get("scaler")
    if scaler is not None and numeric_columns:
        numeric_matrix = scaler.transform(numeric_frame).astype(np.float32)

    text_matrix: sparse.csr_matrix | None = None
    if artifact.get("text_mode") == "tfidf":
        vectorizer = artifact.get("vectorizer")
        text_column = str(artifact.get("text_column", "combined_text"))
        text_matrix = vectorizer.transform(split_df[text_column].fillna(""))
        base_matrix = sparse.hstack([sparse.csr_matrix(numeric_matrix), text_matrix], format="csr")
    else:
        base_matrix = numeric_matrix

    return base_matrix, numeric_matrix, text_matrix


def _build_ablated_matrix(
    numeric_matrix: np.ndarray,
    text_matrix: sparse.csr_matrix | None,
    source: str,
    numeric_source_positions: dict[str, list[int]],
) -> Any | None:
    has_numeric_block = bool(numeric_source_positions.get(source))
    has_text_block = source == "text" and text_matrix is not None and text_matrix.shape[1] > 0
    if not has_numeric_block and not has_text_block:
        return None

    numeric_copy = numeric_matrix.copy()
    positions = numeric_source_positions.get(source, [])
    if positions:
        numeric_copy[:, positions] = 0.0

    if text_matrix is not None:
        ablated_text = text_matrix
        if source == "text":
            ablated_text = sparse.csr_matrix(text_matrix.shape, dtype=text_matrix.dtype)
        return sparse.hstack([sparse.csr_matrix(numeric_copy), ablated_text], format="csr")

    return numeric_copy


def _prepare_numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=frame.index)
    numeric = frame.reindex(columns=columns, fill_value=0.0).copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric.fillna(0.0)


def _predict_with_probabilities(model: Any, matrix: Any) -> tuple[np.ndarray, np.ndarray]:
    probabilities = model.predict_proba(matrix)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return predictions, probabilities


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


def _load_artifact_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = joblib.load(path)
    return loaded if isinstance(loaded, dict) else None


def _extract_importance_map(artifact: dict[str, Any] | None, columns: list[str]) -> dict[str, float]:
    if artifact is None:
        return {}
    model = artifact.get("model")
    artifact_columns = list(artifact.get("numeric_columns", []))
    if model is None or not hasattr(model, "feature_importances_") or len(artifact_columns) != len(model.feature_importances_):
        return {}
    return {
        str(column): float(importance)
        for column, importance in zip(artifact_columns, model.feature_importances_, strict=False)
        if str(column) in columns
    }


def _compute_standardized_mean_difference(human_values: pd.Series, bot_values: pd.Series) -> float:
    pooled_variance = (float(human_values.var()) + float(bot_values.var())) / 2.0
    pooled_std = float(np.sqrt(pooled_variance)) if pooled_variance > 0 else 0.0
    if pooled_std == 0.0:
        return 0.0
    return float((float(bot_values.mean()) - float(human_values.mean())) / pooled_std)


def _is_interpretable_token(token: str) -> bool:
    token = token.strip().lower()
    if not token or len(token) < 3 or len(token) > 40:
        return False
    if any(char.isdigit() for char in token):
        return False
    if not re.fullmatch(r"[a-z][a-z_ ]*", token):
        return False
    parts = [part for part in token.split() if part]
    if not parts:
        return False
    if any(part in TOKEN_EXCLUDE for part in parts):
        return False
    if len(parts) == 1 and parts[0] in ENGLISH_STOP_WORDS:
        return False
    if all(part in ENGLISH_STOP_WORDS for part in parts):
        return False
    return True
