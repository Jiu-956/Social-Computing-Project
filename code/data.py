from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import ProjectConfig

LOGGER = logging.getLogger(__name__)
TWITTER_TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"
REQUIRED_RAW_FILES = ("edge.csv", "node.json", "split.csv", "label.csv")


@dataclass(slots=True)
class PreparedDataset:
    users: pd.DataFrame
    graph_edges: pd.DataFrame
    manifest: dict[str, Any]
    summary: dict[str, Any]


def stream_json_array(path: Path, chunk_size: int = 1_048_576) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        in_array = False
        reached_eof = False

        while True:
            if not reached_eof and len(buffer) < chunk_size:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    reached_eof = True

            if not in_array:
                buffer = buffer.lstrip()
                if not buffer:
                    if reached_eof:
                        return
                    continue
                if buffer[0] != "[":
                    raise ValueError(f"{path} is not a JSON array.")
                in_array = True
                buffer = buffer[1:]

            parsed = False
            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                if buffer[0] == "]":
                    return
                try:
                    record, offset = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                yield record
                buffer = buffer[offset:]
                parsed = True

            # If decoding made no progress, pull another chunk even when the
            # current buffer is already >= chunk_size to avoid a stall loop.
            if not parsed and not reached_eof:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    reached_eof = True

            if reached_eof:
                trailing = buffer.strip()
                if not trailing:
                    return
                if trailing == "]":
                    return
                if trailing.startswith("]"):
                    remainder = trailing[1:].strip()
                    if not remainder:
                        return
                    LOGGER.warning(
                        "Ignoring trailing content after closing JSON array while parsing %s.",
                        path,
                    )
                    return
                LOGGER.warning(
                    "Reached EOF with incomplete JSON content while parsing %s. "
                    "Continuing with parsed records; the source file may be truncated.",
                    path,
                )
                return

            if not parsed and len(buffer) > chunk_size * 8:
                raise ValueError(f"Parser buffer grew too large while reading {path}.")


def prepare_dataset(config: ProjectConfig) -> PreparedDataset:
    config.ensure_directories()
    _validate_raw_data_files(config)

    users = _load_user_splits_and_labels(config)
    graph_user_ids = set(users["user_id"])

    graph_edges, graph_stats, sampled_posts, relation_breakdown = _scan_edges(
        edge_path=config.data_dir / "edge.csv",
        graph_user_ids=graph_user_ids,
        max_tweets_per_user=config.max_tweets_per_user,
    )
    profile_df, tweet_df = _parse_nodes(
        node_path=config.data_dir / "node.json",
        graph_user_ids=graph_user_ids,
        sampled_posts=sampled_posts,
    )

    merged = users.merge(profile_df, on="user_id", how="left").merge(graph_stats, on="user_id", how="left")
    merged = merged.merge(tweet_df, on="user_id", how="left")
    merged = _compute_account_age_bucket(merged)
    for text_column in ("description_text", "tweet_text", "username", "display_name"):
        if text_column in merged.columns:
            merged[text_column] = merged[text_column].fillna("")
    merged["combined_text"] = (merged["description_text"] + " " + merged["tweet_text"]).str.strip()

    numeric_columns = (
        _feature_numeric_columns()
        + _feature_categorical_columns()
        + _graph_structural_columns()
        + _gnn_num_property_columns()
        + _gnn_cat_property_columns()
    )
    numeric_columns = list(dict.fromkeys(numeric_columns))
    for column in numeric_columns:
        if column not in merged.columns:
            merged[column] = 0.0
    merged[numeric_columns] = merged[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    manifest = {
        "id_column": "user_id",
        "label_column": "label_id",
        "split_column": "split",
        "description_text_column": "description_text",
        "tweet_text_column": "tweet_text",
        "combined_text_column": "combined_text",
        "feature_numeric_columns": _feature_numeric_columns(),
        "feature_categorical_columns": _feature_categorical_columns(),
        "graph_structural_columns": _graph_structural_columns(),
        "gnn_num_property_columns": _gnn_num_property_columns(),
        "gnn_cat_property_columns": _gnn_cat_property_columns(),
        "graph_relation_types": ["follow", "friend"],
    }
    summary = {
        "graph_user_count": int(len(merged)),
        "labeled_user_count": int((merged["label_id"] >= 0).sum()),
        "support_user_count": int((merged["split"] == "support").sum()),
        "graph_edge_count": int(len(graph_edges)),
        "relation_breakdown": relation_breakdown,
        "sampled_tweet_count": int(sum(len(value) for value in sampled_posts.values())),
    }

    merged.to_csv(config.cache_dir / "users.csv", index=False)
    graph_edges.to_csv(config.cache_dir / "graph_edges.csv", index=False)
    with (config.cache_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    with (config.cache_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return PreparedDataset(users=merged, graph_edges=graph_edges, manifest=manifest, summary=summary)


def load_prepared_dataset(config: ProjectConfig) -> PreparedDataset:
    users_path = config.cache_dir / "users.csv"
    edges_path = config.cache_dir / "graph_edges.csv"
    manifest_path = config.cache_dir / "manifest.json"
    summary_path = config.cache_dir / "dataset_summary.json"
    if not (users_path.exists() and edges_path.exists() and manifest_path.exists() and summary_path.exists()):
        return prepare_dataset(config)

    users = pd.read_csv(users_path, low_memory=False)
    graph_edges = pd.read_csv(edges_path, low_memory=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if not _is_manifest_compatible(manifest) or not _is_cached_frame_compatible(users, graph_edges):
        LOGGER.info("Cached prepared dataset is from an older schema. Rebuilding cache with the current code layout.")
        return prepare_dataset(config)

    for column in ("description_text", "username", "display_name", "tweet_text", "combined_text", "split", "label"):
        if column in users.columns:
            users[column] = users[column].fillna("")
    return PreparedDataset(users=users, graph_edges=graph_edges, manifest=manifest, summary=summary)


def _load_user_splits_and_labels(config: ProjectConfig) -> pd.DataFrame:
    split_df = pd.read_csv(config.data_dir / "split.csv").rename(columns={"id": "user_id"})
    label_df = pd.read_csv(config.data_dir / "label.csv").rename(columns={"id": "user_id"})
    users = split_df.merge(label_df, on="user_id", how="left")

    if config.max_graph_users is not None and len(users) > config.max_graph_users:
        users = _sample_graph_users(users, config.max_graph_users, config.random_state)

    users["label_id"] = users["label"].map({"human": 0, "bot": 1}).fillna(-1).astype(int)
    users["is_labeled"] = users["label_id"] >= 0
    return users


def _validate_raw_data_files(config: ProjectConfig) -> None:
    missing_files = [file_name for file_name in REQUIRED_RAW_FILES if not (config.data_dir / file_name).exists()]
    if not missing_files:
        return

    required = ", ".join(REQUIRED_RAW_FILES)
    missing = ", ".join(missing_files)
    raise FileNotFoundError(
        "Raw dataset files are missing under "
        f"{config.data_dir}. Missing: {missing}. "
        f"Expected files: {required}. "
        "You can pass --data-dir to either the raw directory itself or its parent directory that contains raw/."
    )


def _sample_graph_users(users: pd.DataFrame, max_graph_users: int, seed: int) -> pd.DataFrame:
    labeled = users[users["label"].notna()].copy()
    support = users[users["label"].isna()].copy()
    if len(labeled) >= max_graph_users:
        return labeled.sample(n=max_graph_users, random_state=seed).copy()
    remaining = max_graph_users - len(labeled)
    if remaining >= len(support):
        return users.copy()
    sampled_support = support.sample(n=remaining, random_state=seed).copy()
    return pd.concat([labeled, sampled_support], ignore_index=True)


def _scan_edges(
    edge_path: Path,
    graph_user_ids: set[str],
    max_tweets_per_user: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]], dict[str, int]]:
    edge_rows: list[tuple[str, str, str]] = []
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    sampled_posts: dict[str, list[str]] = defaultdict(list)
    relation_breakdown: Counter[str] = Counter()

    with edge_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = row["source_id"].strip()
            relation = row["relation"].strip()
            target = row["target_id"].strip()

            if source not in graph_user_ids:
                continue

            if relation in {"follow", "friend"} and target in graph_user_ids:
                edge_rows.append((source, relation, target))
                counters[source][f"{relation}_out_count"] += 1
                counters[target][f"{relation}_in_count"] += 1
                relation_breakdown[relation] += 1
            elif relation == "post":
                counters[source]["post_count"] += 1
                if len(sampled_posts[source]) < max_tweets_per_user:
                    sampled_posts[source].append(target)

    stats_rows = []
    for user_id in graph_user_ids:
        row = counters[user_id]
        follow_in = float(row.get("follow_in_count", 0))
        follow_out = float(row.get("follow_out_count", 0))
        friend_in = float(row.get("friend_in_count", 0))
        friend_out = float(row.get("friend_out_count", 0))
        total_in = follow_in + friend_in
        total_out = follow_out + friend_out
        post_count = float(row.get("post_count", 0))
        stats_rows.append(
            {
                "user_id": user_id,
                "follow_in_count": follow_in,
                "follow_out_count": follow_out,
                "friend_in_count": friend_in,
                "friend_out_count": friend_out,
                "total_in_degree": total_in,
                "total_out_degree": total_out,
                "in_out_ratio": total_in / (total_out + 1.0),
                "friend_share_out": friend_out / (total_out + 1.0),
                "follow_share_out": follow_out / (total_out + 1.0),
                "post_count": post_count,
            }
        )

    graph_edges = pd.DataFrame(edge_rows, columns=["source_id", "relation", "target_id"])
    graph_stats = pd.DataFrame(stats_rows)
    return graph_edges, graph_stats, sampled_posts, dict(relation_breakdown)


def _parse_nodes(
    node_path: Path,
    graph_user_ids: set[str],
    sampled_posts: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    now = datetime.now(timezone.utc)
    profile_rows: dict[str, dict[str, Any]] = {}
    tweet_lookup: dict[str, str] = {}
    target_tweet_ids = {tweet_id for tweet_ids in sampled_posts.values() for tweet_id in tweet_ids}
    remaining_users = set(graph_user_ids)
    remaining_tweets = set(target_tweet_ids)

    for record in stream_json_array(node_path):
        record_id = _clean_text(record.get("id", ""))
        if not record_id:
            continue
        if record_id in remaining_users:
            profile_rows[record_id] = _extract_profile_row(record, now)
            remaining_users.remove(record_id)
        elif record_id in remaining_tweets:
            tweet_lookup[record_id] = _clean_text(record.get("text", ""))
            remaining_tweets.remove(record_id)
        if not remaining_users and not remaining_tweets:
            break

    profile_df = pd.DataFrame(profile_rows.values()) if profile_rows else pd.DataFrame(columns=["user_id"])
    if profile_df.empty:
        profile_df = pd.DataFrame({"user_id": sorted(graph_user_ids)})

    tweet_rows = []
    for user_id, tweet_ids in sampled_posts.items():
        texts = [tweet_lookup[tweet_id] for tweet_id in tweet_ids if tweet_lookup.get(tweet_id)]
        tweet_rows.append(
            {
                "user_id": user_id,
                "tweet_text": " ".join(texts).strip(),
                "sampled_tweet_count": float(len(texts)),
            }
        )
    tweet_df = pd.DataFrame(tweet_rows) if tweet_rows else pd.DataFrame(columns=["user_id", "tweet_text", "sampled_tweet_count"])
    return profile_df, tweet_df


def _extract_profile_row(record: dict[str, Any], now: datetime) -> dict[str, Any]:
    metrics = record.get("public_metrics") or {}
    followers = _safe_float(metrics.get("followers_count"))
    following = _safe_float(metrics.get("following_count"))
    tweet_count = _safe_float(metrics.get("tweet_count"))
    listed = _safe_float(metrics.get("listed_count"))
    description = _clean_text(record.get("description", ""))
    username = _clean_text(record.get("username", ""))
    display_name = _clean_text(record.get("name", ""))
    location = _clean_text(record.get("location", ""))
    url = _clean_text(record.get("url", ""))
    profile_image_url = _clean_text(record.get("profile_image_url", ""))
    created_at = _parse_datetime(record.get("created_at"))
    account_age_days = max((now - created_at).total_seconds() / 86400.0, 0.0) if created_at else 0.0
    default_profile_image = float(("default_profile" in profile_image_url.lower()) or not profile_image_url)

    return {
        "user_id": _clean_text(record.get("id", "")),
        "description_text": description,
        "username": username,
        "display_name": display_name,
        "followers_count": followers,
        "following_count": following,
        "tweet_count": tweet_count,
        "listed_count": listed,
        "account_age_days": account_age_days,
        "username_length": float(len(username)),
        "display_name_length": float(len(display_name)),
        "description_length": float(len(description)),
        "has_location": float(bool(location)),
        "has_url": float(bool(url) and url.lower() != "none"),
        "is_verified": float(_parse_bool(record.get("verified"))),
        "is_protected": float(_parse_bool(record.get("protected"))),
        "default_profile_image": default_profile_image,
        "followers_following_ratio": followers / (following + 1.0),
        "tweets_per_day": tweet_count / (account_age_days + 1.0),
        "listed_per_1k_followers": 1000.0 * listed / (followers + 1.0),
        "log_followers": float(np.log1p(followers)),
        "log_following": float(np.log1p(following)),
        "log_tweet_count": float(np.log1p(tweet_count)),
    }


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "none":
        return ""
    return " ".join(text.split())


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, "", "None"):
        return None
    try:
        return datetime.strptime(str(value).strip(), TWITTER_TIME_FORMAT).astimezone(timezone.utc)
    except ValueError:
        return None


def _feature_numeric_columns() -> list[str]:
    return [
        "followers_count",
        "following_count",
        "tweet_count",
        "listed_count",
        "account_age_days",
        "username_length",
        "display_name_length",
        "description_length",
        "has_location",
        "has_url",
        "followers_following_ratio",
        "tweets_per_day",
        "listed_per_1k_followers",
        "log_followers",
        "log_following",
        "log_tweet_count",
    ]


def _feature_categorical_columns() -> list[str]:
    return ["is_verified", "is_protected", "default_profile_image"]


def _graph_structural_columns() -> list[str]:
    return [
        "follow_in_count",
        "follow_out_count",
        "friend_in_count",
        "friend_out_count",
        "total_in_degree",
        "total_out_degree",
        "in_out_ratio",
        "friend_share_out",
        "follow_share_out",
        "post_count",
        "sampled_tweet_count",
    ]


def _gnn_num_property_columns() -> list[str]:
    return [
        "followers_count",
        "following_count",
        "tweet_count",
        "account_age_days",
        "account_age_bucket",
        "username_length",
    ]


def _gnn_cat_property_columns() -> list[str]:
    return ["is_protected", "is_verified", "default_profile_image"]


def _compute_account_age_bucket(users: pd.DataFrame) -> pd.DataFrame:
    if "account_age_days" not in users.columns:
        users["account_age_bucket"] = 1
        return users
    age = pd.to_numeric(users["account_age_days"], errors="coerce").fillna(0.0)
    q33, q67 = age.quantile([0.33, 0.67]).values
    users["account_age_bucket"] = pd.cut(
        age,
        bins=[-float("inf"), q33, q67, float("inf")],
        labels=[0, 1, 2],
    ).astype(float).fillna(1.0).astype(int)
    return users


def _is_manifest_compatible(manifest: dict[str, Any]) -> bool:
    required_keys = {
        "id_column",
        "label_column",
        "split_column",
        "description_text_column",
        "tweet_text_column",
        "combined_text_column",
        "feature_numeric_columns",
        "feature_categorical_columns",
        "graph_structural_columns",
        "gnn_num_property_columns",
        "gnn_cat_property_columns",
        "graph_relation_types",
    }
    return required_keys.issubset(set(manifest))


def _is_cached_frame_compatible(users: pd.DataFrame, graph_edges: pd.DataFrame) -> bool:
    required_user_columns = {
        "user_id",
        "label_id",
        "split",
        "description_text",
        "tweet_text",
        "combined_text",
        "default_profile_image",
    }
    required_edge_columns = {"source_id", "relation", "target_id"}
    return required_user_columns.issubset(set(users.columns)) and required_edge_columns.issubset(set(graph_edges.columns))
