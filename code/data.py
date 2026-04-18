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

            if reached_eof:
                trailing = buffer.strip()
                if trailing and trailing != "]":
                    raise ValueError(f"Unexpected trailing content while parsing {path}.")
                return

            if not parsed and len(buffer) > chunk_size * 8:
                raise ValueError(f"Parser buffer grew too large while reading {path}.")


def prepare_dataset(config: ProjectConfig) -> PreparedDataset:
    config.ensure_directories()

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
        snapshot_count=config.temporal_snapshot_count,
    )

    merged = users.merge(profile_df, on="user_id", how="left").merge(graph_stats, on="user_id", how="left")
    merged = merged.merge(tweet_df, on="user_id", how="left")
    for text_column in ("description_text", "tweet_text", "username", "display_name"):
        if text_column in merged.columns:
            merged[text_column] = merged[text_column].fillna("")
    merged["combined_text"] = (merged["description_text"] + " " + merged["tweet_text"]).str.strip()

    numeric_columns = (
        _feature_numeric_columns()
        + _feature_categorical_columns()
        + _graph_structural_columns()
        + _time_proxy_columns()
        + _temporal_snapshot_columns(config.temporal_snapshot_count)
        + _gnn_num_property_columns()
        + _gnn_cat_property_columns()
        + _gnn_time_columns()
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
        "time_proxy_columns": _time_proxy_columns(),
        "temporal_snapshot_columns": _temporal_snapshot_columns(config.temporal_snapshot_count),
        "temporal_snapshot_count": int(config.temporal_snapshot_count),
        "gnn_num_property_columns": _gnn_num_property_columns(),
        "gnn_cat_property_columns": _gnn_cat_property_columns(),
        "gnn_time_columns": _gnn_time_columns(),
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
    snapshot_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    now = datetime.now(timezone.utc)
    profile_rows: dict[str, dict[str, Any]] = {}
    tweet_lookup: dict[str, dict[str, Any]] = {}
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
            tweet_lookup[record_id] = _extract_tweet_row(record)
            remaining_tweets.remove(record_id)
        if not remaining_users and not remaining_tweets:
            break

    profile_df = pd.DataFrame(profile_rows.values()) if profile_rows else pd.DataFrame(columns=["user_id"])
    if profile_df.empty:
        profile_df = pd.DataFrame({"user_id": sorted(graph_user_ids)})

    tweet_rows = []
    for user_id, tweet_ids in sampled_posts.items():
        tweets = [tweet_lookup[tweet_id] for tweet_id in tweet_ids if tweet_lookup.get(tweet_id)]
        texts = [str(tweet.get("text", "")) for tweet in tweets if str(tweet.get("text", "")).strip()]
        tweet_row = {
            "user_id": user_id,
            "tweet_text": " ".join(texts).strip(),
            "sampled_tweet_count": float(len(tweets)),
        }
        tweet_row.update(_aggregate_tweet_temporal_features(tweets))
        tweet_row.update(_build_snapshot_feature_row(tweets, snapshot_count))
        tweet_rows.append(tweet_row)
    tweet_columns = (
        ["user_id", "tweet_text", "sampled_tweet_count"]
        + _time_proxy_columns()
        + _temporal_snapshot_columns(snapshot_count)
    )
    tweet_df = pd.DataFrame(tweet_rows) if tweet_rows else pd.DataFrame(columns=tweet_columns)
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


def _extract_tweet_row(record: dict[str, Any]) -> dict[str, Any]:
    created_at = _parse_datetime(record.get("created_at"))
    referenced_types = _extract_referenced_types(record.get("referenced_tweets"))
    is_retweet = float("retweeted" in referenced_types)
    is_reply = float("replied_to" in referenced_types)
    is_quote = float("quoted" in referenced_types)
    is_original = float(not referenced_types)

    return {
        "text": _clean_text(record.get("text", "")),
        "timestamp": created_at.timestamp() if created_at else None,
        "day_ordinal": created_at.date().toordinal() if created_at else None,
        "hour": created_at.hour if created_at else None,
        "weekday": created_at.weekday() if created_at else None,
        "is_retweet": is_retweet,
        "is_reply": is_reply,
        "is_quote": is_quote,
        "is_original": is_original,
    }


def _aggregate_tweet_temporal_features(tweets: list[dict[str, Any]]) -> dict[str, float]:
    if not tweets:
        return {column: 0.0 for column in _time_proxy_columns()}

    timestamps = np.array(
        sorted(float(tweet["timestamp"]) for tweet in tweets if tweet.get("timestamp") is not None),
        dtype=np.float64,
    )
    hours = [int(tweet["hour"]) for tweet in tweets if tweet.get("hour") is not None]
    weekdays = [int(tweet["weekday"]) for tweet in tweets if tweet.get("weekday") is not None]
    day_ordinals = [int(tweet["day_ordinal"]) for tweet in tweets if tweet.get("day_ordinal") is not None]

    gaps = np.diff(timestamps) / 3600.0 if len(timestamps) >= 2 else np.zeros(0, dtype=np.float64)
    mean_gap = float(gaps.mean()) if len(gaps) else 0.0
    std_gap = float(gaps.std(ddof=0)) if len(gaps) else 0.0
    gap_cv = std_gap / (mean_gap + 1e-6) if len(gaps) else 0.0
    burstiness = (std_gap - mean_gap) / (std_gap + mean_gap + 1e-6) if len(gaps) else 0.0

    hour_hist = _discrete_histogram(hours, bucket_count=24)
    weekday_hist = _discrete_histogram(weekdays, bucket_count=7)
    gap_hist = _gap_histogram(gaps)

    timestamped_count = float(len(timestamps))
    observed_span_hours = (float(timestamps[-1] - timestamps[0]) / 3600.0) if len(timestamps) >= 2 else 0.0
    unique_days = len(set(day_ordinals))
    observed_days = max(observed_span_hours / 24.0, 0.0)
    active_day_ratio = unique_days / max(observed_days + 1.0, 1.0)

    retweet_hours = [int(tweet["hour"]) for tweet in tweets if tweet.get("is_retweet", 0.0) > 0.0 and tweet.get("hour") is not None]
    original_hours = [int(tweet["hour"]) for tweet in tweets if tweet.get("is_original", 0.0) > 0.0 and tweet.get("hour") is not None]
    retweet_hour_alignment = _hour_distribution_alignment(retweet_hours, original_hours)
    retweet_night_gap = abs(_night_ratio(retweet_hours) - _night_ratio(original_hours))

    total_tweets = max(float(len(tweets)), 1.0)
    retweet_ratio = float(sum(float(tweet.get("is_retweet", 0.0)) for tweet in tweets) / total_tweets)
    reply_ratio = float(sum(float(tweet.get("is_reply", 0.0)) for tweet in tweets) / total_tweets)
    quote_ratio = float(sum(float(tweet.get("is_quote", 0.0)) for tweet in tweets) / total_tweets)
    original_ratio = float(sum(float(tweet.get("is_original", 0.0)) for tweet in tweets) / total_tweets)

    return {
        "temporal_observed_span_hours": observed_span_hours,
        "temporal_mean_gap_hours": mean_gap,
        "temporal_std_gap_hours": std_gap,
        "temporal_gap_cv": gap_cv,
        "temporal_gap_entropy": _normalized_entropy(gap_hist),
        "temporal_burstiness": burstiness,
        "temporal_hour_entropy": _normalized_entropy(hour_hist),
        "temporal_night_activity_ratio": _night_ratio(hours),
        "temporal_weekend_ratio": _weekend_ratio(weekdays),
        "temporal_peak_hour_share": float(hour_hist.max() / timestamped_count) if timestamped_count > 0.0 else 0.0,
        "temporal_circadian_strength": _circular_strength(hours, period=24),
        "temporal_weekly_strength": _circular_strength(weekdays, period=7),
        "temporal_active_day_ratio": float(active_day_ratio),
        "temporal_retweet_ratio": retweet_ratio,
        "temporal_reply_ratio": reply_ratio,
        "temporal_quote_ratio": quote_ratio,
        "temporal_original_ratio": original_ratio,
        "temporal_retweet_original_hour_alignment": retweet_hour_alignment,
        "temporal_retweet_original_night_gap": float(retweet_night_gap),
    }


def _build_snapshot_feature_row(tweets: list[dict[str, Any]], snapshot_count: int) -> dict[str, float]:
    columns = _temporal_snapshot_columns(snapshot_count)
    if not tweets or snapshot_count <= 0:
        return {column: 0.0 for column in columns}

    timed_tweets = [tweet for tweet in tweets if tweet.get("timestamp") is not None]
    if not timed_tweets:
        return {column: 0.0 for column in columns}

    ordered = sorted(timed_tweets, key=lambda item: float(item["timestamp"]))
    start_time = float(ordered[0]["timestamp"])
    end_time = float(ordered[-1]["timestamp"])
    total_count = max(float(len(ordered)), 1.0)
    span = max(end_time - start_time, 1.0)

    bucket_tweets: list[list[dict[str, Any]]] = [[] for _ in range(snapshot_count)]
    for tweet in ordered:
        ratio = (float(tweet["timestamp"]) - start_time) / span
        bucket_index = min(int(ratio * snapshot_count), snapshot_count - 1)
        bucket_tweets[bucket_index].append(tweet)

    features: dict[str, float] = {}
    for snapshot_index, bucket in enumerate(bucket_tweets):
        hours = [int(tweet["hour"]) for tweet in bucket if tweet.get("hour") is not None]
        weekdays = [int(tweet["weekday"]) for tweet in bucket if tweet.get("weekday") is not None]
        bucket_count = max(float(len(bucket)), 1.0)
        values = {
            "activity_share": float(len(bucket) / total_count),
            "retweet_ratio": float(sum(float(tweet.get("is_retweet", 0.0)) for tweet in bucket) / bucket_count),
            "reply_ratio": float(sum(float(tweet.get("is_reply", 0.0)) for tweet in bucket) / bucket_count),
            "quote_ratio": float(sum(float(tweet.get("is_quote", 0.0)) for tweet in bucket) / bucket_count),
            "night_ratio": _night_ratio(hours),
            "weekend_ratio": _weekend_ratio(weekdays),
            "hour_entropy": _normalized_entropy(_discrete_histogram(hours, bucket_count=24)),
        }
        for feature_name, value in values.items():
            features[f"snapshot_{snapshot_index}_{feature_name}"] = float(value)

    for column in columns:
        features.setdefault(column, 0.0)
    return features


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


def _extract_referenced_types(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    referenced_types: set[str] = set()
    for item in value:
        if isinstance(item, dict):
            relation_type = _clean_text(item.get("type", "")).lower()
            if relation_type:
                referenced_types.add(relation_type)
    return referenced_types


def _discrete_histogram(values: list[int], bucket_count: int) -> np.ndarray:
    histogram = np.zeros(bucket_count, dtype=np.float64)
    for value in values:
        if 0 <= int(value) < bucket_count:
            histogram[int(value)] += 1.0
    return histogram


def _gap_histogram(gaps: np.ndarray) -> np.ndarray:
    histogram = np.zeros(8, dtype=np.float64)
    if gaps.size == 0:
        return histogram
    bins = np.array([0.0, 1.0, 3.0, 6.0, 12.0, 24.0, 72.0, 168.0], dtype=np.float64)
    clipped = np.clip(gaps, a_min=0.0, a_max=None)
    positions = np.digitize(clipped, bins=bins, right=False)
    for position in positions:
        histogram[min(int(position), len(histogram) - 1)] += 1.0
    return histogram


def _normalized_entropy(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probabilities = counts / total
    probabilities = probabilities[probabilities > 0.0]
    if len(probabilities) <= 1:
        return 0.0
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / float(np.log(len(counts)))


def _circular_strength(values: list[int], period: int) -> float:
    if not values or period <= 0:
        return 0.0
    angles = (2.0 * np.pi * np.array(values, dtype=np.float64)) / float(period)
    return float(np.abs(np.exp(1j * angles).mean()))


def _night_ratio(hours: list[int]) -> float:
    if not hours:
        return 0.0
    night_count = sum(1 for hour in hours if hour < 6 or hour >= 23)
    return float(night_count / len(hours))


def _weekend_ratio(weekdays: list[int]) -> float:
    if not weekdays:
        return 0.0
    weekend_count = sum(1 for weekday in weekdays if weekday >= 5)
    return float(weekend_count / len(weekdays))


def _hour_distribution_alignment(left_hours: list[int], right_hours: list[int]) -> float:
    if not left_hours or not right_hours:
        return 0.0
    left_hist = _discrete_histogram(left_hours, bucket_count=24)
    right_hist = _discrete_histogram(right_hours, bucket_count=24)
    left_norm = left_hist / max(float(left_hist.sum()), 1.0)
    right_norm = right_hist / max(float(right_hist.sum()), 1.0)
    denominator = float(np.linalg.norm(left_norm) * np.linalg.norm(right_norm))
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(np.dot(left_norm, right_norm) / denominator, 0.0, 1.0))


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


def _time_proxy_columns() -> list[str]:
    return [
        "temporal_observed_span_hours",
        "temporal_mean_gap_hours",
        "temporal_std_gap_hours",
        "temporal_gap_cv",
        "temporal_gap_entropy",
        "temporal_burstiness",
        "temporal_hour_entropy",
        "temporal_night_activity_ratio",
        "temporal_weekend_ratio",
        "temporal_peak_hour_share",
        "temporal_circadian_strength",
        "temporal_weekly_strength",
        "temporal_active_day_ratio",
        "temporal_retweet_ratio",
        "temporal_reply_ratio",
        "temporal_quote_ratio",
        "temporal_original_ratio",
        "temporal_retweet_original_hour_alignment",
        "temporal_retweet_original_night_gap",
    ]


def _snapshot_feature_names() -> list[str]:
    return [
        "activity_share",
        "retweet_ratio",
        "reply_ratio",
        "quote_ratio",
        "night_ratio",
        "weekend_ratio",
        "hour_entropy",
    ]


def _temporal_snapshot_columns(snapshot_count: int) -> list[str]:
    return [
        f"snapshot_{snapshot_index}_{feature_name}"
        for snapshot_index in range(max(int(snapshot_count), 0))
        for feature_name in _snapshot_feature_names()
    ]


def _gnn_num_property_columns() -> list[str]:
    return [
        "followers_count",
        "following_count",
        "tweet_count",
        "account_age_days",
        "username_length",
    ]


def _gnn_cat_property_columns() -> list[str]:
    return ["is_protected", "is_verified", "default_profile_image"]


def _gnn_time_columns() -> list[str]:
    return _time_proxy_columns()


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
        "time_proxy_columns",
        "temporal_snapshot_columns",
        "temporal_snapshot_count",
        "gnn_num_property_columns",
        "gnn_cat_property_columns",
        "gnn_time_columns",
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
        "temporal_mean_gap_hours",
        "temporal_hour_entropy",
        "snapshot_0_activity_share",
    }
    required_edge_columns = {"source_id", "relation", "target_id"}
    return required_user_columns.issubset(set(users.columns)) and required_edge_columns.issubset(set(graph_edges.columns))
