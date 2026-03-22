from __future__ import annotations

import csv
import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)
TWITTER_TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"


@dataclass(slots=True)
class PreparedData:
    users: pd.DataFrame
    graph_edges: pd.DataFrame
    network_summary: dict[str, Any]
    manifest: dict[str, Any]
    graph_user_ids: list[str]


def load_split_table(config: PipelineConfig) -> pd.DataFrame:
    split_df = pd.read_csv(config.data_dir / "split.csv").rename(columns={"id": "user_id"})
    if config.max_labeled_users is not None:
        labels_df = pd.read_csv(config.data_dir / "label.csv").rename(columns={"id": "user_id"})
        labeled = labels_df.merge(split_df, on="user_id", how="inner")
        labeled = _sample_labeled_users(labeled, config.max_labeled_users, config.random_state)
        keep_ids = set(labeled["user_id"])
        split_df = split_df[split_df["user_id"].isin(keep_ids)].copy()
    return split_df


def load_labeled_users(config: PipelineConfig) -> pd.DataFrame:
    labels_df = pd.read_csv(config.data_dir / "label.csv").rename(columns={"id": "user_id"})
    split_df = pd.read_csv(config.data_dir / "split.csv").rename(columns={"id": "user_id"})
    labeled = labels_df.merge(split_df, on="user_id", how="inner")
    if config.max_labeled_users is not None:
        labeled = _sample_labeled_users(labeled, config.max_labeled_users, config.random_state)
    return labeled.set_index("user_id").sort_index()


def _sample_labeled_users(df: pd.DataFrame, max_users: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_users:
        return df.copy()

    rng = random.Random(seed)
    grouped = list(df.groupby(["split", "label"], sort=False))
    total = len(df)
    sampled_frames: list[pd.DataFrame] = []
    used_ids: set[str] = set()
    target_total = 0

    for _, group in grouped:
        quota = max(1, round(len(group) / total * max_users))
        quota = min(quota, len(group))
        sampled = group.sample(n=quota, random_state=rng.randint(0, 10_000_000))
        sampled_frames.append(sampled)
        used_ids.update(sampled["user_id"])
        target_total += quota

    sampled_df = pd.concat(sampled_frames, ignore_index=True)
    if target_total > max_users:
        sampled_df = sampled_df.sample(n=max_users, random_state=seed)
        return sampled_df

    if target_total < max_users:
        remaining = df[~df["user_id"].isin(used_ids)]
        extra = remaining.sample(n=min(max_users - target_total, len(remaining)), random_state=seed)
        sampled_df = pd.concat([sampled_df, extra], ignore_index=True)

    return sampled_df


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
                stripped = buffer.lstrip()
                consumed = len(buffer) - len(stripped)
                buffer = stripped
                if not buffer:
                    if reached_eof:
                        return
                    continue
                if buffer[0] != "[":
                    raise ValueError(f"{path} is not a JSON array.")
                in_array = True
                buffer = buffer[1:]
                if consumed:
                    LOGGER.debug("Skipped %s leading whitespace characters.", consumed)

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
                buffer = buffer.strip()
                if buffer and buffer != "]":
                    raise ValueError(f"Unexpected trailing content in {path}: {buffer[:80]!r}")
                return

            if not parsed and len(buffer) > chunk_size * 8:
                raise ValueError(f"Parser buffer exceeded safe size while reading {path}.")


def prepare_dataset(config: PipelineConfig) -> PreparedData:
    config.ensure_dirs()

    labeled_users = load_labeled_users(config)
    split_table = load_split_table(config)
    graph_user_ids = set(split_table["user_id"])
    labeled_user_ids = set(labeled_users.index)

    LOGGER.info("Loaded %s labeled users and %s graph users.", len(labeled_user_ids), len(graph_user_ids))

    graph_edges_df, edge_stats_df, sampled_tweets, network_summary = collect_edge_information(
        edge_path=config.data_dir / "edge.csv",
        graph_user_ids=graph_user_ids,
        labeled_user_ids=labeled_user_ids,
        max_tweets_per_user=config.max_tweets_per_user,
    )

    profile_df, text_df = parse_nodes(
        node_path=config.data_dir / "node.json",
        labeled_user_ids=labeled_user_ids,
        sampled_tweets=sampled_tweets,
    )

    users = (
        labeled_users.join(profile_df, how="left")
        .join(text_df, how="left")
        .join(edge_stats_df, how="left")
        .fillna(
            {
                "description_text": "",
                "tweet_text": "",
                "screen_name": "",
                "display_name": "",
                "location": "",
                "profile_url": "",
            }
        )
    )

    numeric_fill_defaults = {
        "followers_count": 0.0,
        "following_count": 0.0,
        "listed_count": 0.0,
        "statuses_count": 0.0,
        "account_age_days": 0.0,
        "description_length": 0.0,
        "username_length": 0.0,
        "name_length": 0.0,
        "has_location": 0.0,
        "has_url": 0.0,
        "is_verified": 0.0,
        "is_protected": 0.0,
        "followers_following_ratio": 0.0,
        "posts_total": 0.0,
        "follow_out_count": 0.0,
        "follow_in_count": 0.0,
        "friend_out_count": 0.0,
        "friend_in_count": 0.0,
        "sampled_tweet_count": 0.0,
    }
    for column, default_value in numeric_fill_defaults.items():
        if column not in users.columns:
            users[column] = default_value
    users = users.fillna(value=numeric_fill_defaults)
    users["combined_text"] = (users["description_text"].fillna("") + " " + users["tweet_text"].fillna("")).str.strip()
    users["label_id"] = users["label"].map({"human": 0, "bot": 1}).astype(int)

    manifest = {
        "profile_numeric_columns": [
            "followers_count",
            "following_count",
            "listed_count",
            "statuses_count",
            "account_age_days",
            "description_length",
            "username_length",
            "name_length",
            "has_location",
            "has_url",
            "is_verified",
            "is_protected",
            "followers_following_ratio",
            "posts_total",
            "sampled_tweet_count",
            "follow_out_count",
            "follow_in_count",
            "friend_out_count",
            "friend_in_count",
        ],
        "graph_numeric_columns": [],
        "text_column": "combined_text",
        "label_column": "label_id",
        "split_column": "split",
        "id_column": "user_id",
    }

    users = users.reset_index().rename(columns={"index": "user_id"})
    users.to_csv(config.cache_dir / "users.csv", index=False)
    graph_edges_df.to_csv(config.cache_dir / "graph_edges.csv", index=False)
    pd.DataFrame({"user_id": sorted(graph_user_ids)}).to_csv(config.cache_dir / "graph_nodes.csv", index=False)

    with (config.cache_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    with (config.cache_dir / "network_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(network_summary, handle, ensure_ascii=False, indent=2)

    return PreparedData(
        users=users,
        graph_edges=graph_edges_df,
        network_summary=network_summary,
        manifest=manifest,
        graph_user_ids=sorted(graph_user_ids),
    )


def collect_edge_information(
    edge_path: Path,
    graph_user_ids: set[str],
    labeled_user_ids: set[str],
    max_tweets_per_user: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, list[str]], dict[str, Any]]:
    edge_weights: Counter[tuple[str, str]] = Counter()
    edge_relations: dict[tuple[str, str], set[str]] = defaultdict(set)
    labeled_stats: dict[str, Counter[str]] = defaultdict(Counter)
    sampled_tweets: dict[int, list[str]] = defaultdict(list)
    sampled_per_user: Counter[str] = Counter()
    relation_counter: Counter[str] = Counter()

    with edge_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = row["source_id"]
            relation = row["relation"]
            target = row["target_id"]
            relation_counter[relation] += 1

            if relation in {"follow", "friend"}:
                if source in graph_user_ids and target in graph_user_ids:
                    edge_weights[(source, target)] += 1
                    edge_relations[(source, target)].add(relation)
                    if source in labeled_user_ids:
                        labeled_stats[source][f"{relation}_out_count"] += 1
                    if target in labeled_user_ids:
                        labeled_stats[target][f"{relation}_in_count"] += 1
                elif source in labeled_user_ids:
                    labeled_stats[source][f"{relation}_out_count"] += 1

            elif relation == "post" and source in labeled_user_ids:
                labeled_stats[source]["posts_total"] += 1
                if sampled_per_user[source] >= max_tweets_per_user:
                    continue
                if target.startswith("t") and target[1:].isdigit():
                    sampled_tweets[int(target[1:])].append(source)
                    sampled_per_user[source] += 1
                    labeled_stats[source]["sampled_tweet_count"] += 1

    edge_records = [
        {
            "source_id": source,
            "target_id": target,
            "weight": weight,
            "relations": "|".join(sorted(edge_relations[(source, target)])),
        }
        for (source, target), weight in edge_weights.items()
    ]
    graph_edges_df = pd.DataFrame(edge_records, columns=["source_id", "target_id", "weight", "relations"])

    labeled_rows = []
    for user_id in labeled_user_ids:
        stats = labeled_stats.get(user_id, Counter())
        row = {"user_id": user_id}
        row.update(stats)
        labeled_rows.append(row)
    edge_stats_df = pd.DataFrame(labeled_rows).set_index("user_id").fillna(0.0)

    network_summary = {
        "graph_user_count": len(graph_user_ids),
        "labeled_user_count": len(labeled_user_ids),
        "directed_user_edges": int(len(graph_edges_df)),
        "relation_breakdown": dict(relation_counter),
        "sampled_tweets": int(sum(sampled_per_user.values())),
        "max_tweets_per_user": max_tweets_per_user,
    }
    return graph_edges_df, edge_stats_df, sampled_tweets, network_summary


def parse_nodes(
    node_path: Path,
    labeled_user_ids: set[str],
    sampled_tweets: dict[int, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_records: dict[str, dict[str, Any]] = {}
    tweets_by_user: dict[str, list[str]] = defaultdict(list)
    remaining_tweets = dict(sampled_tweets)

    for record in stream_json_array(node_path):
        node_id = str(record.get("id", "")).strip()
        if not node_id:
            continue

        if node_id.startswith("u") and node_id in labeled_user_ids:
            profile_records[node_id] = extract_profile_features(record)
        elif node_id.startswith("t") and node_id[1:].isdigit():
            owners = remaining_tweets.pop(int(node_id[1:]), None)
            if owners:
                text = normalize_text(record.get("text"))
                for owner in owners:
                    tweets_by_user[owner].append(text)

        if len(profile_records) == len(labeled_user_ids) and not remaining_tweets:
            break

    profile_df = pd.DataFrame.from_dict(profile_records, orient="index")

    text_rows = [{"user_id": user_id, "tweet_text": " ".join(chunk for chunk in tweets_by_user.get(user_id, []) if chunk).strip()} for user_id in labeled_user_ids]
    text_df = pd.DataFrame(text_rows).set_index("user_id")

    return profile_df, text_df


def extract_profile_features(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("public_metrics") or {}
    followers = float(metrics.get("followers_count", 0.0) or 0.0)
    following = float(metrics.get("following_count", 0.0) or 0.0)
    listed = float(metrics.get("listed_count", 0.0) or 0.0)
    statuses = float(metrics.get("tweet_count", 0.0) or 0.0)

    description = normalize_text(record.get("description"))
    username = normalize_text(record.get("username"))
    display_name = normalize_text(record.get("name"))
    location = normalize_text(record.get("location"))
    profile_url = normalize_text(record.get("url"))

    return {
        "screen_name": username,
        "display_name": display_name,
        "location": location,
        "profile_url": profile_url,
        "description_text": description,
        "followers_count": followers,
        "following_count": following,
        "listed_count": listed,
        "statuses_count": statuses,
        "account_age_days": parse_account_age_days(record.get("created_at")),
        "description_length": float(len(description)),
        "username_length": float(len(username)),
        "name_length": float(len(display_name)),
        "has_location": float(bool(location)),
        "has_url": float(bool(profile_url) and profile_url.lower() != "none"),
        "is_verified": parse_bool(record.get("verified")),
        "is_protected": parse_bool(record.get("protected")),
        "followers_following_ratio": safe_ratio(followers + 1.0, following + 1.0),
    }


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    return " ".join(text.split()).strip()


def parse_bool(value: Any) -> float:
    normalized = normalize_text(value).lower()
    return float(normalized in {"true", "1", "yes"})


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def parse_account_age_days(value: Any) -> float:
    created_at = normalize_text(value)
    if not created_at:
        return 0.0
    try:
        parsed = datetime.strptime(created_at, TWITTER_TIME_FORMAT)
    except ValueError:
        return 0.0
    age = datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)
    return float(max(age.days, 0))
