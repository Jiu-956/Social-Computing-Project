from __future__ import annotations

import csv
import json
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from dafbot.utils import clean_text, dump_json, dump_pickle, parse_bool, parse_iso_datetime, parse_twitter_datetime, safe_div, safe_float


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
                    possible_record = trailing.lstrip(",").lstrip()
                    if possible_record.startswith("{"):
                        warnings.warn(
                            f"{path} ended with a truncated JSON object; ignoring the incomplete trailing record.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        return
                    raise ValueError(f"Unexpected trailing content in {path}.")
                return

            if not parsed and len(buffer) > chunk_size * 8:
                raise ValueError(f"Parser buffer grew too large while reading {path}.")


def _load_users(raw_dir: Path) -> pd.DataFrame:
    split_df = pd.read_csv(raw_dir / "split.csv").rename(columns={"id": "user_id"})
    label_df = pd.read_csv(raw_dir / "label.csv").rename(columns={"id": "user_id"})
    users = split_df.merge(label_df, on="user_id", how="left")
    users["label_id"] = users["label"].map({"human": 0, "bot": 1}).fillna(-1).astype(int)
    return users


def _scan_edges(edge_path: Path, user_ids: set[str], max_tweets_per_user: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]], dict[str, int]]:
    relation_counter: Counter[str] = Counter()
    node_counter: dict[str, Counter[str]] = defaultdict(Counter)
    sampled_posts: dict[str, list[str]] = defaultdict(list)
    edge_rows: list[dict[str, Any]] = []

    with edge_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_id = clean_text(row["source_id"])
            relation = clean_text(row["relation"])
            target_id = clean_text(row["target_id"])
            if source_id not in user_ids:
                continue

            if relation in {"follow", "friend"} and target_id in user_ids:
                relation_counter[relation] += 1
                edge_rows.append({"source_id": source_id, "relation": relation, "target_id": target_id})
                node_counter[source_id][f"{relation}_out_count"] += 1
                node_counter[target_id][f"{relation}_in_count"] += 1
            elif relation == "post":
                node_counter[source_id]["post_count"] += 1
                if len(sampled_posts[source_id]) < max_tweets_per_user:
                    sampled_posts[source_id].append(target_id)

    stats_rows = []
    for user_id in sorted(user_ids):
        stats = node_counter[user_id]
        follow_in = float(stats.get("follow_in_count", 0))
        follow_out = float(stats.get("follow_out_count", 0))
        friend_in = float(stats.get("friend_in_count", 0))
        friend_out = float(stats.get("friend_out_count", 0))
        total_in = follow_in + friend_in
        total_out = follow_out + friend_out
        stats_rows.append(
            {
                "user_id": user_id,
                "follow_in_count": follow_in,
                "follow_out_count": follow_out,
                "friend_in_count": friend_in,
                "friend_out_count": friend_out,
                "total_in_degree": total_in,
                "total_out_degree": total_out,
                "neighbor_num": total_in + total_out,
                "in_out_ratio": safe_div(total_in, total_out + 1.0),
                "friend_share_out": safe_div(friend_out, total_out + 1.0),
                "follow_share_out": safe_div(follow_out, total_out + 1.0),
                "post_count": float(stats.get("post_count", 0)),
            }
        )

    edge_frame = pd.DataFrame(edge_rows, columns=["source_id", "relation", "target_id"])
    stats_frame = pd.DataFrame(stats_rows)
    return edge_frame, stats_frame, sampled_posts, dict(relation_counter)


def _extract_profile_row(record: dict[str, Any], fallback_created_at: datetime) -> dict[str, Any]:
    metrics = record.get("public_metrics") or {}
    followers_count = safe_float(metrics.get("followers_count"))
    friends_count = safe_float(metrics.get("following_count"))
    statuses_count = safe_float(metrics.get("tweet_count"))
    listed_count = safe_float(metrics.get("listed_count"))
    favourites_count = safe_float(metrics.get("favourites_count"))

    description_text = clean_text(record.get("description"))
    username = clean_text(record.get("username"))
    display_name = clean_text(record.get("name"))
    location = clean_text(record.get("location"))
    url = clean_text(record.get("url"))
    profile_image_url = clean_text(record.get("profile_image_url"))
    created_at = parse_twitter_datetime(record.get("created_at")) or fallback_created_at
    now = datetime.now(timezone.utc)
    account_age_days = max((now - created_at).total_seconds() / 86400.0, 0.0)
    username_digits = sum(char.isdigit() for char in username)

    default_profile_image = 1.0 if (not profile_image_url or "default_profile" in profile_image_url.lower()) else 0.0
    return {
        "user_id": clean_text(record.get("id")),
        "created_at_iso": created_at.isoformat(),
        "followers_count": followers_count,
        "friends_count": friends_count,
        "following_count": friends_count,
        "listed_count": listed_count,
        "favourites_count": favourites_count,
        "statuses_count": statuses_count,
        "verified": float(parse_bool(record.get("verified"))),
        "default_profile": 0.0,
        "default_profile_image": default_profile_image,
        "has_extended_profile": 0.0,
        "protected": float(parse_bool(record.get("protected"))),
        "geo_enabled": 0.0,
        "description_text": description_text,
        "url": url,
        "screen_name": username,
        "username": username,
        "name": display_name,
        "display_name": display_name,
        "location": location,
        "username_length": float(len(username)),
        "display_name_length": float(len(display_name)),
        "username_digit_ratio": safe_div(username_digits, len(username)) if username else 0.0,
        "description_length": float(len(description_text)),
        "url_is_empty": float(not url),
        "description_is_empty": float(not description_text),
        "account_age_days": account_age_days,
        "profile_missing_count": float(sum(int(not field) for field in (description_text, url, location))),
    }


def _parse_nodes(node_path: Path, user_ids: set[str], sampled_posts: dict[str, list[str]], fallback_created_at: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_tweet_ids = {tweet_id for tweet_ids in sampled_posts.values() for tweet_id in tweet_ids}
    remaining_users = set(user_ids)
    remaining_tweets = set(target_tweet_ids)
    profile_rows: dict[str, dict[str, Any]] = {}
    tweet_lookup: dict[str, str] = {}

    for record in stream_json_array(node_path):
        record_id = clean_text(record.get("id"))
        if not record_id:
            continue
        if record_id in remaining_users:
            profile_rows[record_id] = _extract_profile_row(record, fallback_created_at)
            remaining_users.remove(record_id)
        elif record_id in remaining_tweets:
            tweet_lookup[record_id] = clean_text(record.get("text"))
            remaining_tweets.remove(record_id)
        if not remaining_users and not remaining_tweets:
            break

    profile_df = pd.DataFrame(profile_rows.values())
    if profile_df.empty:
        profile_df = pd.DataFrame({"user_id": sorted(user_ids)})

    tweet_rows = []
    for user_id, tweet_ids in sampled_posts.items():
        texts = [tweet_lookup[tweet_id] for tweet_id in tweet_ids if tweet_lookup.get(tweet_id)]
        tweet_rows.append(
            {
                "user_id": user_id,
                "tweet_text": " [SEP] ".join(texts).strip(),
                "sampled_tweet_count": float(len(texts)),
            }
        )
    tweet_df = pd.DataFrame(tweet_rows, columns=["user_id", "tweet_text", "sampled_tweet_count"])
    return profile_df, tweet_df


def build_user_table(config: dict[str, Any]) -> pd.DataFrame:
    raw_dir = Path(config["paths"]["data_root"])
    processed_dir = Path(config["paths"]["processed_dir"])
    fallback_created_at = parse_iso_datetime(config["preprocess"]["fallback_created_at"])

    users = _load_users(raw_dir)
    user_ids = set(users["user_id"].tolist())
    edge_frame, graph_stats, sampled_posts, relation_breakdown = _scan_edges(
        raw_dir / "edge.csv",
        user_ids,
        config["preprocess"]["max_tweets_per_user"],
    )
    profile_df, tweet_df = _parse_nodes(
        raw_dir / "node.json",
        user_ids,
        sampled_posts,
        fallback_created_at=fallback_created_at,
    )

    user_table = users.merge(profile_df, on="user_id", how="left").merge(graph_stats, on="user_id", how="left")
    user_table = user_table.merge(tweet_df, on="user_id", how="left")
    if "tweet_text" not in user_table.columns:
        user_table["tweet_text"] = ""
    user_table["tweet_text"] = user_table["tweet_text"].fillna("")
    if "sampled_tweet_count" not in user_table.columns:
        user_table["sampled_tweet_count"] = 0.0
    user_table["sampled_tweet_count"] = user_table["sampled_tweet_count"].fillna(0.0)

    fill_defaults = {
        "followers_count": 0.0,
        "friends_count": 0.0,
        "following_count": 0.0,
        "listed_count": 0.0,
        "favourites_count": 0.0,
        "statuses_count": 0.0,
        "verified": 0.0,
        "default_profile": 0.0,
        "default_profile_image": 0.0,
        "has_extended_profile": 0.0,
        "protected": 0.0,
        "geo_enabled": 0.0,
        "description_text": "",
        "username": "",
        "display_name": "",
        "screen_name": "",
        "name": "",
        "url": "",
        "location": "",
        "username_length": 0.0,
        "display_name_length": 0.0,
        "username_digit_ratio": 0.0,
        "description_length": 0.0,
        "url_is_empty": 1.0,
        "description_is_empty": 1.0,
        "account_age_days": 0.0,
        "profile_missing_count": 3.0,
        "follow_in_count": 0.0,
        "follow_out_count": 0.0,
        "friend_in_count": 0.0,
        "friend_out_count": 0.0,
        "total_in_degree": 0.0,
        "total_out_degree": 0.0,
        "neighbor_num": 0.0,
        "in_out_ratio": 0.0,
        "friend_share_out": 0.0,
        "follow_share_out": 0.0,
        "post_count": 0.0,
        "created_at_iso": fallback_created_at.isoformat(),
    }
    for column, default in fill_defaults.items():
        if column not in user_table.columns:
            user_table[column] = default
        else:
            user_table[column] = user_table[column].fillna(default)

    user_table = user_table.sort_values("user_id").reset_index(drop=True)
    user_table.insert(0, "user_index", range(len(user_table)))

    uid2index = dict(zip(user_table["user_id"], user_table["user_index"]))
    index2uid = user_table["user_id"].tolist()

    user_table.to_csv(processed_dir / "user_table.csv", index=False)
    edge_frame.to_csv(processed_dir / "graph_edges.csv", index=False)
    dump_pickle(uid2index, processed_dir / "uid2index.pkl")
    dump_pickle(index2uid, processed_dir / "index2uid.pkl")
    dump_pickle(sampled_posts, processed_dir / "sampled_posts.pkl")
    dump_json(
        {
            "user_count": int(len(user_table)),
            "graph_edge_count": int(len(edge_frame)),
            "relation_breakdown": relation_breakdown,
            "sampled_tweet_users": int(sum(bool(value) for value in sampled_posts.values())),
        },
        processed_dir / "dataset_manifest.json",
    )
    return user_table
