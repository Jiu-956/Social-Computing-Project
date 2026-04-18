from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch

from dafbot.utils import dump_json, dump_pickle


def _month_floor(value: datetime) -> datetime:
    return value.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _add_months(value: datetime, months: int) -> datetime:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    return value.replace(year=year, month=month, day=1)


def _quarter_floor(value: datetime) -> datetime:
    quarter_month = ((value.month - 1) // 3) * 3 + 1
    return value.replace(month=quarter_month, day=1, hour=0, minute=0, second=0, microsecond=0)


def _year_floor(value: datetime) -> datetime:
    return value.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


def build_snapshot_dates(created_dates: list[datetime], granularity: str) -> list[datetime]:
    min_date = min(created_dates)
    max_date = max(created_dates)
    if granularity == "monthly":
        start = _month_floor(min_date)
        end = _add_months(_month_floor(max_date), 1)
        step = 1
        return [_add_months(start, offset) for offset in range(((end.year - start.year) * 12 + end.month - start.month) + 1)]
    if granularity == "quarterly":
        start = _quarter_floor(min_date)
        end = _add_months(_quarter_floor(max_date), 3)
        count = ((end.year - start.year) * 12 + end.month - start.month) // 3
        return [_add_months(start, 3 * offset) for offset in range(count + 1)]
    if granularity == "yearly":
        start = _year_floor(min_date)
        end = _year_floor(max_date).replace(year=_year_floor(max_date).year + 1)
        return [start.replace(year=start.year + offset) for offset in range(end.year - start.year + 1)]
    raise ValueError(f"Unsupported granularity: {granularity}")


def _relation_edges_to_arrays(graph_edges: pd.DataFrame, uid2index: dict[str, int], relation_map: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_indices = graph_edges["source_id"].map(uid2index).to_numpy(dtype=np.int64)
    target_indices = graph_edges["target_id"].map(uid2index).to_numpy(dtype=np.int64)
    edge_types = graph_edges["relation"].map(relation_map).to_numpy(dtype=np.int64)
    return source_indices, target_indices, edge_types


def _clustering_feature(active_nodes: np.ndarray, edge_pairs: list[tuple[int, int]], node_count: int) -> np.ndarray:
    values = np.zeros(node_count, dtype=np.float32)
    if len(active_nodes) == 0:
        return values.reshape(-1, 1)
    graph = nx.Graph()
    graph.add_nodes_from(active_nodes.tolist())
    graph.add_edges_from(edge_pairs)
    clustering = nx.clustering(graph)
    for node, value in clustering.items():
        values[node] = float(value)
    return values.reshape(-1, 1)


def _bidirectional_feature(active_nodes: np.ndarray, edge_pairs: list[tuple[int, int]], node_count: int) -> np.ndarray:
    values = np.zeros(node_count, dtype=np.float32)
    if len(active_nodes) == 0:
        return values.reshape(-1, 1)
    outgoing: dict[int, set[int]] = defaultdict(set)
    for source, target in edge_pairs:
        outgoing[source].add(target)
    for node in active_nodes.tolist():
        neighbors = outgoing.get(node, set())
        if not neighbors:
            continue
        mutual = sum(1 for neighbor in neighbors if node in outgoing.get(neighbor, set()))
        values[node] = mutual / max(len(neighbors), 1)
    return values.reshape(-1, 1)


def _build_quality_features(user_table: pd.DataFrame, exist_ratio: np.ndarray, processed_dir: Path) -> torch.Tensor:
    numeric = pd.DataFrame(index=user_table.index)
    numeric["text_len"] = user_table["description_text"].fillna("").astype(str).str.len() + user_table["tweet_text"].fillna("").astype(str).str.len()
    numeric["tweet_num"] = pd.to_numeric(user_table["post_count"], errors="coerce").fillna(0.0)
    numeric["neighbor_num"] = pd.to_numeric(user_table["neighbor_num"], errors="coerce").fillna(0.0)
    numeric["missing_ratio"] = pd.to_numeric(user_table["profile_missing_count"], errors="coerce").fillna(0.0) / 3.0
    numeric["account_age_days"] = pd.to_numeric(user_table["account_age_days"], errors="coerce").fillna(0.0)
    numeric["statuses_count"] = pd.to_numeric(user_table["statuses_count"], errors="coerce").fillna(0.0)
    numeric["exist_ratio_in_snapshots"] = exist_ratio.astype(np.float32)

    values = numeric.to_numpy(dtype=np.float32)
    values = np.log1p(np.clip(values, 0.0, None))
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    normalized = (values - means) / stds
    tensor = torch.tensor(normalized, dtype=torch.float32)
    torch.save(tensor, processed_dir / "quality_features.pt")
    dump_json(
        {
            "columns": numeric.columns.tolist(),
            "mean": means.tolist(),
            "std": stds.tolist(),
            "feature_dim": int(tensor.shape[1]),
        },
        processed_dir / "quality_feature_meta.json",
    )
    return tensor


def build_dynamic_snapshots(config: dict[str, Any], user_table: pd.DataFrame) -> dict[str, Any]:
    processed_dir = Path(config["paths"]["processed_dir"])
    snapshot_dir = Path(config["paths"]["snapshot_dir"])
    relation_map = config["graph"]["relation_map"]
    graph_edges = pd.read_csv(processed_dir / "graph_edges.csv")
    uid2index = {user_id: int(index) for user_id, index in zip(user_table["user_id"], user_table["user_index"])}
    created_dates = [datetime.fromisoformat(value) for value in user_table["created_at_iso"].tolist()]
    snapshot_dates = build_snapshot_dates(created_dates, config["graph"]["granularity"])

    source_idx, target_idx, edge_types = _relation_edges_to_arrays(graph_edges, uid2index, relation_map)
    created_ts = np.array([value.timestamp() for value in created_dates], dtype=np.float64)
    node_count = len(user_table)
    snapshot_paths: list[str] = []
    exist_masks: list[np.ndarray] = []

    labels = torch.tensor(user_table["label_id"].clip(lower=0).to_numpy(dtype=np.int64), dtype=torch.long)
    train_idx = torch.tensor(np.flatnonzero((user_table["split"] == "train").to_numpy()), dtype=torch.long)
    val_idx = torch.tensor(np.flatnonzero((user_table["split"] == "val").to_numpy()), dtype=torch.long)
    test_idx = torch.tensor(np.flatnonzero((user_table["split"] == "test").to_numpy()), dtype=torch.long)
    torch.save(labels, processed_dir / "labels.pt")
    torch.save(train_idx, processed_dir / "train_idx.pt")
    torch.save(val_idx, processed_dir / "val_idx.pt")
    torch.save(test_idx, processed_dir / "test_idx.pt")

    for snapshot_date in snapshot_dates:
        active_mask = created_ts < snapshot_date.timestamp()
        active_nodes = np.flatnonzero(active_mask)
        exist_mask = active_mask.astype(np.float32)
        exist_masks.append(exist_mask)

        if len(active_nodes) == 0:
            edge_mask = np.zeros_like(source_idx, dtype=bool)
        else:
            edge_mask = active_mask[source_idx] & active_mask[target_idx]

        active_sources = source_idx[edge_mask]
        active_targets = target_idx[edge_mask]
        active_edge_types = edge_types[edge_mask]
        edge_index = torch.tensor(np.stack([active_sources, active_targets], axis=0), dtype=torch.long) if len(active_sources) else torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.tensor(active_edge_types, dtype=torch.long)
        edge_pairs = list(zip(active_sources.tolist(), active_targets.tolist()))

        clustering = _clustering_feature(active_nodes, edge_pairs, node_count)
        bidirectional = _bidirectional_feature(active_nodes, edge_pairs, node_count)

        snapshot_obj = {
            "date": snapshot_date.strftime("%Y-%m-%d"),
            "edge_index": edge_index,
            "edge_type": edge_type,
            "exist_nodes": torch.tensor(exist_mask, dtype=torch.float32),
            "global_index": torch.tensor(active_nodes, dtype=torch.long),
            "clustering_coefficient": torch.tensor(clustering, dtype=torch.float32),
            "bidirectional_links_ratio": torch.tensor(bidirectional, dtype=torch.float32),
        }
        snapshot_path = snapshot_dir / f"snapshot_{snapshot_date.strftime('%Y-%m-%d')}.pt"
        torch.save(snapshot_obj, snapshot_path)
        snapshot_paths.append(str(snapshot_path))

    exist_ratio = np.stack(exist_masks, axis=1).mean(axis=1) if exist_masks else np.zeros(node_count, dtype=np.float32)
    _build_quality_features(user_table, exist_ratio, processed_dir)

    final_snapshot = torch.load(snapshot_paths[-1]) if snapshot_paths else None
    if final_snapshot is not None:
        torch.save(final_snapshot, processed_dir / "static_graph.pt")

    meta = {
        "granularity": config["graph"]["granularity"],
        "snapshot_dates": [value.strftime("%Y-%m-%d") for value in snapshot_dates],
        "snapshot_paths": snapshot_paths,
        "relation_map": relation_map,
        "node_count": node_count,
    }
    dump_pickle(meta, processed_dir / "dynamic_graph" / "meta.pkl")
    dump_json(meta, processed_dir / "dynamic_graph" / "meta.json")
    return meta
