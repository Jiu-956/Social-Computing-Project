from __future__ import annotations

import pandas as pd
import numpy as np
import torch


def _build_botdgt_snapshot_bundle(
    users: pd.DataFrame,
    graph_edges: pd.DataFrame,
    id_to_index: dict[str, int],
    snapshot_count: int,
    min_keep_ratio: float,
) -> dict:
    snapshot_count = max(3, int(snapshot_count))
    node_count = len(users)
    min_keep_ratio = float(np.clip(min_keep_ratio, 0.05, 0.9))

    if "account_age_days" in users.columns:
        account_age = pd.to_numeric(users["account_age_days"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    else:
        account_age = np.zeros((node_count,), dtype=np.float32)

    keep_ratios = np.linspace(min_keep_ratio, 1.0, snapshot_count)
    quantiles = np.clip(1.0 - keep_ratios, 0.0, 1.0)
    thresholds = [float(np.quantile(account_age, quantile)) for quantile in quantiles]

    source_indices: list[int] = []
    target_indices: list[int] = []
    for row in graph_edges.itertuples(index=False):
        source = id_to_index.get(row.source_id)
        target = id_to_index.get(row.target_id)
        if source is None or target is None:
            continue
        source_indices.append(source)
        target_indices.append(target)

    if not source_indices:
        return {
            "edge_indices": [torch.empty((2, 0), dtype=torch.long) for _ in range(snapshot_count)],
            "clustering": torch.zeros((snapshot_count, node_count, 1), dtype=torch.float32),
            "bidirectional_ratio": torch.zeros((snapshot_count, node_count, 1), dtype=torch.float32),
            "edge_density": torch.zeros((snapshot_count, node_count, 1), dtype=torch.float32),
            "keep_ratio": torch.tensor(keep_ratios, dtype=torch.float32).view(snapshot_count, 1, 1).repeat(1, node_count, 1),
        }

    source_array = np.asarray(source_indices, dtype=np.int64)
    target_array = np.asarray(target_indices, dtype=np.int64)

    edge_indices: list[torch.Tensor] = []
    clustering_list: list[torch.Tensor] = []
    bidirectional_list: list[torch.Tensor] = []
    density_list: list[torch.Tensor] = []

    previous_mask = np.zeros(source_array.shape[0], dtype=bool)
    denominator = max(1.0, float(node_count * max(node_count - 1, 1)))
    for keep_ratio, threshold in zip(keep_ratios, thresholds, strict=False):
        exists = account_age >= (threshold - 1e-9)
        valid_edges = exists[source_array] & exists[target_array]
        valid_edges = valid_edges | previous_mask
        previous_mask = valid_edges
        snapshot_source = source_array[valid_edges]
        snapshot_target = target_array[valid_edges]

        if snapshot_source.size == 0:
            edge_indices.append(torch.empty((2, 0), dtype=torch.long))
            clustering_list.append(torch.zeros((node_count, 1), dtype=torch.float32))
            bidirectional_list.append(torch.zeros((node_count, 1), dtype=torch.float32))
            density_list.append(torch.zeros((node_count, 1), dtype=torch.float32))
            continue

        edge_indices.append(torch.tensor(np.stack([snapshot_source, snapshot_target], axis=0), dtype=torch.long))

        undirected_degree = np.bincount(
            np.concatenate([snapshot_source, snapshot_target]),
            minlength=node_count,
        ).astype(np.float32)
        max_degree = max(1.0, float(undirected_degree.max()))
        clustering_proxy = (undirected_degree / max_degree).reshape(-1, 1)
        clustering_list.append(torch.tensor(clustering_proxy, dtype=torch.float32))

        out_degree = np.bincount(snapshot_source, minlength=node_count).astype(np.float32)
        reciprocal = np.zeros((node_count,), dtype=np.float32)
        directed_edges = set(zip(snapshot_source.tolist(), snapshot_target.tolist()))
        for source, target in directed_edges:
            if (target, source) in directed_edges:
                reciprocal[source] += 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(reciprocal, out_degree, out=np.zeros_like(reciprocal), where=out_degree > 0)
        bidirectional_list.append(torch.tensor(ratio.reshape(-1, 1), dtype=torch.float32))

        density_value = float(snapshot_source.size / denominator)
        density_list.append(torch.full((node_count, 1), density_value, dtype=torch.float32))

    return {
        "edge_indices": edge_indices,
        "clustering": torch.stack(clustering_list, dim=0),
        "bidirectional_ratio": torch.stack(bidirectional_list, dim=0),
        "edge_density": torch.stack(density_list, dim=0),
        "keep_ratio": torch.tensor(keep_ratios, dtype=torch.float32).view(snapshot_count, 1, 1).repeat(1, node_count, 1),
    }
