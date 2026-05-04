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

    # Vectorized edge building: extract all source/target arrays at once
    if len(graph_edges) > 0:
        all_sources = graph_edges["source_id"].map(id_to_index).values
        all_targets = graph_edges["target_id"].map(id_to_index).values
        valid_mask = pd.notna(all_sources) & pd.notna(all_targets)
        source_array = np.asarray(all_sources[valid_mask], dtype=np.int64)
        target_array = np.asarray(all_targets[valid_mask], dtype=np.int64)
        relation_array = graph_edges["relation"].values[valid_mask]
    else:
        source_array = np.array([], dtype=np.int64)
        target_array = np.array([], dtype=np.int64)
        relation_array = np.array([], dtype=object)

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
        snapshot_edge_count = snapshot_source.size

        if snapshot_edge_count == 0:
            edge_indices.append(torch.empty((2, 0), dtype=torch.long))
            clustering_list.append(torch.zeros((node_count, 1), dtype=torch.float32))
            bidirectional_list.append(torch.zeros((node_count, 1), dtype=torch.float32))
            density_list.append(torch.zeros((node_count, 1), dtype=torch.float32))
            continue

        edge_indices.append(torch.tensor(np.stack([snapshot_source, snapshot_target], axis=0), dtype=torch.long))

        # --- Clustering coefficient (networkx, matches reference) ---
        clustering_tensor = _compute_clustering_coefficient(
            snapshot_source, snapshot_target, node_count,
        )

        # --- Bidirectional links ratio (matches reference logic, vectorized) ---
        bidirectional_tensor = _compute_bidirectional_ratio(
            snapshot_source, snapshot_target, relation_array[valid_edges], node_count,
        )

        clustering_list.append(clustering_tensor)
        bidirectional_list.append(bidirectional_tensor)

        density_value = float(snapshot_edge_count / denominator)
        density_list.append(torch.full((node_count, 1), density_value, dtype=torch.float32))

    return {
        "edge_indices": edge_indices,
        "clustering": torch.stack(clustering_list, dim=0),
        "bidirectional_ratio": torch.stack(bidirectional_list, dim=0),
        "edge_density": torch.stack(density_list, dim=0),
        "keep_ratio": torch.tensor(keep_ratios, dtype=torch.float32).view(snapshot_count, 1, 1).repeat(1, node_count, 1),
    }


def _compute_clustering_coefficient(
    source: np.ndarray,
    target: np.ndarray,
    node_count: int,
) -> torch.Tensor:
    """Compute clustering coefficient using triangle counting (vectorized).

    Matches networkx.clustering(G) for undirected graphs without requiring
    the full networkx dependency or the O(|V| * d_max^2) overhead.
    """
    import numpy as np
    n = node_count

    # Build adjacency as dict-of-sets for efficient triangle counting
    # We only build for nodes that have edges
    adj: dict[int, set] = {}
    degree = np.zeros(n, dtype=np.float32)
    for s, t in zip(source, target, strict=False):
        si, ti = int(s), int(t)
        if si == ti:
            continue
        if si not in adj:
            adj[si] = set()
        if ti not in adj:
            adj[ti] = set()
        adj[si].add(ti)
        adj[ti].add(si)
        degree[si] += 1.0
        degree[ti] += 1.0

    clustering = np.zeros(n, dtype=np.float32)
    for v, neighbors in adj.items():
        dv = len(neighbors)
        if dv < 2:
            clustering[v] = 0.0
            continue
        triangles = 0
        neighbors_list = list(neighbors)
        for i in range(dv):
            u = neighbors_list[i]
            u_neighbors = adj.get(u, set())
            for j in range(i + 1, dv):
                w = neighbors_list[j]
                if w in u_neighbors:
                    triangles += 1
        clustering[v] = (2.0 * triangles) / (dv * (dv - 1))

    return torch.tensor(clustering.reshape(-1, 1), dtype=torch.float32)


def _compute_bidirectional_ratio(
    source: np.ndarray,
    target: np.ndarray,
    relations: np.ndarray,
    node_count: int,
) -> torch.Tensor:
    """Compute bidirectional links ratio using the reference logic.

    Only considers 'follow' edges. For each node, counts outgoing follow edges
    that have a reciprocal follow edge, divided by out_degree.
    """
    n = node_count
    follow_mask = relations == "follow"
    follow_src = source[follow_mask]
    follow_tgt = target[follow_mask]

    out_degree = np.bincount(follow_src, minlength=n).astype(np.float32)
    reciprocal = np.zeros(n, dtype=np.float32)

    if follow_src.size > 0:
        forward_keys = follow_src * n + follow_tgt
        reverse_keys = follow_tgt * n + follow_src
        uf = np.unique(forward_keys)
        ur_sorted = np.sort(np.unique(reverse_keys))
        if ur_sorted.size > 0:
            pos = np.searchsorted(ur_sorted, uf)
            pos = np.clip(pos, 0, ur_sorted.size - 1)
            matches = ur_sorted[pos] == uf
            match_src = uf[matches] // n
            np.add.at(reciprocal, match_src, 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(reciprocal, out_degree, out=np.zeros_like(reciprocal), where=out_degree > 0)
    return torch.tensor(ratio.reshape(-1, 1), dtype=torch.float32)
