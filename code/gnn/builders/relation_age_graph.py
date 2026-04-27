from __future__ import annotations

import pandas as pd
import torch


def _build_age_relation_graph(
    id_to_index: dict[str, int],
    user_age_buckets: torch.Tensor,
    graph_edges: pd.DataFrame,
    num_age_buckets: int = 3,
    relation_to_id: dict[str, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if relation_to_id is None:
        relation_to_id = {"follow": 0, "friend": 1}
    buckets = num_age_buckets

    rows: list[int] = []
    cols: list[int] = []
    edge_types: list[int] = []

    for row in graph_edges.itertuples(index=False):
        source = id_to_index.get(row.source_id)
        target = id_to_index.get(row.target_id)
        rel_id = relation_to_id.get(row.relation)
        if source is None or target is None or rel_id is None:
            continue

        src_bucket = int(user_age_buckets[source].item())
        tgt_bucket = int(user_age_buckets[target].item())
        src_bucket = max(0, min(buckets - 1, src_bucket))
        tgt_bucket = max(0, min(buckets - 1, tgt_bucket))

        edge_type = rel_id * buckets * buckets + src_bucket * buckets + tgt_bucket
        rows.append(source)
        cols.append(target)
        edge_types.append(edge_type)

    if not rows:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long), torch.tensor(edge_types, dtype=torch.long)