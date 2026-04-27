from __future__ import annotations

import pandas as pd
import torch


def _build_relation_graph(id_to_index: dict[str, int], graph_edges: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[int] = []
    cols: list[int] = []
    edge_types: list[int] = []
    relation_to_id = {"follow": 0, "friend": 1}
    for row in graph_edges.itertuples(index=False):
        source = id_to_index.get(row.source_id)
        target = id_to_index.get(row.target_id)
        relation_id = relation_to_id.get(row.relation)
        if source is None or target is None or relation_id is None:
            continue
        rows.append(source)
        cols.append(target)
        edge_types.append(relation_id)
    if not rows:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long), torch.tensor(edge_types, dtype=torch.long)
