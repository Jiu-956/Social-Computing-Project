from __future__ import annotations

import pandas as pd
import torch


def _build_combined_edge_index(id_to_index: dict[str, int], graph_edges: pd.DataFrame) -> torch.Tensor:
    rows: list[int] = []
    cols: list[int] = []
    for row in graph_edges.itertuples(index=False):
        source = id_to_index.get(row.source_id)
        target = id_to_index.get(row.target_id)
        if source is None or target is None:
            continue
        rows.extend([source, target])
        cols.extend([target, source])
    if not rows:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long)
