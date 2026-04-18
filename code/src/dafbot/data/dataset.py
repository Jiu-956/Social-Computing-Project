from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from dafbot.utils import load_pickle


@dataclass(slots=True)
class ProcessedDataset:
    user_table: pd.DataFrame
    attr_features: torch.Tensor
    text_features: torch.Tensor
    quality_features: torch.Tensor
    labels: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    snapshots: list[dict[str, Any]]
    meta: dict[str, Any]


def _sample_indices(length: int, count: int) -> list[int]:
    if count is None or count >= length:
        return list(range(length))
    if count <= 1:
        return [length - 1]
    positions = torch.linspace(0, length - 1, steps=count)
    return sorted({int(round(value.item())) for value in positions})


def load_processed_dataset(config: dict[str, Any]) -> ProcessedDataset:
    processed_dir = Path(config["paths"]["processed_dir"])
    user_table = pd.read_csv(processed_dir / "user_table.csv", low_memory=False)
    attr_features = torch.load(processed_dir / "attr_features.pt", map_location="cpu")
    text_features = torch.load(processed_dir / "text_features.pt", map_location="cpu")
    quality_features = torch.load(processed_dir / "quality_features.pt", map_location="cpu")
    labels = torch.load(processed_dir / "labels.pt", map_location="cpu")
    train_idx = torch.load(processed_dir / "train_idx.pt", map_location="cpu")
    val_idx = torch.load(processed_dir / "val_idx.pt", map_location="cpu")
    test_idx = torch.load(processed_dir / "test_idx.pt", map_location="cpu")
    meta = load_pickle(processed_dir / "dynamic_graph" / "meta.pkl")

    snapshot_paths = meta["snapshot_paths"]
    max_snapshots = config["graph"].get("max_snapshots")
    indices = _sample_indices(len(snapshot_paths), max_snapshots) if max_snapshots else list(range(len(snapshot_paths)))
    snapshots = [torch.load(snapshot_paths[index], map_location="cpu") for index in indices]
    selected_dates = [meta["snapshot_dates"][index] for index in indices]
    meta = dict(meta)
    meta["selected_snapshot_indices"] = indices
    meta["selected_snapshot_dates"] = selected_dates
    return ProcessedDataset(
        user_table=user_table,
        attr_features=attr_features,
        text_features=text_features,
        quality_features=quality_features,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        snapshots=snapshots,
        meta=meta,
    )
