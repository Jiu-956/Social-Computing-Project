from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dafbot.utils import dump_json


def export_metrics_table(metrics: list[dict[str, Any]], path: str | Path) -> pd.DataFrame:
    frame = pd.DataFrame(metrics)
    frame.to_csv(path, index=False)
    return frame


def export_modal_weight_statistics(
    user_table: pd.DataFrame,
    labels: torch.Tensor,
    modal_weights: torch.Tensor,
    output_dir: str | Path,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    frame = pd.DataFrame(modal_weights.detach().cpu().numpy(), columns=["attr_weight", "text_weight", "dyn_weight"])
    frame["label_id"] = labels.detach().cpu().numpy()
    frame["neighbor_num"] = pd.to_numeric(user_table["neighbor_num"], errors="coerce").fillna(0.0).to_numpy()

    rows = [
        {"group": "all", **frame[["attr_weight", "text_weight", "dyn_weight"]].mean().to_dict()},
        {"group": "human", **frame.loc[frame["label_id"] == 0, ["attr_weight", "text_weight", "dyn_weight"]].mean().to_dict()},
        {"group": "bot", **frame.loc[frame["label_id"] == 1, ["attr_weight", "text_weight", "dyn_weight"]].mean().to_dict()},
    ]
    bins = [(0, 5), (6, 20), (21, None)]
    for lower, upper in bins:
        if upper is None:
            mask = frame["neighbor_num"] >= lower
            label = f"degree_{lower}_plus"
        else:
            mask = (frame["neighbor_num"] >= lower) & (frame["neighbor_num"] <= upper)
            label = f"degree_{lower}_{upper}"
        if mask.any():
            rows.append({"group": label, **frame.loc[mask, ["attr_weight", "text_weight", "dyn_weight"]].mean().to_dict()})

    stats_frame = pd.DataFrame(rows)
    stats_frame.to_csv(output_dir / "modal_weight_statistics.csv", index=False)

    plot_frame = stats_frame.set_index("group")
    plot_frame.plot(kind="bar", figsize=(10, 5))
    plt.tight_layout()
    plt.savefig(output_dir / "modal_weight_statistics.png", dpi=200)
    plt.close()
    return stats_frame


def export_temporal_heatmap(
    temporal_weights: torch.Tensor | None,
    user_table: pd.DataFrame,
    snapshot_dates: list[str],
    output_dir: str | Path,
    top_k: int,
) -> None:
    if temporal_weights is None:
        return
    output_dir = Path(output_dir)
    weights = temporal_weights.detach().cpu().numpy()
    sample_count = min(top_k, weights.shape[0])
    sample_indices = np.linspace(0, weights.shape[0] - 1, num=sample_count, dtype=int)
    sampled = weights[sample_indices]

    plt.figure(figsize=(max(8, len(snapshot_dates) * 0.4), max(6, sample_count * 0.2)))
    plt.imshow(sampled, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.yticks(range(sample_count), user_table.iloc[sample_indices]["user_id"].tolist(), fontsize=6)
    plt.xticks(range(len(snapshot_dates)), snapshot_dates, rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "temporal_weight_heatmap.png", dpi=200)
    plt.close()

    pd.DataFrame(sampled, index=user_table.iloc[sample_indices]["user_id"], columns=snapshot_dates).to_csv(
        output_dir / "temporal_weight_heatmap.csv"
    )


def export_error_analysis(
    user_table: pd.DataFrame,
    labels: torch.Tensor,
    logits: torch.Tensor,
    modal_weights: torch.Tensor,
    temporal_weights: torch.Tensor | None,
    split_indices: torch.Tensor,
    output_dir: str | Path,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    probabilities = torch.softmax(logits[split_indices], dim=1)[:, 1].detach().cpu().numpy()
    preds = (probabilities >= 0.5).astype(np.int64)
    truth = labels[split_indices].detach().cpu().numpy()
    indices = split_indices.detach().cpu().numpy()
    subset = user_table.iloc[indices].copy()
    subset["true_label"] = truth
    subset["pred_label"] = preds
    subset["bot_probability"] = probabilities
    subset["attr_weight"] = modal_weights[split_indices, 0].detach().cpu().numpy()
    subset["text_weight"] = modal_weights[split_indices, 1].detach().cpu().numpy()
    subset["dyn_weight"] = modal_weights[split_indices, 2].detach().cpu().numpy()
    if temporal_weights is not None:
        subset["temporal_weights"] = [
            json.dumps(values.tolist(), ensure_ascii=False)
            for values in temporal_weights[split_indices].detach().cpu().numpy()
        ]
    errors = subset[subset["true_label"] != subset["pred_label"]].copy()
    errors.to_csv(output_dir / "misclassified_samples.csv", index=False)
    return errors


def export_prediction_table(
    user_table: pd.DataFrame,
    labels: torch.Tensor,
    logits: torch.Tensor,
    split_indices: torch.Tensor,
    split_name: str,
    output_dir: str | Path,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    probabilities = torch.softmax(logits[split_indices], dim=1)[:, 1].detach().cpu().numpy()
    preds = (probabilities >= 0.5).astype(np.int64)
    truth = labels[split_indices].detach().cpu().numpy()
    indices = split_indices.detach().cpu().numpy()

    frame = user_table.iloc[indices][["user_id", "split"]].copy()
    frame["true_label"] = truth
    frame["pred_label"] = preds
    frame["bot_probability"] = probabilities
    frame.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)
    return frame
