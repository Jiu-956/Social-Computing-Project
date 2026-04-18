from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from dafbot.models.full_model import DynamicAdaptiveFusionBotDetector
from dafbot.train.analysis import export_error_analysis, export_metrics_table, export_modal_weight_statistics, export_prediction_table, export_temporal_heatmap
from dafbot.train.losses import build_total_loss
from dafbot.train.metrics import compute_classification_metrics
from dafbot.utils import dump_json, set_seed


EXPERIMENT_VARIANTS = {
    "attr_only": {"use_attr": True, "use_text": False, "use_graph": False, "fusion_type": "adaptive", "use_quality": False, "temporal_enabled": False},
    "text_only": {"use_attr": False, "use_text": True, "use_graph": False, "fusion_type": "adaptive", "use_quality": False, "temporal_enabled": False},
    "static_graph_only": {"use_attr": False, "use_text": False, "use_graph": True, "fusion_type": "adaptive", "use_quality": False, "temporal_enabled": False, "graph_mode": "static"},
    "attr_text_concat": {"use_attr": True, "use_text": True, "use_graph": False, "fusion_type": "concat", "use_quality": False, "temporal_enabled": False},
    "attr_text_static_graph_concat": {"use_attr": True, "use_text": True, "use_graph": True, "fusion_type": "concat", "use_quality": False, "temporal_enabled": False, "graph_mode": "static"},
    "dynamic_graph_only": {"use_attr": False, "use_text": False, "use_graph": True, "fusion_type": "adaptive", "use_quality": False, "temporal_enabled": True, "graph_mode": "dynamic"},
    "dynamic_graph_concat": {"use_attr": True, "use_text": True, "use_graph": True, "fusion_type": "concat", "use_quality": False, "temporal_enabled": True, "graph_mode": "dynamic"},
    "dynamic_graph_adaptive": {"use_attr": True, "use_text": True, "use_graph": True, "fusion_type": "adaptive", "use_quality": True, "temporal_enabled": True, "graph_mode": "dynamic"},
    "ablation_no_temporal": {"use_attr": True, "use_text": True, "use_graph": True, "fusion_type": "adaptive", "use_quality": True, "temporal_enabled": False, "graph_mode": "static"},
    "ablation_no_adaptive_fusion": {"use_attr": True, "use_text": True, "use_graph": True, "fusion_type": "concat", "use_quality": False, "temporal_enabled": True, "graph_mode": "dynamic"},
    "ablation_no_quality": {"use_attr": True, "use_text": True, "use_graph": True, "fusion_type": "adaptive", "use_quality": False, "temporal_enabled": True, "graph_mode": "dynamic"},
}


@dataclass(slots=True)
class TrainArtifacts:
    checkpoint_path: Path
    metrics_frame: pd.DataFrame
    logits: torch.Tensor
    aux: dict[str, torch.Tensor | None]
    variant_name: str


def _move_snapshot(snapshot: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in snapshot.items()}


def _build_model(config: dict[str, Any], dataset, variant_name: str) -> DynamicAdaptiveFusionBotDetector:
    variant = deepcopy(EXPERIMENT_VARIANTS[variant_name])
    return DynamicAdaptiveFusionBotDetector(
        attr_dim=int(dataset.attr_features.shape[1]),
        text_dim=int(dataset.text_features.shape[1]),
        quality_dim=int(dataset.quality_features.shape[1]),
        model_config=config["model"],
        variant=variant,
    )


def train_experiment(config: dict[str, Any], dataset, variant_name: str) -> TrainArtifacts:
    set_seed(int(config["project"]["seed"]))
    device = torch.device(config["project"]["device"])
    model = _build_model(config, dataset, variant_name).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    attr_features = dataset.attr_features.to(device)
    text_features = dataset.text_features.to(device)
    quality_features = dataset.quality_features.to(device)
    labels = dataset.labels.to(device)
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    test_idx = dataset.test_idx.to(device)
    snapshots = [_move_snapshot(snapshot, device) for snapshot in dataset.snapshots]

    best_val_f1 = -1.0
    best_state = None
    patience_left = int(config["training"]["patience"])
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        optimizer.zero_grad()
        logits, aux = model(attr_features, text_features, snapshots, quality_features)
        loss, loss_terms = build_total_loss(
            logits=logits,
            labels=labels,
            indices=train_idx,
            aux=aux,
            lambda_modal=float(config["training"]["lambda_modal"]),
            lambda_temporal=float(config["training"]["lambda_temporal"]),
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(attr_features, text_features, snapshots, quality_features)
            val_metrics = compute_classification_metrics(val_logits, labels, val_idx, threshold=float(config["evaluation"]["decision_threshold"]))

        history.append({"epoch": epoch, **loss_terms, **{f"val_{key}": value for key, value in val_metrics.items()}})
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = int(config["training"]["patience"])
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, aux = model(attr_features, text_features, snapshots, quality_features)

    metrics = []
    for split_name, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        row = compute_classification_metrics(logits, labels, indices, threshold=float(config["evaluation"]["decision_threshold"]))
        row.update({"split": split_name, "experiment": variant_name})
        metrics.append(row)

    checkpoint_path = Path(config["paths"]["checkpoint_dir"]) / f"{variant_name}.pt"
    torch.save(
        {
            "variant_name": variant_name,
            "variant": deepcopy(EXPERIMENT_VARIANTS[variant_name]),
            "state_dict": model.state_dict(),
            "history": history,
            "metrics": metrics,
            "selected_snapshot_dates": dataset.meta["selected_snapshot_dates"],
        },
        checkpoint_path,
    )
    metrics_frame = export_metrics_table(metrics, Path(config["paths"]["table_dir"]) / f"{variant_name}_metrics.csv")
    dump_json(history, Path(config["paths"]["log_dir"]) / f"{variant_name}_history.json")
    return TrainArtifacts(checkpoint_path=checkpoint_path, metrics_frame=metrics_frame, logits=logits.cpu(), aux={k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in aux.items()}, variant_name=variant_name)


def evaluate_checkpoint(config: dict[str, Any], dataset, checkpoint_path: str | Path) -> pd.DataFrame:
    device = torch.device(config["project"]["device"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    variant_name = checkpoint["variant_name"]
    model = _build_model(config, dataset, variant_name)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    attr_features = dataset.attr_features.to(device)
    text_features = dataset.text_features.to(device)
    quality_features = dataset.quality_features.to(device)
    labels = dataset.labels.to(device)
    snapshots = [_move_snapshot(snapshot, device) for snapshot in dataset.snapshots]

    with torch.no_grad():
        logits, aux = model(attr_features, text_features, snapshots, quality_features)

    metrics = []
    for split_name, indices in (("train", dataset.train_idx), ("val", dataset.val_idx), ("test", dataset.test_idx)):
        row = compute_classification_metrics(logits, labels, indices.to(device), threshold=float(config["evaluation"]["decision_threshold"]))
        row.update({"split": split_name, "experiment": variant_name})
        metrics.append(row)

    report_dir = Path(config["paths"]["output_dir"]) / variant_name
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = export_metrics_table(metrics, report_dir / "metrics.csv")
    export_prediction_table(dataset.user_table, dataset.labels, logits.cpu(), dataset.test_idx, "test", report_dir)
    export_prediction_table(dataset.user_table, dataset.labels, logits.cpu(), dataset.val_idx, "val", report_dir)

    modal_weights = aux.get("modal_weights")
    temporal_weights = aux.get("temporal_weights")
    if modal_weights is not None:
        export_modal_weight_statistics(dataset.user_table, dataset.labels, modal_weights.cpu(), report_dir)
    export_temporal_heatmap(
        temporal_weights.cpu() if isinstance(temporal_weights, torch.Tensor) else None,
        dataset.user_table,
        dataset.meta["selected_snapshot_dates"],
        report_dir,
        top_k=int(config["evaluation"]["top_temporal_cases"]),
    )
    if modal_weights is not None:
        export_error_analysis(
            dataset.user_table,
            dataset.labels,
            logits.cpu(),
            modal_weights.cpu(),
            temporal_weights.cpu() if isinstance(temporal_weights, torch.Tensor) else None,
            dataset.test_idx,
            report_dir,
        )
    return metrics_frame
