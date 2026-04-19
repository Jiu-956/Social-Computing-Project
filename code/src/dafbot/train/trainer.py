from __future__ import annotations

import time
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


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _build_checkpoint_payload(
    variant_name: str,
    state_dict: dict[str, torch.Tensor],
    history: list[dict[str, Any]],
    selected_snapshot_dates: list[str],
    metrics: list[dict[str, Any]] | None = None,
    *,
    epoch: int | None = None,
    best_val_f1: float | None = None,
    is_partial: bool = False,
) -> dict[str, Any]:
    payload = {
        "variant_name": variant_name,
        "variant": deepcopy(EXPERIMENT_VARIANTS[variant_name]),
        "state_dict": state_dict,
        "history": history,
        "selected_snapshot_dates": selected_snapshot_dates,
        "is_partial": is_partial,
    }
    if metrics is not None:
        payload["metrics"] = metrics
    if epoch is not None:
        payload["epoch"] = epoch
    if best_val_f1 is not None:
        payload["best_val_f1"] = float(best_val_f1)
    return payload


def _write_partial_training_state(
    checkpoint_dir: Path,
    log_dir: Path,
    variant_name: str,
    state_dict: dict[str, torch.Tensor],
    history: list[dict[str, Any]],
    selected_snapshot_dates: list[str],
    *,
    epoch: int,
    best_val_f1: float,
    write_best: bool,
) -> None:
    latest_checkpoint_path = checkpoint_dir / f"{variant_name}.latest.pt"
    torch.save(
        _build_checkpoint_payload(
            variant_name=variant_name,
            state_dict=state_dict,
            history=history,
            selected_snapshot_dates=selected_snapshot_dates,
            epoch=epoch,
            best_val_f1=best_val_f1,
            is_partial=True,
        ),
        latest_checkpoint_path,
    )
    dump_json(history, log_dir / f"{variant_name}_history.json")
    if write_best:
        best_checkpoint_path = checkpoint_dir / f"{variant_name}.pt"
        torch.save(
            _build_checkpoint_payload(
                variant_name=variant_name,
                state_dict=state_dict,
                history=history,
                selected_snapshot_dates=selected_snapshot_dates,
                epoch=epoch,
                best_val_f1=best_val_f1,
                is_partial=True,
            ),
            best_checkpoint_path,
        )


def _print_epoch_start(epoch: int, total_epochs: int) -> None:
    print(f"Epoch {epoch:03d}/{total_epochs:03d} started...", flush=True)


def _print_epoch_stage(epoch: int, total_epochs: int, stage: str) -> None:
    print(f"Epoch {epoch:03d}/{total_epochs:03d} | {stage}", flush=True)


def _print_training_progress(
    epoch: int,
    total_epochs: int,
    loss: float,
    val_metrics: dict[str, float],
    best_val_f1: float,
    patience_left: int,
    improved: bool,
    epoch_seconds: float,
    elapsed_seconds: float,
) -> None:
    status = "improved" if improved else "no_improve"
    print(
        " | ".join(
            [
                f"Epoch {epoch:03d}/{total_epochs:03d}",
                f"loss={loss:.4f}",
                f"val_f1={val_metrics['f1']:.4f}",
                f"val_acc={val_metrics['accuracy']:.4f}",
                f"val_auc={val_metrics['auc_roc']:.4f}",
                f"best_f1={best_val_f1:.4f}",
                f"patience_left={patience_left}",
                f"epoch_time={_format_duration(epoch_seconds)}",
                f"elapsed={_format_duration(elapsed_seconds)}",
                status,
            ]
        ),
        flush=True,
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
    total_epochs = int(config["training"]["epochs"])
    total_patience = int(config["training"]["patience"])
    log_every = max(1, int(config["training"].get("log_every", 1)))
    save_every = max(1, int(config["training"].get("save_every", 1)))
    patience_left = total_patience
    history: list[dict[str, Any]] = []
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    log_dir = Path(config["paths"]["log_dir"])
    selected_snapshot_dates = dataset.meta["selected_snapshot_dates"]
    training_start = time.perf_counter()

    print(
        " | ".join(
            [
                f"Training {variant_name}",
                f"device={device}",
                f"epochs={total_epochs}",
                f"patience={total_patience}",
                f"log_every={log_every}",
                f"save_every={save_every}",
                f"train={int(train_idx.numel())}",
                f"val={int(val_idx.numel())}",
                f"test={int(test_idx.numel())}",
                f"snapshots={len(snapshots)}",
            ]
        ),
        flush=True,
    )

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.perf_counter()
        if epoch == 1 or epoch % log_every == 0:
            _print_epoch_start(epoch, total_epochs)
        model.train()
        optimizer.zero_grad()
        if epoch == 1:
            _print_epoch_stage(epoch, total_epochs, "train_forward")
        logits, aux = model(attr_features, text_features, snapshots, quality_features)
        loss, loss_terms = build_total_loss(
            logits=logits,
            labels=labels,
            indices=train_idx,
            aux=aux,
            lambda_modal=float(config["training"]["lambda_modal"]),
            lambda_temporal=float(config["training"]["lambda_temporal"]),
        )
        if epoch == 1:
            _print_epoch_stage(epoch, total_epochs, "backward")
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if epoch == 1:
                _print_epoch_stage(epoch, total_epochs, "val_forward")
            val_logits, _ = model(attr_features, text_features, snapshots, quality_features)
            val_metrics = compute_classification_metrics(val_logits, labels, val_idx, threshold=float(config["evaluation"]["decision_threshold"]))

        history.append({"epoch": epoch, **loss_terms, **{f"val_{key}": value for key, value in val_metrics.items()}})
        improved = val_metrics["f1"] > best_val_f1
        if improved:
            best_val_f1 = val_metrics["f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = total_patience
        else:
            patience_left -= 1

        should_save = improved or epoch == 1 or epoch == total_epochs or epoch % save_every == 0 or patience_left <= 1
        if should_save:
            latest_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            state_to_save = best_state if improved and best_state is not None else latest_state
            _write_partial_training_state(
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
                variant_name=variant_name,
                state_dict=state_to_save,
                history=history,
                selected_snapshot_dates=selected_snapshot_dates,
                epoch=epoch,
                best_val_f1=best_val_f1,
                write_best=improved,
            )

        epoch_seconds = time.perf_counter() - epoch_start
        elapsed_seconds = time.perf_counter() - training_start
        should_log = improved or epoch == 1 or epoch == total_epochs or epoch % log_every == 0 or patience_left <= 1
        if should_log:
            _print_training_progress(
                epoch=epoch,
                total_epochs=total_epochs,
                loss=float(loss.item()),
                val_metrics=val_metrics,
                best_val_f1=best_val_f1,
                patience_left=patience_left,
                improved=improved,
                epoch_seconds=epoch_seconds,
                elapsed_seconds=elapsed_seconds,
            )
        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch:03d}. Best validation F1 = {best_val_f1:.4f}", flush=True)
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

    checkpoint_path = checkpoint_dir / f"{variant_name}.pt"
    torch.save(
        _build_checkpoint_payload(
            variant_name=variant_name,
            state_dict=model.state_dict(),
            history=history,
            selected_snapshot_dates=selected_snapshot_dates,
            metrics=metrics,
            epoch=len(history),
            best_val_f1=best_val_f1,
            is_partial=False,
        ),
        checkpoint_path,
    )
    metrics_frame = export_metrics_table(metrics, Path(config["paths"]["table_dir"]) / f"{variant_name}_metrics.csv")
    dump_json(history, log_dir / f"{variant_name}_history.json")
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
