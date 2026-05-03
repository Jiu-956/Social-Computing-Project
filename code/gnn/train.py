from __future__ import annotations

import logging
import random as _random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import ProjectConfig

LOGGER = logging.getLogger(__name__)

torch.manual_seed(42)
_random.seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@dataclass(slots=True)
class GNNResult:
    metrics_rows: list[dict[str, float | str]]
    predictions: pd.DataFrame
    best_val_f1: float
    artifact_path: Path
    training_history: pd.DataFrame


def _scaled_tensor(users: pd.DataFrame, columns: list[str]) -> torch.Tensor:
    frame = users.reindex(columns=columns, fill_value=0.0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    matrix = frame.to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    matrix = scaler.fit_transform(matrix).astype(np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.tensor(matrix, dtype=torch.float32)


def _split_model_output(model_output: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(model_output, tuple):
        logits = model_output[0]
        aux_loss = model_output[1] if len(model_output) > 1 else None
        if isinstance(aux_loss, torch.Tensor):
            return logits, aux_loss
        return logits, None
    return model_output, None


def _find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> float:
    best_f1 = -1.0
    best_th = 0.5
    for th in thresholds:
        preds = (probabilities >= th).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
        f1_val = float(f1)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_th = float(th)
    return best_th


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc_roc = roc_auc_score(y_true, probabilities)
    except ValueError:
        auc_roc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(auc_roc),
    }


def _train_gnn_model(
    config: ProjectConfig,
    name: str,
    users: pd.DataFrame,
    model: nn.Module,
    description_tensor: torch.Tensor,
    tweet_tensor: torch.Tensor,
    num_prop_tensor: torch.Tensor,
    cat_prop_tensor: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    test_indices: torch.Tensor,
    edge_index: dict[str, Any] | torch.Tensor,
    edge_type: torch.Tensor | None,
) -> GNNResult:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.gnn_learning_rate,
        weight_decay=config.gnn_weight_decay,
    )
    total_steps = config.gnn_epochs
    warmup_steps = min(5, total_steps // 5)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.01, ((total_steps - step) / (total_steps - warmup_steps)) ** 0.5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_labels = labels[train_indices].numpy()
    class_counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32), label_smoothing=0.1)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0
    patience_left = config.gnn_patience
    history_rows: list[dict[str, float | int | str]] = []

    for epoch in range(1, config.gnn_epochs + 1):
        model.train()
        optimizer.zero_grad()
        raw_output = model(description_tensor, tweet_tensor, num_prop_tensor, cat_prop_tensor, edge_index, edge_type)
        logits, aux_loss = _split_model_output(raw_output)
        loss = criterion(logits[train_indices], labels[train_indices])
        if aux_loss is not None:
            loss = loss + aux_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            raw_output = model(description_tensor, tweet_tensor, num_prop_tensor, cat_prop_tensor, edge_index, edge_type)
            logits, _ = _split_model_output(raw_output)
            train_probs = torch.softmax(logits[train_indices], dim=1)[:, 1].cpu().numpy()
            train_preds = (train_probs >= 0.5).astype(int)
            train_true = labels[train_indices].cpu().numpy()
            train_metrics = _compute_metrics(train_true, train_preds, train_probs)

            val_logits = logits[val_indices]
            val_probs = torch.softmax(logits[val_indices], dim=1)[:, 1].cpu().numpy()
            val_preds = (val_probs >= 0.5).astype(int)
            val_true = labels[val_indices].cpu().numpy()
            val_metrics = _compute_metrics(val_true, val_preds, val_probs)
            val_loss = float(criterion(val_logits, labels[val_indices]).detach().cpu())
            val_f1 = val_metrics["f1"]

        history_rows.append(
            {
                "experiment": name,
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_f1,
                "best_val_f1": best_val_f1 if best_val_f1 >= 0 else float("nan"),
                "patience_left": patience_left,
            }
        )

        LOGGER.info(
            "[%s] epoch %03d/%03d train_loss=%.4f val_loss=%.4f train_f1=%.4f val_f1=%.4f best_val_f1=%.4f patience_left=%d",
            name,
            epoch,
            config.gnn_epochs,
            float(loss.detach().cpu()),
            val_loss,
            train_metrics["f1"],
            val_f1,
            best_val_f1 if best_val_f1 >= 0 else val_f1,
            patience_left,
        )

        if float(val_f1) > best_val_f1:
            best_val_f1 = float(val_f1)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = config.gnn_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        raw_output = model(description_tensor, tweet_tensor, num_prop_tensor, cat_prop_tensor, edge_index, edge_type)
        logits, _ = _split_model_output(raw_output)

    # Search for better threshold on val, but only apply if it improves F1 by >0.005 vs 0.5
    search_thresholds = np.linspace(0.2, 0.8, 61)
    val_probs = torch.softmax(logits[val_indices], dim=1)[:, 1].cpu().numpy()
    val_true = labels[val_indices].cpu().numpy()
    best_th = _find_best_threshold(val_true, val_probs, search_thresholds)
    test_probs = torch.softmax(logits[test_indices], dim=1)[:, 1].cpu().numpy()
    test_true = labels[test_indices].cpu().numpy()

    # Compute F1 at default 0.5 threshold vs best threshold on val
    val_preds_default = (val_probs >= 0.5).astype(int)
    _, _, val_f1_default, _ = precision_recall_fscore_support(val_true, val_preds_default, average="binary", zero_division=0)
    val_f1_default = float(val_f1_default)
    val_preds_best = (val_probs >= best_th).astype(int)
    _, _, val_f1_best, _ = precision_recall_fscore_support(val_true, val_preds_best, average="binary", zero_division=0)
    val_f1_best = float(val_f1_best)

    # Only use the found threshold if it gives meaningful improvement
    if val_f1_best - val_f1_default > 0.005:
        use_th = best_th
    else:
        use_th = 0.5

    LOGGER.info(
        "[%s] threshold search val_f1@0.5=%.4f val_f1@best=%.4f chosen_threshold=%.3f",
        name,
        val_f1_default,
        val_f1_best,
        use_th,
    )

    metrics_rows = []
    prediction_rows = []
    for split_name, split_probs, split_true, split_indices in (
        ("val", val_probs, val_true, val_indices),
        ("test", test_probs, test_true, test_indices),
    ):
        split_preds = (split_probs >= use_th).astype(int)
        metrics_rows.append(
            {
                "experiment": name,
                "family": "feature_text_graph",
                "split": split_name,
                "best_threshold": float(best_th),
                "selected_threshold": float(use_th),
                **_compute_metrics(split_true, split_preds, split_probs),
            }
        )
        user_subset = users.iloc[split_indices.cpu().numpy()]
        for user_id, true_label, pred_label, probability in zip(
            user_subset["user_id"].tolist(),
            split_true.tolist(),
            split_preds.tolist(),
            split_probs.tolist(),
        ):
            prediction_rows.append(
                {
                    "experiment": name,
                    "family": "feature_text_graph",
                    "split": split_name,
                    "user_id": user_id,
                    "true_label": int(true_label),
                    "pred_label": int(pred_label),
                    "bot_probability": float(probability),
                }
            )

    artifact_path = config.models_dir / f"{name}.pt"
    torch.save({"state_dict": model.state_dict(), "best_val_f1": best_val_f1}, artifact_path)
    history_frame = pd.DataFrame(history_rows)
    _save_training_curve(history_frame, config.figures_dir / f"{name}_training_curve.png", name)
    history_frame.to_csv(config.tables_dir / f"{name}_training_history.csv", index=False)
    LOGGER.info(
        "[%s] final_val_f1=%.4f final_test_f1=%.4f saved_curve=%s",
        name,
        val_f1_best if use_th == best_th else val_f1_default,
        _compute_metrics(test_true, (test_probs >= use_th).astype(int), test_probs)["f1"],
        config.figures_dir / f"{name}_training_curve.png",
    )
    return GNNResult(
        metrics_rows=metrics_rows,
        predictions=pd.DataFrame(prediction_rows),
        best_val_f1=best_val_f1,
        artifact_path=artifact_path,
        training_history=history_frame,
    )


def _save_training_curve(history: pd.DataFrame, output_path: Path, title: str) -> None:
    if history.empty:
        return

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = history["epoch"].to_numpy(dtype=int)

    axes[0].plot(epochs, history["train_loss"], label="train loss", color="#4472c4")
    axes[0].plot(epochs, history["val_loss"], label="val loss", color="#ed7d31")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, history["train_f1"], label="train F1", color="#2f855a")
    axes[1].plot(epochs, history["val_f1"], label="val F1", color="#9b2c2c")
    axes[1].set_title("F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    figure.suptitle(f"Training Curve - {title}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
