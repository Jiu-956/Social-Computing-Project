from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ..config import ProjectConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GNNResult:
    metrics_rows: list[dict[str, float | str]]
    predictions: pd.DataFrame
    best_val_f1: float
    artifact_path: Path


def _scaled_tensor(users: pd.DataFrame, columns: list[str]) -> torch.Tensor:
    frame = users.reindex(columns=columns, fill_value=0.0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    matrix = frame.to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    matrix = scaler.fit_transform(matrix).astype(np.float32)
    return torch.tensor(matrix, dtype=torch.float32)


def _split_model_output(model_output: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(model_output, tuple):
        logits = model_output[0]
        aux_loss = model_output[1] if len(model_output) > 1 else None
        if isinstance(aux_loss, torch.Tensor):
            return logits, aux_loss
        return logits, None
    return model_output, None


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
    train_labels = labels[train_indices].numpy()
    class_counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))

    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0
    patience_left = config.gnn_patience

    for _ in range(config.gnn_epochs):
        model.train()
        optimizer.zero_grad()
        raw_output = model(description_tensor, tweet_tensor, num_prop_tensor, cat_prop_tensor, edge_index, edge_type)
        logits, aux_loss = _split_model_output(raw_output)
        loss = criterion(logits[train_indices], labels[train_indices])
        if aux_loss is not None:
            loss = loss + aux_loss
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            raw_output = model(description_tensor, tweet_tensor, num_prop_tensor, cat_prop_tensor, edge_index, edge_type)
            logits, _ = _split_model_output(raw_output)
            val_probs = torch.softmax(logits[val_indices], dim=1)[:, 1].cpu().numpy()
            val_preds = (val_probs >= 0.5).astype(int)
            val_true = labels[val_indices].cpu().numpy()
            _, _, val_f1, _ = precision_recall_fscore_support(val_true, val_preds, average="binary", zero_division=0)

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

    metrics_rows: list[dict[str, float | str]] = []
    prediction_rows: list[dict[str, float | str | int]] = []
    for split_name, indices in (("val", val_indices), ("test", test_indices)):
        split_probs = torch.softmax(logits[indices], dim=1)[:, 1].cpu().numpy()
        split_preds = (split_probs >= 0.5).astype(int)
        split_true = labels[indices].cpu().numpy()
        metrics_rows.append(
            {
                "experiment": name,
                "family": "feature_text_graph",
                "split": split_name,
                **_compute_metrics(split_true, split_preds, split_probs),
            }
        )
        user_subset = users.iloc[indices.cpu().numpy()]
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
    return GNNResult(
        metrics_rows=metrics_rows,
        predictions=pd.DataFrame(prediction_rows),
        best_val_f1=best_val_f1,
        artifact_path=artifact_path,
    )
