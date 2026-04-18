from __future__ import annotations

import math

import numpy as np
import torch


def _auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    pos = int(pos_mask.sum())
    neg = int(neg_mask.sum())
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        ranks[start:end] = average_rank
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    sum_pos = original_ranks[pos_mask].sum()
    return float((sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def compute_classification_metrics(logits: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probabilities = torch.softmax(logits[indices], dim=1)[:, 1].detach().cpu().numpy()
    y_true = labels[indices].detach().cpu().numpy()
    y_pred = (probabilities >= threshold).astype(np.int64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)
    auc_roc = _auc_roc(y_true, probabilities)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(auc_roc if not math.isnan(auc_roc) else np.nan),
    }
