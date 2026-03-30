from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from torch_geometric.nn import GATConv, GCNConv, RGCNConv

    TORCH_GEOMETRIC_AVAILABLE = True
    TORCH_GEOMETRIC_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on local environment
    GATConv = None
    GCNConv = None
    RGCNConv = None
    TORCH_GEOMETRIC_AVAILABLE = False
    TORCH_GEOMETRIC_IMPORT_ERROR = exc

from .config import ProjectConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GNNResult:
    metrics_rows: list[dict[str, float | str]]
    predictions: pd.DataFrame
    best_val_f1: float
    artifact_path: Path


class _FeatureTextGraphBase(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        description_dim: int,
        tweet_dim: int,
        num_prop_dim: int,
        cat_prop_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        block_dim = max(4, hidden_dim // 4)
        self.description_mlp = nn.Sequential(nn.Linear(description_dim, block_dim), nn.LeakyReLU())
        self.tweet_mlp = nn.Sequential(nn.Linear(tweet_dim, block_dim), nn.LeakyReLU())
        self.num_mlp = nn.Sequential(nn.Linear(num_prop_dim, block_dim), nn.LeakyReLU())
        self.cat_mlp = nn.Sequential(nn.Linear(cat_prop_dim, block_dim), nn.LeakyReLU())
        self.input_mlp = nn.Sequential(nn.Linear(block_dim * 4, hidden_dim), nn.LeakyReLU())
        self.output_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU())
        self.output_head = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def encode_inputs(
        self,
        description: torch.Tensor,
        tweet: torch.Tensor,
        num_prop: torch.Tensor,
        cat_prop: torch.Tensor,
    ) -> torch.Tensor:
        d = self.description_mlp(description)
        t = self.tweet_mlp(tweet)
        n = self.num_mlp(num_prop)
        c = self.cat_mlp(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)
        x = self.dropout(x)
        return self.input_mlp(x)


class FeatureTextGraphGCN(_FeatureTextGraphBase):
    def __init__(
        self,
        hidden_dim: int,
        description_dim: int,
        tweet_dim: int,
        num_prop_dim: int,
        cat_prop_dim: int,
        dropout: float,
    ) -> None:
        super().__init__(hidden_dim, description_dim, tweet_dim, num_prop_dim, cat_prop_dim, dropout)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

    def forward(
        self,
        description: torch.Tensor,
        tweet: torch.Tensor,
        num_prop: torch.Tensor,
        cat_prop: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.encode_inputs(description, tweet, num_prop, cat_prop)
        x = self.gcn1(x, edge_index)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)
        x = self.output_mlp(x)
        return self.output_head(x)


class FeatureTextGraphGAT(_FeatureTextGraphBase):
    def __init__(
        self,
        hidden_dim: int,
        description_dim: int,
        tweet_dim: int,
        num_prop_dim: int,
        cat_prop_dim: int,
        dropout: float,
    ) -> None:
        super().__init__(hidden_dim, description_dim, tweet_dim, num_prop_dim, cat_prop_dim, dropout)
        self.gat1 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout)

    def forward(
        self,
        description: torch.Tensor,
        tweet: torch.Tensor,
        num_prop: torch.Tensor,
        cat_prop: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.encode_inputs(description, tweet, num_prop, cat_prop)
        x = self.gat1(x, edge_index)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = self.output_mlp(x)
        return self.output_head(x)


class FeatureTextGraphBotRGCN(_FeatureTextGraphBase):
    def __init__(
        self,
        hidden_dim: int,
        description_dim: int,
        tweet_dim: int,
        num_prop_dim: int,
        cat_prop_dim: int,
        dropout: float,
        relation_count: int,
    ) -> None:
        super().__init__(hidden_dim, description_dim, tweet_dim, num_prop_dim, cat_prop_dim, dropout)
        self.rgcn1 = RGCNConv(hidden_dim, hidden_dim, num_relations=relation_count)
        self.rgcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations=relation_count)

    def forward(
        self,
        description: torch.Tensor,
        tweet: torch.Tensor,
        num_prop: torch.Tensor,
        cat_prop: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_type is None:
            raise ValueError("BotRGCN requires edge_type.")
        x = self.encode_inputs(description, tweet, num_prop, cat_prop)
        x = self.rgcn1(x, edge_index, edge_type)
        x = self.dropout(x)
        x = self.rgcn2(x, edge_index, edge_type)
        x = self.output_mlp(x)
        return self.output_head(x)


def run_graph_neural_models(
    config: ProjectConfig,
    users: pd.DataFrame,
    graph_edges: pd.DataFrame,
    description_columns: list[str],
    tweet_columns: list[str],
    num_property_columns: list[str],
    cat_property_columns: list[str],
) -> list[GNNResult]:
    if users.empty:
        return []
    if not TORCH_GEOMETRIC_AVAILABLE:
        LOGGER.warning(
            "Skipping GCN/GAT/BotRGCN experiments because torch_geometric is unavailable: %s",
            TORCH_GEOMETRIC_IMPORT_ERROR,
        )
        return []

    users = users.copy()
    users[description_columns + tweet_columns + num_property_columns + cat_property_columns] = (
        users[description_columns + tweet_columns + num_property_columns + cat_property_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    description_tensor = _scaled_tensor(users, description_columns)
    tweet_tensor = _scaled_tensor(users, tweet_columns)
    num_prop_tensor = _scaled_tensor(users, num_property_columns)
    cat_prop_tensor = torch.tensor(users[cat_property_columns].to_numpy(dtype=np.float32), dtype=torch.float32)

    labels = torch.tensor(users["label_id"].clip(lower=0).to_numpy(dtype=np.int64), dtype=torch.long)
    train_indices = torch.tensor(np.flatnonzero(((users["split"] == "train") & (users["label_id"] >= 0)).to_numpy()), dtype=torch.long)
    val_indices = torch.tensor(np.flatnonzero(((users["split"] == "val") & (users["label_id"] >= 0)).to_numpy()), dtype=torch.long)
    test_indices = torch.tensor(np.flatnonzero(((users["split"] == "test") & (users["label_id"] >= 0)).to_numpy()), dtype=torch.long)

    user_ids = users["user_id"].tolist()
    id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    combined_edge_index = _build_combined_edge_index(id_to_index, graph_edges)
    relation_edge_index, relation_edge_type = _build_relation_graph(id_to_index, graph_edges)

    outputs = []
    model_specs = [
        (
            "feature_text_graph_gcn",
            FeatureTextGraphGCN(
                hidden_dim=config.gnn_hidden_dim,
                description_dim=description_tensor.shape[1],
                tweet_dim=tweet_tensor.shape[1],
                num_prop_dim=num_prop_tensor.shape[1],
                cat_prop_dim=cat_prop_tensor.shape[1],
                dropout=config.gnn_dropout,
            ),
            combined_edge_index,
            None,
        ),
        (
            "feature_text_graph_gat",
            FeatureTextGraphGAT(
                hidden_dim=config.gnn_hidden_dim,
                description_dim=description_tensor.shape[1],
                tweet_dim=tweet_tensor.shape[1],
                num_prop_dim=num_prop_tensor.shape[1],
                cat_prop_dim=cat_prop_tensor.shape[1],
                dropout=config.gnn_dropout,
            ),
            combined_edge_index,
            None,
        ),
        (
            "feature_text_graph_botrgcn",
            FeatureTextGraphBotRGCN(
                hidden_dim=config.gnn_hidden_dim,
                description_dim=description_tensor.shape[1],
                tweet_dim=tweet_tensor.shape[1],
                num_prop_dim=num_prop_tensor.shape[1],
                cat_prop_dim=cat_prop_tensor.shape[1],
                dropout=config.gnn_dropout,
                relation_count=2,
            ),
            relation_edge_index,
            relation_edge_type,
        ),
    ]

    for name, model, edge_index, edge_type in model_specs:
        LOGGER.info("Running graph neural model: %s", name)
        outputs.append(
            _train_gnn_model(
                config=config,
                name=name,
                users=users,
                model=model,
                description_tensor=description_tensor,
                tweet_tensor=tweet_tensor,
                num_prop_tensor=num_prop_tensor,
                cat_prop_tensor=cat_prop_tensor,
                labels=labels,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
                edge_index=edge_index,
                edge_type=edge_type,
            )
        )
    return outputs


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
    edge_index: torch.Tensor,
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
        logits = model(description_tensor, tweet_tensor, num_prop_tensor, cat_prop_tensor, edge_index, edge_type)
        loss = criterion(logits[train_indices], labels[train_indices])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(description_tensor, tweet_tensor, num_prop_tensor, cat_prop_tensor, edge_index, edge_type)
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
        logits = model(description_tensor, tweet_tensor, num_prop_tensor, cat_prop_tensor, edge_index, edge_type)

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


def _scaled_tensor(users: pd.DataFrame, columns: list[str]) -> torch.Tensor:
    frame = users.reindex(columns=columns, fill_value=0.0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    matrix = frame.to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    matrix = scaler.fit_transform(matrix).astype(np.float32)
    return torch.tensor(matrix, dtype=torch.float32)


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
