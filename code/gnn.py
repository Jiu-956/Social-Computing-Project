from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GNNExperimentOutput:
    name: str
    family: str
    metrics_rows: list[dict[str, float | str]]
    predictions: pd.DataFrame
    artifact_path: str
    best_val_f1: float


def run_gnn_experiment(
    config: PipelineConfig,
    name: str,
    family: str,
    users: pd.DataFrame,
    feature_frame: pd.DataFrame,
    graph_edges: pd.DataFrame,
    model_type: str,
    text_encoder: str = "none",
) -> GNNExperimentOutput:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    features = torch.tensor(feature_frame.to_numpy(dtype=np.float32), dtype=torch.float32)
    labels = torch.tensor(users["label_id"].to_numpy(dtype=np.int64), dtype=torch.long)

    masks = {
        "train": torch.tensor(users["split"].eq("train").to_numpy(), dtype=torch.bool),
        "val": torch.tensor(users["split"].eq("val").to_numpy(), dtype=torch.bool),
        "test": torch.tensor(users["split"].eq("test").to_numpy(), dtype=torch.bool),
    }

    adjacency, relation_adjacency = build_graph_tensors(users["user_id"].astype(str).tolist(), graph_edges)

    if model_type == "gcn":
        model = GCNClassifier(
            in_dim=features.shape[1],
            hidden_dim=config.gnn_hidden_dim,
            out_dim=2,
            dropout=config.gnn_dropout,
        )
    else:
        model = BotRGCNClassifier(
            in_dim=features.shape[1],
            hidden_dim=config.gnn_hidden_dim,
            out_dim=2,
            relation_names=sorted(relation_adjacency.keys()),
            dropout=config.gnn_dropout,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = features.to(device)
    labels = labels.to(device)
    adjacency = adjacency.to(device)
    relation_adjacency = {key: value.to(device) for key, value in relation_adjacency.items()}
    masks = {key: value.to(device) for key, value in masks.items()}
    model = model.to(device)

    train_labels = labels[masks["train"]]
    class_weights = compute_class_weights(train_labels.cpu().numpy())
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.gnn_learning_rate,
        weight_decay=config.gnn_weight_decay,
    )

    best_state = None
    best_val_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(config.gnn_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adjacency, relation_adjacency)
        loss = criterion(logits[masks["train"]], labels[masks["train"]])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(features, adjacency, relation_adjacency)
            probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            predictions = logits.argmax(dim=1).detach().cpu().numpy()
            val_metrics = evaluate_predictions(
                labels[masks["val"]].detach().cpu().numpy(),
                predictions[masks["val"].detach().cpu().numpy()],
                probabilities[masks["val"].detach().cpu().numpy()],
            )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.gnn_patience:
            LOGGER.info("%s early stopped at epoch %s", name, epoch + 1)
            break

    if best_state is None:
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(features, adjacency, relation_adjacency)
        probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32)
        predictions = logits.argmax(dim=1).detach().cpu().numpy().astype(int)

    val_mask = users["split"].eq("val").to_numpy()
    test_mask = users["split"].eq("test").to_numpy()
    val_metrics = evaluate_predictions(users.loc[val_mask, "label_id"].to_numpy(), predictions[val_mask], probabilities[val_mask])
    test_metrics = evaluate_predictions(
        users.loc[test_mask, "label_id"].to_numpy(),
        predictions[test_mask],
        probabilities[test_mask],
    )

    prediction_frame = users[["user_id", "split", "label", "label_id"]].copy()
    prediction_frame["experiment"] = name
    prediction_frame["family"] = family
    prediction_frame["model_type"] = model_type
    prediction_frame["text_encoder"] = text_encoder
    prediction_frame["graph_encoder"] = "gcn" if model_type == "gcn" else "botrgcn"
    prediction_frame["prediction"] = predictions
    prediction_frame["bot_probability"] = probabilities

    artifact_path = str(config.models_dir / f"{name}.pt")
    torch.save(best_state, artifact_path)

    metrics_rows = [
        {
            "experiment": name,
            "family": family,
            "model_type": model_type,
            "text_encoder": text_encoder,
            "graph_encoder": "gcn" if model_type == "gcn" else "botrgcn",
            "split": "val",
            **val_metrics,
        },
        {
            "experiment": name,
            "family": family,
            "model_type": model_type,
            "text_encoder": text_encoder,
            "graph_encoder": "gcn" if model_type == "gcn" else "botrgcn",
            "split": "test",
            **test_metrics,
        },
    ]
    return GNNExperimentOutput(
        name=name,
        family=family,
        metrics_rows=metrics_rows,
        predictions=prediction_frame,
        artifact_path=artifact_path,
        best_val_f1=best_val_f1,
    )


class GraphConvolution:
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        import torch.nn as nn

        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def __call__(self, adjacency, features):
        import torch

        support = torch.sparse.mm(adjacency, features)
        return self.linear(support)


def _graph_linear(in_dim: int, out_dim: int, bias: bool = True):
    import torch.nn as nn

    return nn.Linear(in_dim, out_dim, bias=bias)


def _sparse_mm(adjacency, features):
    import torch

    return torch.sparse.mm(adjacency, features)


def _relu(value):
    import torch.nn.functional as F

    return F.relu(value)


def _dropout(value, p: float, training: bool):
    import torch.nn.functional as F

    return F.dropout(value, p=p, training=training)


def _module_dict(items: dict[str, tuple[int, int]]):
    import torch.nn as nn

    return nn.ModuleDict({key: _graph_linear(in_dim, out_dim, bias=False) for key, (in_dim, out_dim) in items.items()})


class GCNClassifier:
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        import torch.nn as nn

        self.gc1 = GraphConvolution(in_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, out_dim)
        self.dropout = dropout
        self.training = True
        self._module = nn.Module()
        self._module.gc1 = self.gc1.linear
        self._module.gc2 = self.gc2.linear

    def to(self, device):
        self._module.to(device)
        return self

    def parameters(self):
        return self._module.parameters()

    def train(self):
        self.training = True
        self._module.train()

    def eval(self):
        self.training = False
        self._module.eval()

    def state_dict(self):
        return self._module.state_dict()

    def load_state_dict(self, state_dict):
        self._module.load_state_dict(state_dict)

    def __call__(self, features, adjacency, relation_adjacency):
        hidden = self.gc1(adjacency, features)
        hidden = _relu(hidden)
        hidden = _dropout(hidden, self.dropout, self.training)
        return self.gc2(adjacency, hidden)


class BotRGCNClassifier:
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, relation_names: list[str], dropout: float):
        import torch.nn as nn

        self.dropout = dropout
        self.training = True
        self.self_linear1 = _graph_linear(in_dim, hidden_dim, bias=False)
        self.relation_linear1 = _module_dict({name: (in_dim, hidden_dim) for name in relation_names})
        self.self_linear2 = _graph_linear(hidden_dim, out_dim, bias=False)
        self.relation_linear2 = _module_dict({name: (hidden_dim, out_dim) for name in relation_names})
        self.bias1 = nn.Parameter(torch_zeros(hidden_dim))
        self.bias2 = nn.Parameter(torch_zeros(out_dim))
        self._module = nn.Module()
        self._module.self_linear1 = self.self_linear1
        self._module.relation_linear1 = self.relation_linear1
        self._module.self_linear2 = self.self_linear2
        self._module.relation_linear2 = self.relation_linear2
        self._module.bias1 = self.bias1
        self._module.bias2 = self.bias2

    def to(self, device):
        self._module.to(device)
        return self

    def parameters(self):
        return self._module.parameters()

    def train(self):
        self.training = True
        self._module.train()

    def eval(self):
        self.training = False
        self._module.eval()

    def state_dict(self):
        return self._module.state_dict()

    def load_state_dict(self, state_dict):
        self._module.load_state_dict(state_dict)

    def __call__(self, features, adjacency, relation_adjacency):
        hidden = self.self_linear1(features)
        for relation_name, relation_matrix in relation_adjacency.items():
            hidden = hidden + self.relation_linear1[relation_name](_sparse_mm(relation_matrix, features))
        hidden = hidden + self.bias1
        hidden = _relu(hidden)
        hidden = _dropout(hidden, self.dropout, self.training)

        logits = self.self_linear2(hidden)
        for relation_name, relation_matrix in relation_adjacency.items():
            logits = logits + self.relation_linear2[relation_name](_sparse_mm(relation_matrix, hidden))
        logits = logits + self.bias2
        return logits


def torch_zeros(size: int):
    import torch

    return torch.zeros(size, dtype=torch.float32)


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (2.0 * counts)
    return weights.astype(np.float32)


def build_graph_tensors(user_ids: list[str], graph_edges: pd.DataFrame):
    import torch

    index_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    undirected_entries: dict[tuple[int, int], float] = defaultdict(float)
    relation_entries: dict[str, dict[tuple[int, int], float]] = defaultdict(lambda: defaultdict(float))

    for row in graph_edges.itertuples(index=False):
        source_id = str(row.source_id)
        target_id = str(row.target_id)
        if source_id not in index_map or target_id not in index_map:
            continue
        source_idx = index_map[source_id]
        target_idx = index_map[target_id]
        weight = float(row.weight)
        undirected_entries[(source_idx, target_idx)] += weight
        undirected_entries[(target_idx, source_idx)] += weight

        relations = [part.strip() for part in str(getattr(row, "relations", "")).split("|") if part.strip()]
        if not relations:
            relations = ["social"]
        for relation_name in relations:
            relation_entries[relation_name][(source_idx, target_idx)] += weight
            relation_entries[relation_name][(target_idx, source_idx)] += weight

    adjacency = normalize_sparse_matrix(to_sparse_matrix(undirected_entries, len(user_ids)))
    relation_tensors = {
        relation_name: normalize_sparse_matrix(to_sparse_matrix(entries, len(user_ids)))
        for relation_name, entries in relation_entries.items()
    }
    if not relation_tensors:
        relation_tensors = {"social": adjacency}
    return scipy_to_torch_sparse(adjacency), {key: scipy_to_torch_sparse(value) for key, value in relation_tensors.items()}


def to_sparse_matrix(entries: dict[tuple[int, int], float], size: int) -> sparse.coo_matrix:
    if not entries:
        return sparse.eye(size, dtype=np.float32, format="coo")
    rows = np.fromiter((key[0] for key in entries.keys()), dtype=np.int32)
    cols = np.fromiter((key[1] for key in entries.keys()), dtype=np.int32)
    data = np.fromiter(entries.values(), dtype=np.float32)
    return sparse.coo_matrix((data, (rows, cols)), shape=(size, size), dtype=np.float32)


def normalize_sparse_matrix(matrix: sparse.coo_matrix) -> sparse.coo_matrix:
    matrix = matrix.tocsr()
    matrix = matrix + sparse.eye(matrix.shape[0], dtype=np.float32, format="csr")
    degrees = np.asarray(matrix.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0
    inv_sqrt = np.power(degrees, -0.5)
    d_mat = sparse.diags(inv_sqrt)
    normalized = d_mat @ matrix @ d_mat
    return normalized.tocoo()


def scipy_to_torch_sparse(matrix: sparse.coo_matrix):
    import torch

    matrix = matrix.tocoo()
    indices = torch.tensor(np.vstack([matrix.row, matrix.col]), dtype=torch.long)
    values = torch.tensor(matrix.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=matrix.shape).coalesce()


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0,
    }
