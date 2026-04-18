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

try:
    from torch_geometric.nn import RGCNConv

    TORCH_GEOMETRIC_AVAILABLE = True
    TORCH_GEOMETRIC_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on local environment
    RGCNConv = None
    TORCH_GEOMETRIC_AVAILABLE = False
    TORCH_GEOMETRIC_IMPORT_ERROR = exc

from .config import ProjectConfig

LOGGER = logging.getLogger(__name__)

MODALITY_NAMES = ("feature", "text", "graph")
PRIMARY_METHOD_NAME = "modality_reliability_adaptive_fusion"
PRIMARY_METHOD_FAMILY = "adaptive_dynamic_fusion"


@dataclass(slots=True)
class GNNResult:
    metrics_rows: list[dict[str, float | str]]
    predictions: pd.DataFrame
    best_val_f1: float
    artifact_path: Path


class ReliabilityAwareFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        quality_dim: int,
        quality_feature_names: list[str],
        gate_hidden_dim: int,
        dropout: float,
        temperature: float,
    ) -> None:
        super().__init__()
        gate_hidden_dim = max(8, gate_hidden_dim)
        attention_heads = 4 if hidden_dim % 4 == 0 and hidden_dim >= 16 else 1
        self.gate_network = nn.Sequential(
            nn.Linear(quality_dim, gate_hidden_dim),
            nn.LayerNorm(gate_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, len(MODALITY_NAMES)),
        )
        self.modality_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.temperature = max(float(temperature), 1e-3)
        self.prior_scale = 0.6
        self.quality_positions = {name: index for index, name in enumerate(quality_feature_names)}

    def forward(
        self,
        feature_repr: torch.Tensor,
        text_repr: torch.Tensor,
        graph_repr: torch.Tensor,
        quality: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        modality_stack = torch.stack((feature_repr, text_repr, graph_repr), dim=1)
        gate_logits = (self.gate_network(quality) + self.prior_scale * self._build_quality_prior(quality)) / self.temperature
        gate_weights = torch.softmax(gate_logits, dim=1)

        weighted_sum = (modality_stack * gate_weights.unsqueeze(-1)).sum(dim=1)
        attention_output, attention_weights = self.modality_attention(
            query=weighted_sum.unsqueeze(1),
            key=modality_stack,
            value=modality_stack,
            need_weights=True,
            average_attn_weights=False,
        )
        fused = self.output_norm(weighted_sum + self.dropout(attention_output.squeeze(1)))
        diagnostics = {
            "gate_weights": gate_weights,
            "attention_weights": attention_weights.mean(dim=1).squeeze(1),
        }
        return fused, diagnostics

    def _quality_column(self, quality: torch.Tensor, name: str) -> torch.Tensor:
        position = self.quality_positions.get(name)
        if position is None:
            return torch.zeros(quality.shape[0], dtype=quality.dtype, device=quality.device)
        return quality[:, position]

    def _build_quality_prior(self, quality: torch.Tensor) -> torch.Tensor:
        text_richness = self._quality_column(quality, "quality_text_richness")
        text_completeness = self._quality_column(quality, "quality_text_completeness")
        graph_connectivity = self._quality_column(quality, "quality_graph_connectivity")
        graph_reciprocity = self._quality_column(quality, "quality_graph_reciprocity")
        feature_completeness = self._quality_column(quality, "quality_feature_completeness")
        profile_anomaly = self._quality_column(quality, "quality_profile_anomaly")

        feature_prior = 1.10 * profile_anomaly + 0.45 * feature_completeness
        text_prior = 1.15 * text_richness + 0.75 * text_completeness - 0.35 * graph_connectivity
        graph_prior = 1.30 * graph_connectivity + 0.90 * graph_reciprocity - 0.35 * text_richness
        return torch.stack((feature_prior, text_prior, graph_prior), dim=1)


class FeatureTextGraphAdaptiveFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        description_dim: int,
        tweet_dim: int,
        num_prop_dim: int,
        cat_prop_dim: int,
        quality_dim: int,
        quality_feature_names: list[str],
        gate_hidden_dim: int,
        dropout: float,
        temperature: float,
        relation_count: int,
    ) -> None:
        super().__init__()
        branch_dim = max(16, hidden_dim // 2)
        self.description_encoder = nn.Sequential(nn.Linear(description_dim, branch_dim), nn.LayerNorm(branch_dim), nn.LeakyReLU())
        self.tweet_encoder = nn.Sequential(nn.Linear(tweet_dim, branch_dim), nn.LayerNorm(branch_dim), nn.LeakyReLU())
        self.numeric_encoder = nn.Sequential(nn.Linear(num_prop_dim, branch_dim), nn.LayerNorm(branch_dim), nn.LeakyReLU())
        self.categorical_encoder = nn.Sequential(nn.Linear(cat_prop_dim, branch_dim), nn.LayerNorm(branch_dim), nn.LeakyReLU())

        self.text_branch = nn.Sequential(
            nn.Linear(branch_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
        )
        self.feature_branch = nn.Sequential(
            nn.Linear(branch_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
        )
        self.graph_seed = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
        )
        self.graph_encoder_1 = RGCNConv(hidden_dim, hidden_dim, num_relations=relation_count)
        self.graph_encoder_2 = RGCNConv(hidden_dim, hidden_dim, num_relations=relation_count)
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
        )
        self.fusion = ReliabilityAwareFusion(
            hidden_dim=hidden_dim,
            quality_dim=quality_dim,
            quality_feature_names=quality_feature_names,
            gate_hidden_dim=gate_hidden_dim,
            dropout=dropout,
            temperature=temperature,
        )
        self.residual_norm = nn.LayerNorm(hidden_dim)
        self.output_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
        )
        self.output_head = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def _encode_local_modalities(
        self,
        description: torch.Tensor,
        tweet: torch.Tensor,
        num_prop: torch.Tensor,
        cat_prop: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        description_repr = self.description_encoder(description)
        tweet_repr = self.tweet_encoder(tweet)
        numeric_repr = self.numeric_encoder(num_prop)
        categorical_repr = self.categorical_encoder(cat_prop)

        text_repr = self.text_branch(torch.cat((description_repr, tweet_repr), dim=1))
        feature_repr = self.feature_branch(torch.cat((numeric_repr, categorical_repr), dim=1))
        graph_seed = self.graph_seed(torch.cat((feature_repr, text_repr), dim=1))
        return feature_repr, text_repr, graph_seed

    def forward(
        self,
        description: torch.Tensor,
        tweet: torch.Tensor,
        num_prop: torch.Tensor,
        cat_prop: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
        quality: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if edge_type is None:
            raise ValueError("Adaptive fusion model requires relation-aware edge types.")
        if quality is None:
            raise ValueError("Adaptive fusion model requires modality quality features.")

        feature_repr, text_repr, graph_seed = self._encode_local_modalities(description, tweet, num_prop, cat_prop)
        graph_repr = self.graph_encoder_1(graph_seed, edge_index, edge_type)
        graph_repr = self.dropout(graph_repr)
        graph_repr = self.graph_encoder_2(graph_repr, edge_index, edge_type)
        graph_repr = self.graph_projection(graph_repr)

        fused_repr, diagnostics = self.fusion(feature_repr, text_repr, graph_repr, quality)
        combined_repr = self.residual_norm(graph_repr + fused_repr)
        logits = self.output_head(self.output_block(combined_repr))
        if return_aux:
            return logits, diagnostics
        return logits


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
            "Skipping %s because torch_geometric is unavailable: %s",
            PRIMARY_METHOD_NAME,
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
    quality_frame = _build_quality_feature_frame(users)
    quality_tensor = _scaled_frame_to_tensor(quality_frame)

    labels = torch.tensor(users["label_id"].clip(lower=0).to_numpy(dtype=np.int64), dtype=torch.long)
    train_indices = torch.tensor(np.flatnonzero(((users["split"] == "train") & (users["label_id"] >= 0)).to_numpy()), dtype=torch.long)
    val_indices = torch.tensor(np.flatnonzero(((users["split"] == "val") & (users["label_id"] >= 0)).to_numpy()), dtype=torch.long)
    test_indices = torch.tensor(np.flatnonzero(((users["split"] == "test") & (users["label_id"] >= 0)).to_numpy()), dtype=torch.long)

    id_to_index = {user_id: index for index, user_id in enumerate(users["user_id"].tolist())}
    edge_index, edge_type = _build_relation_graph(id_to_index, graph_edges)

    model = FeatureTextGraphAdaptiveFusion(
        hidden_dim=config.gnn_hidden_dim,
        description_dim=description_tensor.shape[1],
        tweet_dim=tweet_tensor.shape[1],
        num_prop_dim=num_prop_tensor.shape[1],
        cat_prop_dim=cat_prop_tensor.shape[1],
        quality_dim=quality_tensor.shape[1],
        quality_feature_names=quality_frame.columns.tolist(),
        gate_hidden_dim=config.adaptive_gate_hidden_dim,
        dropout=config.gnn_dropout,
        temperature=config.adaptive_fusion_temperature,
        relation_count=2,
    )
    return [
        _train_gnn_model(
            config=config,
            name=PRIMARY_METHOD_NAME,
            family=PRIMARY_METHOD_FAMILY,
            users=users,
            model=model,
            description_tensor=description_tensor,
            tweet_tensor=tweet_tensor,
            num_prop_tensor=num_prop_tensor,
            cat_prop_tensor=cat_prop_tensor,
            quality_tensor=quality_tensor,
            quality_frame=quality_frame,
            labels=labels,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            edge_index=edge_index,
            edge_type=edge_type,
        )
    ]


def _train_gnn_model(
    config: ProjectConfig,
    name: str,
    family: str,
    users: pd.DataFrame,
    model: nn.Module,
    description_tensor: torch.Tensor,
    tweet_tensor: torch.Tensor,
    num_prop_tensor: torch.Tensor,
    cat_prop_tensor: torch.Tensor,
    quality_tensor: torch.Tensor,
    quality_frame: pd.DataFrame,
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
        logits = _forward_model(
            model=model,
            description_tensor=description_tensor,
            tweet_tensor=tweet_tensor,
            num_prop_tensor=num_prop_tensor,
            cat_prop_tensor=cat_prop_tensor,
            quality_tensor=quality_tensor,
            edge_index=edge_index,
            edge_type=edge_type,
        )
        loss = criterion(logits[train_indices], labels[train_indices])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = _forward_model(
                model=model,
                description_tensor=description_tensor,
                tweet_tensor=tweet_tensor,
                num_prop_tensor=num_prop_tensor,
                cat_prop_tensor=cat_prop_tensor,
                quality_tensor=quality_tensor,
                edge_index=edge_index,
                edge_type=edge_type,
            )
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
        logits, aux = _forward_model(
            model=model,
            description_tensor=description_tensor,
            tweet_tensor=tweet_tensor,
            num_prop_tensor=num_prop_tensor,
            cat_prop_tensor=cat_prop_tensor,
            quality_tensor=quality_tensor,
            edge_index=edge_index,
            edge_type=edge_type,
            return_aux=True,
        )

    metrics_rows: list[dict[str, float | str]] = []
    prediction_rows: list[dict[str, float | str | int]] = []
    gate_weights = aux.get("gate_weights")
    attention_weights = aux.get("attention_weights")
    for split_name, indices in (("val", val_indices), ("test", test_indices)):
        split_probs = torch.softmax(logits[indices], dim=1)[:, 1].cpu().numpy()
        split_preds = (split_probs >= 0.5).astype(int)
        split_true = labels[indices].cpu().numpy()
        metrics_rows.append(
            {
                "experiment": name,
                "family": family,
                "split": split_name,
                **_compute_metrics(split_true, split_preds, split_probs),
            }
        )
        user_subset = users.iloc[indices.cpu().numpy()]
        split_gate_weights = gate_weights[indices].cpu().numpy() if gate_weights is not None else None
        split_attention_weights = attention_weights[indices].cpu().numpy() if attention_weights is not None else None
        for row_index, (user_id, true_label, pred_label, probability) in enumerate(
            zip(
                user_subset["user_id"].tolist(),
                split_true.tolist(),
                split_preds.tolist(),
                split_probs.tolist(),
            )
        ):
            row: dict[str, float | str | int] = {
                "experiment": name,
                "family": family,
                "split": split_name,
                "user_id": user_id,
                "true_label": int(true_label),
                "pred_label": int(pred_label),
                "bot_probability": float(probability),
            }
            if split_gate_weights is not None:
                for modality_index, modality_name in enumerate(MODALITY_NAMES):
                    row[f"{modality_name}_weight"] = float(split_gate_weights[row_index, modality_index])
            if split_attention_weights is not None:
                for modality_index, modality_name in enumerate(MODALITY_NAMES):
                    row[f"attention_to_{modality_name}"] = float(split_attention_weights[row_index, modality_index])
            prediction_rows.append(row)

    _write_gate_diagnostics(
        config=config,
        name=name,
        users=users,
        logits=logits,
        aux=aux,
        quality_frame=quality_frame,
    )

    artifact_path = config.models_dir / f"{name}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "best_val_f1": best_val_f1,
            "quality_columns": quality_frame.columns.tolist(),
        },
        artifact_path,
    )
    return GNNResult(
        metrics_rows=metrics_rows,
        predictions=pd.DataFrame(prediction_rows),
        best_val_f1=best_val_f1,
        artifact_path=artifact_path,
    )


def _forward_model(
    model: nn.Module,
    description_tensor: torch.Tensor,
    tweet_tensor: torch.Tensor,
    num_prop_tensor: torch.Tensor,
    cat_prop_tensor: torch.Tensor,
    quality_tensor: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor | None,
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    output = model(
        description_tensor,
        tweet_tensor,
        num_prop_tensor,
        cat_prop_tensor,
        edge_index,
        edge_type,
        quality=quality_tensor,
        return_aux=return_aux,
    )
    if return_aux:
        if isinstance(output, tuple):
            return output
        return output, {}
    if isinstance(output, tuple):
        return output[0]
    return output


def _scaled_tensor(users: pd.DataFrame, columns: list[str]) -> torch.Tensor:
    frame = users.reindex(columns=columns, fill_value=0.0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return _scaled_frame_to_tensor(frame)


def _scaled_frame_to_tensor(frame: pd.DataFrame) -> torch.Tensor:
    matrix = frame.to_numpy(dtype=np.float32)
    if matrix.shape[1] == 0:
        return torch.zeros((len(frame), 0), dtype=torch.float32)
    scaler = StandardScaler()
    matrix = scaler.fit_transform(matrix).astype(np.float32)
    return torch.tensor(matrix, dtype=torch.float32)


def _numeric_series(users: pd.DataFrame, column: str) -> pd.Series:
    if column not in users.columns:
        return pd.Series(np.zeros(len(users), dtype=np.float32), index=users.index, dtype=np.float32)
    return pd.to_numeric(users[column], errors="coerce").fillna(0.0)


def _build_quality_feature_frame(users: pd.DataFrame) -> pd.DataFrame:
    description_length = _numeric_series(users, "description_length")
    sampled_tweet_count = _numeric_series(users, "sampled_tweet_count")
    total_in_degree = _numeric_series(users, "total_in_degree")
    total_out_degree = _numeric_series(users, "total_out_degree")
    friend_in_count = _numeric_series(users, "friend_in_count")
    friend_out_count = _numeric_series(users, "friend_out_count")
    has_location = _numeric_series(users, "has_location").clip(lower=0.0, upper=1.0)
    has_url = _numeric_series(users, "has_url").clip(lower=0.0, upper=1.0)
    default_profile_image = _numeric_series(users, "default_profile_image").clip(lower=0.0, upper=1.0)
    in_out_ratio = _numeric_series(users, "in_out_ratio")
    account_age_days = _numeric_series(users, "account_age_days")

    missing_rate = (
        (description_length <= 0.0).astype(float)
        + (1.0 - has_location)
        + (1.0 - has_url)
        + default_profile_image
    ) / 4.0

    anomaly_sources = pd.DataFrame(
        {
            "followers_following_ratio": _numeric_series(users, "followers_following_ratio"),
            "tweets_per_day": _numeric_series(users, "tweets_per_day"),
            "listed_per_1k_followers": _numeric_series(users, "listed_per_1k_followers"),
            "log_followers": _numeric_series(users, "log_followers"),
            "log_following": _numeric_series(users, "log_following"),
            "log_tweet_count": _numeric_series(users, "log_tweet_count"),
        }
    ).replace([np.inf, -np.inf], 0.0)
    anomaly_centered = anomaly_sources - anomaly_sources.mean()
    anomaly_scale = anomaly_sources.std(ddof=0).replace(0.0, 1.0)
    profile_anomaly = (anomaly_centered / anomaly_scale).abs().mean(axis=1)

    quality_frame = pd.DataFrame(index=users.index)
    quality_frame["quality_text_richness"] = np.log1p(description_length + 32.0 * sampled_tweet_count)
    quality_frame["quality_text_completeness"] = (
        (description_length > 0.0).astype(float) + np.clip(sampled_tweet_count / 5.0, 0.0, 1.0)
    ) / 2.0
    quality_frame["quality_graph_connectivity"] = np.log1p(total_in_degree + total_out_degree)
    quality_frame["quality_graph_balance"] = in_out_ratio
    quality_frame["quality_graph_reciprocity"] = (friend_in_count + friend_out_count) / (
        total_in_degree + total_out_degree + 1.0
    )
    quality_frame["quality_feature_completeness"] = 1.0 - missing_rate
    quality_frame["quality_profile_anomaly"] = profile_anomaly
    quality_frame["quality_account_maturity"] = np.log1p(account_age_days)
    return quality_frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)


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


def _write_gate_diagnostics(
    config: ProjectConfig,
    name: str,
    users: pd.DataFrame,
    logits: torch.Tensor,
    aux: dict[str, Any],
    quality_frame: pd.DataFrame,
) -> None:
    gate_weights = aux.get("gate_weights")
    if gate_weights is None or quality_frame.empty:
        return

    probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    predictions = (probabilities >= 0.5).astype(int)
    gate_array = gate_weights.detach().cpu().numpy()
    diagnostics = users[["user_id", "split", "label_id"]].copy().reset_index(drop=True)
    diagnostics["pred_label"] = predictions
    diagnostics["bot_probability"] = probabilities
    diagnostics["dominant_modality"] = [MODALITY_NAMES[index] for index in gate_array.argmax(axis=1)]
    for modality_index, modality_name in enumerate(MODALITY_NAMES):
        diagnostics[f"{modality_name}_weight"] = gate_array[:, modality_index]

    attention_weights = aux.get("attention_weights")
    if attention_weights is not None:
        attention_array = attention_weights.detach().cpu().numpy()
        for modality_index, modality_name in enumerate(MODALITY_NAMES):
            diagnostics[f"attention_to_{modality_name}"] = attention_array[:, modality_index]

    diagnostics = pd.concat([diagnostics, quality_frame.reset_index(drop=True)], axis=1)
    diagnostics.to_csv(config.tables_dir / f"{name}_gate_diagnostics.csv", index=False)


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
