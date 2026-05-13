"""Multi-granularity BotDGTv1.

Multi-granularity dynamic BotDGT that processes snapshots at multiple temporal granularities
(e.g., year, six_months, three_months) and fuses representations using Gate/Mean/Concat.
"""
from __future__ import annotations

import logging
import os
import pickle
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ...config import ProjectConfig
from .data import BOTDGT_ABLATION_MODES, BotDGTDataset

LOGGER = logging.getLogger(__name__)


@dataclass
class _ArgsMG:
    """Arguments for multi-granularity BotDGT."""
    hidden_dim: int
    structural_head_config: int
    structural_drop: float
    temporal_head_config: int
    temporal_drop: float
    temporal_module_type: str
    structural_learning_rate: float
    temporal_learning_rate: float
    weight_decay: float
    epoch: int
    seed: int
    device: str
    dataset_name: str
    batch_size: int
    coefficient: float
    early_stop: bool
    patience: int
    use_multi_granularity: bool = False
    granularities: list[str] = field(default_factory=list)
    granularity_fusion: str = "gate"
    share_structural_encoder: bool = True
    share_temporal_encoder: bool = True
    temporal_readout: str = "last"
    granularity_window_sizes: list[int] = field(default_factory=list)


def _null_metrics():
    return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}


class GranularityFusion(nn.Module):
    """Fuses logits or representations from multiple temporal granularities.

    Input: list of [batch_size, 2] tensors (bot/human logits) or [batch_size, hidden_dim]
    Output: [batch_size, 2] (fused logits) or [batch_size, hidden_dim]
    """

    def __init__(self, hidden_dim: int, num_granularities: int, fusion_type: str = "gate"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_granularities = num_granularities
        self.fusion_type = fusion_type
        self.input_dim = 2  # bot/human logits

        if fusion_type == "gate":
            self.gate_mlp = nn.Sequential(
                nn.Linear(self.input_dim * num_granularities, self.input_dim * num_granularities // 2),
                nn.LeakyReLU(),
                nn.Linear(self.input_dim * num_granularities // 2, num_granularities),
            )
        elif fusion_type == "concat":
            self.concat_mlp = nn.Sequential(
                nn.Linear(self.input_dim * num_granularities, self.input_dim * num_granularities),
                nn.LeakyReLU(),
                nn.Linear(self.input_dim * num_granularities, self.input_dim),
            )

    def forward(self, representations: list[torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        """
        Args:
            representations: List of [batch_size, 2] tensors (logits) or [batch_size, hidden_dim], one per granularity
        Returns:
            fused: [batch_size, 2] tensor (fused logits)
            info: Optional dict with fusion info (e.g., gate weights)
        """
        stacked = torch.stack(representations, dim=1)  # [B, num_gr, 2]

        if self.fusion_type == "gate":
            concat_repr = torch.cat(representations, dim=-1)  # [B, 2 * num_gr]
            gates = self.gate_mlp(concat_repr)  # [B, num_gr]
            weights = F.softmax(gates, dim=-1)  # [B, num_gr]
            fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # [B, 2]
            info = {"gate_weights": weights}

        elif self.fusion_type == "mean":
            fused = stacked.mean(dim=1)  # [B, 2]
            info = None

        elif self.fusion_type == "concat":
            concat_repr = torch.cat(representations, dim=-1)  # [B, 2 * num_gr]
            fused = self.concat_mlp(concat_repr)  # [B, 2]
            info = None

        return fused, info


class MultiGranularityBotDyGNN(nn.Module):
    """BotDGT model that processes multiple temporal granularities."""

    def __init__(self, args: _ArgsMG, datasets: dict):
        super().__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.granularities = args.granularities
        self.num_granularities = len(args.granularities)
        self.fusion_type = args.granularity_fusion
        self.datasets = datasets

        from .model import (
            NodeFeatureEmbeddingLayer,
            GraphStructuralLayer,
            GraphTemporalLayer,
            PositionEncodingClusteringCoefficient,
            PositionEncodingBidirectionalLinks,
        )

        self.node_feature_embedding = NodeFeatureEmbeddingLayer(hidden_dim=args.hidden_dim)

        if args.share_structural_encoder:
            self.structural_encoder = GraphStructuralLayer(
                hidden_dim=args.hidden_dim,
                n_heads=args.structural_head_config,
                dropout=args.structural_drop,
            )
        else:
            self.structural_encoders = nn.ModuleDict({
                g: GraphStructuralLayer(
                    hidden_dim=args.hidden_dim,
                    n_heads=args.structural_head_config,
                    dropout=args.structural_drop,
                ) for g in self.granularities
            })

        if args.share_temporal_encoder:
            self.temporal_encoders = nn.ModuleDict({
                g: GraphTemporalLayer(
                    hidden_dim=args.hidden_dim,
                    n_heads=args.temporal_head_config,
                    dropout=args.temporal_drop,
                    num_time_steps=ws,
                    temporal_module_type=args.temporal_module_type,
                ) for g, ws in zip(self.granularities, args.granularity_window_sizes)
            })
        else:
            self.temporal_encoders = nn.ModuleDict({
                g: GraphTemporalLayer(
                    hidden_dim=args.hidden_dim,
                    n_heads=args.temporal_head_config,
                    dropout=args.temporal_drop,
                    num_time_steps=ws,
                    temporal_module_type=args.temporal_module_type,
                ) for g, ws in zip(self.granularities, args.granularity_window_sizes)
            })

        self.pos_clustering = PositionEncodingClusteringCoefficient(hidden_dim=args.hidden_dim)
        self.pos_bidirectional = PositionEncodingBidirectionalLinks(hidden_dim=args.hidden_dim)

        self.fusion = GranularityFusion(
            hidden_dim=args.hidden_dim,
            num_granularities=self.num_granularities,
            fusion_type=args.granularity_fusion,
        )

    def _process_one_granularity(self, granularity: str, batch_data: dict, current_batch_size: int) -> torch.Tensor:
        """Process one granularity to get final representation."""
        structural_layer = (
            self.structural_encoder if hasattr(self, "structural_encoder")
            else self.structural_encoders[granularity]
        )
        temporal_layer = self.temporal_encoders[granularity]

        des_list = batch_data[f"{granularity}_des"]  # list of [N_t, D] tensors
        tweet_list = batch_data[f"{granularity}_tweet"]
        num_list = batch_data[f"{granularity}_num"]
        cat_list = batch_data[f"{granularity}_cat"]
        edge_index_list = batch_data[f"{granularity}_edge_index"]
        clustering_list = batch_data[f"{granularity}_clustering"]
        bidirectional_list = batch_data[f"{granularity}_bidirectional"]
        exist_nodes = batch_data[f"{granularity}_exist_nodes"]  # [T, B]

        structural_outputs = []
        num_snapshots = len(edge_index_list)
        for t in range(num_snapshots):
            x = self.node_feature_embedding(
                des_list[t], tweet_list[t], num_list[t], cat_list[t],
            )
            output = structural_layer(x, edge_index_list[t])[:current_batch_size]
            structural_outputs.append(output)

        structural_outputs = torch.stack(structural_outputs, dim=1)  # [B, T, H]

        # Position encodings - clustering and bidirectional are lists
        pos_clustering_list = [
            self.pos_clustering(clustering_list[t][:current_batch_size])
            for t in range(len(clustering_list))
        ]
        pos_clustering = torch.stack(pos_clustering_list, dim=1)  # [B, T, H]

        pos_bidirectional_list = [
            self.pos_bidirectional(bidirectional_list[t][:current_batch_size])
            for t in range(len(bidirectional_list))
        ]
        pos_bidirectional = torch.stack(pos_bidirectional_list, dim=1)  # [B, T, H]

        # exist_nodes is a list, need to stack and transpose
        exist_nodes_tensor = torch.stack([en[:current_batch_size] for en in exist_nodes], dim=0)  # [T, B]
        exist_nodes = exist_nodes_tensor.transpose(0, 1)  # [B, T]

        temporal_output = temporal_layer(
            structural_outputs, pos_clustering, pos_bidirectional, exist_nodes
        )

        # temporal_output shape: [B, T, 2] - logits for bot/human
        # Use only the logits for fusion
        if self.args.temporal_readout == "last":
            representation = temporal_output[:, -1, :]  # [B, 2]
        else:  # masked_mean
            mask = exist_nodes.unsqueeze(-1).float()  # [B, T, 1]
            masked_output = temporal_output * mask
            count = mask.sum(dim=1).clamp(min=1)  # [B, 1]
            representation = masked_output.sum(dim=1) / count  # [B, 2]

        return representation

    def forward(self, batch_data: dict, current_batch_size: int) -> tuple[torch.Tensor, dict | None]:
        """Forward pass for multi-granularity BotDGT."""
        representations = []

        for granularity in self.granularities:
            repr_node = self._process_one_granularity(granularity, batch_data, current_batch_size)
            representations.append(repr_node)

        fused, fusion_info = self.fusion(representations)

        return fused, fusion_info


class MultiGranularityBotDGTTrainer:
    """Trainer for multi-granularity BotDGT."""

    def __init__(self, args: _ArgsMG, datasets: dict):
        self.args = args
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.datasets = datasets
        self.device = args.device

        self.des_tensor = datasets[self.args.granularities[0]].des_tensor
        self.tweets_tensor = datasets[self.args.granularities[0]].tweets_tensor
        self.num_prop = datasets[self.args.granularities[0]].num_prop
        self.category_prop = datasets[self.args.granularities[0]].category_prop
        self.labels = datasets[self.args.granularities[0]].labels

        self.model = MultiGranularityBotDyGNN(args, datasets)
        self.model.to(self.device)

        params = [
            {"params": self.model.node_feature_embedding.parameters(), "lr": args.structural_learning_rate},
            {"params": self.model.fusion.parameters(), "lr": args.structural_learning_rate},
        ]
        if hasattr(self.model, "structural_encoder"):
            params.append({"params": self.model.structural_encoder.parameters(), "lr": args.structural_learning_rate})
        else:
            params.append({"params": self.model.structural_encoders.parameters(), "lr": args.structural_learning_rate})

        params.extend([
            {"params": enc.parameters(), "lr": args.temporal_learning_rate}
            for enc in self.model.temporal_encoders.values()
        ])

        self.optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0)

        self.best_val_metrics = _null_metrics()
        self.best_val_f1 = 0.0
        self.test_state_dict = None
        self.history_rows: list[dict] = []

        self._prepare_batch_data()

    def _prepare_batch_data(self):
        """Prepare batch data indices for train/val/test."""
        self.train_batches = self._build_batches("train")
        self.val_batches = self._build_batches("val")
        self.test_batches = self._build_batches("test")

    def _build_batches(self, split: str) -> list[dict]:
        """Build batch data for a split across all granularities."""
        ref_gran = self.args.granularities[0]
        ref_data = self.datasets[ref_gran]
        right_list = getattr(ref_data, f"{split}_right")
        n_id_list = getattr(ref_data, f"{split}_n_id")

        batches = []
        for batch_idx in range(len(right_list)):
            batch_size = right_list[batch_idx]
            n_id = n_id_list[batch_idx]

            batch_data = {
                "n_id": n_id,
                "batch_size": batch_size,
            }

            for granularity in self.args.granularities:
                gd = self.datasets[granularity]
                gran_n_id = getattr(gd, f"{split}_n_id")[batch_idx]  # list of tensors (window_size tensors)

                # Keep as list since node counts vary per snapshot
                batch_data[f"{granularity}_des"] = [self.des_tensor[nid] for nid in gran_n_id]
                batch_data[f"{granularity}_tweet"] = [self.tweets_tensor[nid] for nid in gran_n_id]
                batch_data[f"{granularity}_num"] = [self.num_prop[nid] for nid in gran_n_id]
                batch_data[f"{granularity}_cat"] = [self.category_prop[nid] for nid in gran_n_id]
                batch_data[f"{granularity}_edge_index"] = [ei.to(self.device) for ei in getattr(gd, f"{split}_edge_index")[batch_idx]]
                batch_data[f"{granularity}_clustering"] = [c.to(self.device) for c in getattr(gd, f"{split}_clustering_coefficient")[batch_idx]]
                batch_data[f"{granularity}_bidirectional"] = [b.to(self.device) for b in getattr(gd, f"{split}_bidirectional_links_ratio")[batch_idx]]
                batch_data[f"{granularity}_exist_nodes"] = [en.to(self.device) for en in getattr(gd, f"{split}_exist_nodes")[batch_idx]]

            batches.append(batch_data)

        return batches

    def forward_one_batch(self, batch_data: dict) -> tuple:
        # n_id is a list of tensors (one per snapshot), use first snapshot's n_id for labels
        n_id = batch_data["n_id"][0]  # first snapshot's node ids
        labels = self.labels[n_id.tolist()].to(self.device)
        output, fusion_info = self.model(batch_data, batch_data["batch_size"])
        loss = self.criterion(output, labels)
        return output, loss, labels, fusion_info

    def train_per_epoch(self, current_epoch: int) -> dict:
        self.model.train()
        all_preds, all_labels, total_loss = [], [], 0.0

        for batch_data in self.train_batches:
            self.optimizer.zero_grad()
            output, loss, labels, _ = self.forward_one_batch(batch_data)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(output, dim=-1).cpu().numpy()
            all_preds.extend(preds[:batch_data["batch_size"]])
            all_labels.extend(labels[:batch_data["batch_size"]].cpu().numpy())

        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="binary", zero_division=0),
            "precision": precision_score(all_labels, all_preds, average="binary", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="binary", zero_division=0),
            "loss": total_loss / len(self.train_batches),
        }
        LOGGER.info("Epoch-%d train loss: %.6f acc: %.6f f1: %.6f",
                    current_epoch, metrics["loss"], metrics["accuracy"], metrics["f1"])
        return metrics

    @torch.no_grad()
    def val_per_epoch(self, current_epoch: int) -> dict:
        self.model.eval()
        all_preds, all_labels, total_loss = [], [], 0.0

        for batch_data in self.val_batches:
            output, loss, labels, _ = self.forward_one_batch(batch_data)
            total_loss += loss.item()
            preds = torch.argmax(output, dim=-1).cpu().numpy()
            all_preds.extend(preds[:batch_data["batch_size"]])
            all_labels.extend(labels[:batch_data["batch_size"]].cpu().numpy())

        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="binary", zero_division=0),
            "precision": precision_score(all_labels, all_preds, average="binary", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="binary", zero_division=0),
            "loss": total_loss / len(self.val_batches),
        }
        LOGGER.info("Epoch-%d val loss: %.6f acc: %.6f f1: %.6f",
                    current_epoch, metrics["loss"], metrics["accuracy"], metrics["f1"])
        return metrics

    def train(self) -> tuple:
        patience_counter = 0
        for epoch in range(self.args.epoch):
            train_metrics = self.train_per_epoch(epoch)
            self.scheduler.step()
            val_metrics = self.val_per_epoch(epoch)

            self.history_rows.append({
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
            })

            if val_metrics["f1"] >= self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self.best_val_metrics = val_metrics
                self.test_state_dict = deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if self.args.early_stop and patience_counter >= self.args.patience:
                LOGGER.info("Early stopping at epoch %d", epoch)
                break

        return self.test_state_dict, self.best_val_metrics


def run_botdgt_multi_granularity(
    config: ProjectConfig,
    *,
    reset_random_state: bool = False,
) -> dict:
    """Run multi-granularity BotDGT experiment."""
    if reset_random_state:
        import random as _random
        _random.seed(config.random_state)
        np.random.seed(config.random_state)
        torch.manual_seed(config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    granularity_list = [g.strip() for g in config.granularities.split(",")]

    LOGGER.info("=== Multi-granularity BotDGTv1: Loading datasets ===")
    LOGGER.info("Granularities: %s", granularity_list)
    LOGGER.info("Fusion type: %s", config.granularity_fusion)

    datasets = {}
    window_sizes = []
    for interval in granularity_list:
        LOGGER.info("Loading dataset for granularity: %s", interval)
        ds = BotDGTDataset(config, interval=interval)
        datasets[interval] = ds
        window_sizes.append(ds.window_size)
        LOGGER.info("  window_size=%d snapshots", ds.window_size)

    args = _ArgsMG(
        hidden_dim=config.gnn_hidden_dim,
        structural_head_config=config.botdgt_structural_heads,
        structural_drop=config.botdgt_structural_dropout,
        temporal_head_config=config.botdgt_temporal_heads,
        temporal_drop=config.botdgt_temporal_dropout,
        temporal_module_type=config.botdgt_temporal_module,
        structural_learning_rate=config.botdgt_structural_lr,
        temporal_learning_rate=config.botdgt_temporal_lr,
        weight_decay=config.botdgt_weight_decay,
        epoch=config.botdgt_epochs,
        seed=config.random_state,
        device=device_str,
        dataset_name="Twibot-20",
        batch_size=config.botdgt_batch_size,
        coefficient=config.botdgt_loss_coefficient,
        early_stop=False,
        patience=10,
        use_multi_granularity=True,
        granularities=granularity_list,
        granularity_fusion=config.granularity_fusion,
        share_structural_encoder=config.share_structural_encoder,
        share_temporal_encoder=config.share_temporal_encoder,
        temporal_readout=config.temporal_readout,
        granularity_window_sizes=window_sizes,
    )

    LOGGER.info(
        "=== Multi-granularity BotDGTv1: Training === "
        "fusion=%s share_struct=%s share_temp=%s",
        config.granularity_fusion, config.share_structural_encoder, config.share_temporal_encoder
    )

    trainer = MultiGranularityBotDGTTrainer(args, datasets)
    best_state, val_metrics = trainer.train()

    LOGGER.info(
        "=== Multi-granularity BotDGTv1: Val results — acc=%.5f f1=%.5f ===",
        val_metrics["accuracy"], val_metrics["f1"]
    )

    experiment_name = "feature_text_graph_botdgt_multi_granularity"

    model_dir = "Twibot-20"
    os.makedirs(model_dir, exist_ok=True)
    model_name = f"mg_{'+'.join(granularity_list)} + {args.seed} + {val_metrics['accuracy']} + {val_metrics['f1']}.pt"
    torch.save(best_state, os.path.join(model_dir, model_name))
    LOGGER.info("Saved model: %s", model_name)

    with open(str(datasets[granularity_list[0]]._DATA_DIR / "processed_data" / "uid2global_index.pkl"), "rb") as f:
        uid2global = pickle.load(f)
    global2uid = {v: k for k, v in uid2global.items()}

    model = trainer.model
    model.eval()

    with torch.no_grad():
        all_preds, all_probs, all_labels, all_uids = [], [], [], []
        for batch_data in trainer.test_batches:
            output, _, labels, _ = trainer.forward_one_batch(batch_data)
            probs = torch.softmax(output, dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
            all_preds.extend(preds[:batch_data["batch_size"]])
            all_probs.extend(probs[:batch_data["batch_size"], 1].cpu().numpy())
            all_labels.extend(labels[:batch_data["batch_size"]].cpu().numpy())
            for idx in batch_data["n_id"][0]:  # first snapshot's node ids
                all_uids.append(global2uid.get(int(idx), f"unknown_{idx}"))

    predictions_df = pd.DataFrame({
        "experiment": experiment_name,
        "family": "feature_text_graph",
        "split": "test",
        "user_id": all_uids[:len(all_labels)],
        "true_label": all_labels,
        "pred_label": all_preds,
        "bot_probability": all_probs,
    })

    metrics_rows = [{
        "experiment": experiment_name,
        "family": "feature_text_graph",
        "split": "test",
        "accuracy": val_metrics["accuracy"],
        "precision": val_metrics["precision"],
        "recall": val_metrics["recall"],
        "f1": val_metrics["f1"],
        "auc_roc": float("nan"),
    }]

    artifact_path = config.models_dir / f"{experiment_name}.pt"
    torch.save({"state_dict": best_state, "best_val_f1": trainer.best_val_f1}, artifact_path)

    return {
        "metrics_rows": metrics_rows,
        "predictions": predictions_df,
        "best_val_f1": trainer.best_val_f1,
        "artifact_path": artifact_path,
        "training_history": pd.DataFrame(trainer.history_rows).assign(experiment=experiment_name),
        "granularities": granularity_list,
        "fusion": config.granularity_fusion,
    }
