from __future__ import annotations

import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ...config import ProjectConfig
from .data import BOTDGT_ABLATION_MODES, BotDGTDataset, _DATA_DIR
from .loss import all_snapshots_loss
from .model import BotDyGNN

LOGGER = logging.getLogger(__name__)

BOTDGT_EXPERIMENT_PREFIX = "feature_text_graph_botdgt"
BOTDGT_ABLATION_LABELS = {
    "full": "none",
    "no_profile": "profile",
    "no_text": "text",
    "no_graph": "graph",
}


def botdgt_experiment_name(ablation_mode: str) -> str:
    if ablation_mode == "full":
        return BOTDGT_EXPERIMENT_PREFIX
    return f"{BOTDGT_EXPERIMENT_PREFIX}_{ablation_mode}"


def _reset_botdgt_random_state(seed: int) -> None:
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class _Args:
    hidden_dim: int
    structural_head_config: int
    structural_drop: float
    temporal_head_config: int
    temporal_drop: float
    window_size: int
    temporal_module_type: str
    structural_learning_rate: float
    temporal_learning_rate: float
    weight_decay: float
    epoch: int
    seed: int
    device: str
    dataset_name: str
    interval: str
    batch_size: int
    coefficient: float
    early_stop: bool
    patience: int


def _null_metrics():
    return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}


def _compute_metrics_one_snapshot(y_true, y_output, exist_nodes):
    metrics = _null_metrics()
    if torch.any(torch.isnan(y_output)):
        return metrics
    y_pred = torch.softmax(y_output, dim=-1)
    if exist_nodes != "all":
        y_pred = y_pred[torch.where(exist_nodes == 1)]
        y_true = y_true[torch.where(exist_nodes == 1)]
    y_true_np = y_true.to("cpu").detach().numpy()
    y_pred_label = torch.argmax(y_pred, dim=-1).to("cpu").detach().numpy()
    metrics["accuracy"] = round(float(accuracy_score(y_true_np, y_pred_label)), 5)
    metrics["f1"] = round(float(f1_score(y_true_np, y_pred_label)), 5)
    metrics["precision"] = round(float(precision_score(y_true_np, y_pred_label)), 5)
    metrics["recall"] = round(float(recall_score(y_true_np, y_pred_label)), 5)
    return metrics


def _is_better(now, pre):
    if now["accuracy"] > pre["accuracy"]:
        return True
    if now["accuracy"] < pre["accuracy"]:
        return False
    return now["f1"] >= pre["f1"]


class BotDGTTrainer:
    def __init__(self, args: _Args, dataset: BotDGTDataset):
        self.args = args
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.dataset = dataset

        self.des_tensor = dataset.des_tensor
        self.tweets_tensor = dataset.tweets_tensor
        self.num_prop = dataset.num_prop
        self.category_prop = dataset.category_prop
        self.labels = dataset.labels

        self.train_right = dataset.train_right
        self.train_n_id = dataset.train_n_id
        self.train_edge_index = dataset.train_edge_index
        self.train_edge_type = dataset.train_edge_type
        self.train_exist_nodes = dataset.train_exist_nodes
        self.train_clustering_coefficient = dataset.train_clustering_coefficient
        self.train_bidirectional_links_ratio = dataset.train_bidirectional_links_ratio

        self.test_right = dataset.test_right
        self.test_n_id = dataset.test_n_id
        self.test_edge_index = dataset.test_edge_index
        self.test_edge_type = dataset.test_edge_type
        self.test_exist_nodes = dataset.test_exist_nodes
        self.test_clustering_coefficient = dataset.test_clustering_coefficient
        self.test_bidirectional_links_ratio = dataset.test_bidirectional_links_ratio

        self.val_right = dataset.val_right
        self.val_n_id = dataset.val_n_id
        self.val_edge_index = dataset.val_edge_index
        self.val_edge_type = dataset.val_edge_type
        self.val_exist_nodes = dataset.val_exist_nodes
        self.val_clustering_coefficient = dataset.val_clustering_coefficient
        self.val_bidirectional_links_ratio = dataset.val_bidirectional_links_ratio

        self.model = BotDyGNN(self.args)
        self.model.to(self.args.device)

        params = [
            {"params": self.model.node_feature_embedding_layer.parameters(),
             "lr": self.args.structural_learning_rate},
            {"params": self.model.structural_layer.parameters(),
             "lr": self.args.structural_learning_rate},
            {"params": self.model.temporal_layer.parameters(),
             "lr": self.args.temporal_learning_rate},
        ]
        self.optimizer = torch.optim.AdamW(params, weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20, eta_min=0,
        )

        self.best_val_metrics = _null_metrics()
        self.best_val_f1 = 0.0
        self.test_state_dict_list = []
        self.test_epoch_list = []
        self.test_metrics = _null_metrics()
        self.test_state_dict = None
        self.last_state_dict = None
        self.history_rows: list[dict] = []

    def forward_one_batch(self, batch_size, batch_n_id, batch_edge_index, batch_exist_nodes,
                           batch_clustering_coefficient, batch_bidirectional_links_ratio):
        des_tensor_list = [self.des_tensor[n_id].to(self.args.device) for n_id in batch_n_id]
        tweet_tensor_list = [self.tweets_tensor[n_id].to(self.args.device) for n_id in batch_n_id]
        num_prop_list = [self.num_prop[n_id].to(self.args.device) for n_id in batch_n_id]
        category_prop_list = [self.category_prop[n_id].to(self.args.device) for n_id in batch_n_id]
        label_list = [self.labels[n_id][:batch_size].to(self.args.device) for n_id in batch_n_id]
        label_list = torch.stack(label_list, dim=0)
        edge_index_list = [_.to(self.args.device) for _ in batch_edge_index]
        clustering_coefficient_list = [_.to(self.args.device) for _ in batch_clustering_coefficient]
        bidirectional_links_ratio_list = [_.to(self.args.device) for _ in batch_bidirectional_links_ratio]
        exist_nodes_list = [exist_nodes[:batch_size].to(self.args.device) for exist_nodes in batch_exist_nodes]
        exist_nodes_list = torch.stack(exist_nodes_list, dim=0)

        output = self.model(
            des_tensor_list, tweet_tensor_list, num_prop_list, category_prop_list,
            edge_index_list, clustering_coefficient_list, bidirectional_links_ratio_list,
            exist_nodes_list, batch_size,
        )
        output = output.transpose(0, 1)
        loss = all_snapshots_loss(self.criterion, output, label_list, exist_nodes_list,
                                   coefficient=self.args.coefficient)
        return output, loss, label_list, exist_nodes_list

    def forward_one_epoch(self, right, n_id, edge_index, exist_nodes, clustering_coefficient,
                           bidirectional_links_ratio):
        all_label = []
        all_output = []
        all_exist_nodes = []
        total_loss = 0.0
        for batch_size, batch_n_id, batch_edge_index, batch_exist_nodes, \
            batch_clustering_coefficient, batch_bidirectional_links_ratio in \
                zip(right, n_id, edge_index, exist_nodes, clustering_coefficient, bidirectional_links_ratio):
            output, loss, label_list, exist_nodes_list = self.forward_one_batch(
                batch_size, batch_n_id, batch_edge_index, batch_exist_nodes,
                batch_clustering_coefficient, batch_bidirectional_links_ratio,
            )
            total_loss += loss.item() / self.args.window_size / len(right)
            all_output.append(output)
            all_label.append(label_list)
            all_exist_nodes.append(exist_nodes_list)
        all_output = torch.cat(all_output, dim=1)
        all_label = torch.cat(all_label, dim=1)
        all_exist_nodes = torch.cat(all_exist_nodes, dim=1)
        metrics = _compute_metrics_one_snapshot(all_label[-1], all_output[-1], exist_nodes=all_exist_nodes[-1])
        metrics["loss"] = total_loss
        return metrics

    def train_per_epoch(self, current_epoch):
        self.model.train()
        all_label = []
        all_output = []
        all_exist_nodes = []
        total_loss = 0.0
        for batch_size, batch_n_id, batch_edge_index, batch_exist_nodes, \
            batch_clustering_coefficient, batch_bidirectional_links_ratio in \
                zip(self.train_right, self.train_n_id, self.train_edge_index, self.train_exist_nodes,
                    self.train_clustering_coefficient, self.train_bidirectional_links_ratio):
            self.optimizer.zero_grad()
            output, loss, label_list, exist_nodes_list = self.forward_one_batch(
                batch_size, batch_n_id, batch_edge_index, batch_exist_nodes,
                batch_clustering_coefficient, batch_bidirectional_links_ratio,
            )
            total_loss += loss.item() / self.args.window_size / len(self.train_right)
            loss.backward()
            self.optimizer.step()
            all_output.append(output)
            all_label.append(label_list)
            all_exist_nodes.append(exist_nodes_list)
        all_output = torch.cat(all_output, dim=1)
        all_label = torch.cat(all_label, dim=1)
        all_exist_nodes = torch.cat(all_exist_nodes, dim=1)
        metrics = _compute_metrics_one_snapshot(all_label[-1], all_output[-1], exist_nodes=all_exist_nodes[-1])
        metrics["loss"] = total_loss

        parts = [f"Epoch-{current_epoch} train loss: {total_loss:.6f}"]
        for key in ["accuracy", "precision", "recall", "f1"]:
            parts.append(f"{key}: {metrics[key]:.6f}")
        LOGGER.info(" ".join(parts))
        return metrics

    @torch.no_grad()
    def val_per_epoch(self, current_epoch):
        self.model.eval()
        metrics = self.forward_one_epoch(
            self.val_right, self.val_n_id, self.val_edge_index, self.val_exist_nodes,
            self.val_clustering_coefficient, self.val_bidirectional_links_ratio,
        )
        parts = [f"Epoch-{current_epoch} val loss: {metrics['loss']:.6f}"]
        for key in ["accuracy", "precision", "recall", "f1"]:
            parts.append(f"{key}: {metrics[key]:.6f}")
        LOGGER.info(" ".join(parts))
        return metrics

    @torch.no_grad()
    def test_model(self, state_dict=None):
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        self.model.eval()
        metrics = self.forward_one_epoch(
            self.test_right, self.test_n_id, self.test_edge_index, self.test_exist_nodes,
            self.test_clustering_coefficient, self.test_bidirectional_links_ratio,
        )
        return metrics

    def train(self):
        validate_score_non_improvement_count = 0
        self.model.train()
        for current_epoch in range(self.args.epoch):
            self.train_per_epoch(current_epoch)
            self.scheduler.step()
            val_metrics = self.val_per_epoch(current_epoch)

            # Track history
            self.history_rows.append({
                "epoch": current_epoch,
                "train_loss": 0.0,  # Will be filled
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
            })

            if _is_better(val_metrics, self.best_val_metrics):
                self.best_val_metrics = val_metrics
                self.best_val_f1 = val_metrics["f1"]
                self.test_epoch_list.append(current_epoch)
                self.test_state_dict = deepcopy(self.model.state_dict())
                self.test_state_dict_list.append(self.test_state_dict)
                validate_score_non_improvement_count = 0
            else:
                validate_score_non_improvement_count += 1
            self.last_state_dict = deepcopy(self.model.state_dict())

            if self.args.early_stop and validate_score_non_improvement_count >= self.args.patience:
                LOGGER.info("Early stopping at epoch: %d", current_epoch)
                break

        # Test best model
        best_state = self.test_state_dict_list[-1] if self.test_state_dict_list else self.last_state_dict
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        test_metrics = self.test_model(state_dict=best_state)

        # Save model
        model_dir = self.args.dataset_name
        os.makedirs(model_dir, exist_ok=True)
        model_name = f"{self.args.interval} + {self.args.seed} + {test_metrics['accuracy']} + {test_metrics['f1']}.pt"
        torch.save(best_state, os.path.join(model_dir, model_name))
        LOGGER.info("Saved model: %s", model_name)

        return test_metrics, best_state


def run_botdgt(config: ProjectConfig, ablation_mode: str = "full") -> dict:
    if ablation_mode not in BOTDGT_ABLATION_MODES:
        raise ValueError(f"Unknown BotDGT ablation mode: {ablation_mode}")
    _reset_botdgt_random_state(config.random_state)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_name = botdgt_experiment_name(ablation_mode)

    LOGGER.info("=== BotDGT: Loading preprocessed dataset (ablation=%s) ===", ablation_mode)
    dataset = BotDGTDataset(config, ablation_mode=ablation_mode)

    args = _Args(
        hidden_dim=config.gnn_hidden_dim,
        structural_head_config=config.botdgt_structural_heads,
        structural_drop=config.botdgt_structural_dropout,
        temporal_head_config=config.botdgt_temporal_heads,
        temporal_drop=config.botdgt_temporal_dropout,
        window_size=dataset.window_size,
        temporal_module_type=config.botdgt_temporal_module,
        structural_learning_rate=config.botdgt_structural_lr,
        temporal_learning_rate=config.botdgt_temporal_lr,
        weight_decay=config.botdgt_weight_decay,
        epoch=config.botdgt_epochs,
        seed=config.random_state,
        device=device_str,
        dataset_name="Twibot-20",
        interval=config.botdgt_interval,
        batch_size=config.botdgt_batch_size,
        coefficient=config.botdgt_loss_coefficient,
        early_stop=False,
        patience=10,
    )

    LOGGER.info("=== BotDGT: Starting training (%s) ===", experiment_name)
    LOGGER.info(
        "structural_lr=%.1e temporal_lr=%.1e weight_decay=%.1e epochs=%d batch_size=%d interval=%s window_size=%d",
        args.structural_learning_rate, args.temporal_learning_rate,
        args.weight_decay, args.epoch, args.batch_size, args.interval, args.window_size,
    )
    trainer = BotDGTTrainer(args, dataset)
    test_metrics, best_state = trainer.train()

    LOGGER.info(
        "=== BotDGT: Test results — acc=%.5f f1=%.5f precision=%.5f recall=%.5f ===",
        test_metrics["accuracy"], test_metrics["f1"],
        test_metrics["precision"], test_metrics["recall"],
    )

    # Build predictions for test nodes
    # Load uid mapping for prediction output
    import pickle
    with open(str(_DATA_DIR / "processed_data" / "uid2global_index.pkl"), "rb") as f:
        uid2global = pickle.load(f)
    global2uid = {v: k for k, v in uid2global.items()}

    model = trainer.model
    model.eval()
    with torch.no_grad():
        all_output, all_label, all_exist = [], [], []
        for batch_size, batch_n_id, batch_edge_index, batch_exist_nodes, \
            batch_clustering_coefficient, batch_bidirectional_links_ratio in \
                zip(dataset.test_right, dataset.test_n_id, dataset.test_edge_index,
                    dataset.test_exist_nodes, dataset.test_clustering_coefficient,
                    dataset.test_bidirectional_links_ratio):
            output, _, label_list, exist_nodes_list = trainer.forward_one_batch(
                batch_size, batch_n_id, batch_edge_index, batch_exist_nodes,
                batch_clustering_coefficient, batch_bidirectional_links_ratio,
            )
            all_output.append(output)
            all_label.append(label_list)
            all_exist.append(exist_nodes_list)
        all_output = torch.cat(all_output, dim=1)
        all_label = torch.cat(all_label, dim=1)
        all_exist = torch.cat(all_exist, dim=1)

        last_output = all_output[-1]
        last_label = all_label[-1]
        last_exist = all_exist[-1]

        exist_mask = last_exist == 1
        logits = last_output[exist_mask]
        labels = last_label[exist_mask]
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        test_preds = preds.cpu().numpy()
        test_probs_list = probs[:, 1].cpu().numpy()
        test_labels = labels.cpu().numpy()

    # Map global indices to user IDs
    test_global_indices = dataset.test_idx.cpu().numpy()
    test_uids = []
    for idx in test_global_indices:
        test_uids.append(global2uid.get(int(idx), f"unknown_{idx}"))

    predictions_df = pd.DataFrame({
        "experiment": experiment_name,
        "family": "feature_text_graph",
        "split": "test",
        "user_id": test_uids[:len(test_labels)],
        "true_label": test_labels.tolist(),
        "pred_label": test_preds.tolist(),
        "bot_probability": test_probs_list.tolist(),
    })

    # Build val predictions too
    model.eval()
    with torch.no_grad():
        all_output, all_label, all_exist = [], [], []
        for batch_size, batch_n_id, batch_edge_index, batch_exist_nodes, \
            batch_clustering_coefficient, batch_bidirectional_links_ratio in \
                zip(dataset.val_right, dataset.val_n_id, dataset.val_edge_index,
                    dataset.val_exist_nodes, dataset.val_clustering_coefficient,
                    dataset.val_bidirectional_links_ratio):
            output, _, label_list, exist_nodes_list = trainer.forward_one_batch(
                batch_size, batch_n_id, batch_edge_index, batch_exist_nodes,
                batch_clustering_coefficient, batch_bidirectional_links_ratio,
            )
            all_output.append(output)
            all_label.append(label_list)
            all_exist.append(exist_nodes_list)
        all_output = torch.cat(all_output, dim=1)
        all_label = torch.cat(all_label, dim=1)
        all_exist = torch.cat(all_exist, dim=1)

        last_output = all_output[-1]
        last_label = all_label[-1]
        last_exist = all_exist[-1]

        exist_mask = last_exist == 1
        logits = last_output[exist_mask]
        labels = last_label[exist_mask]
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        val_preds = preds.cpu().numpy()
        val_probs_list = probs[:, 1].cpu().numpy()
        val_labels = labels.cpu().numpy()

    val_global_indices = dataset.val_idx.cpu().numpy()
    val_uids = [global2uid.get(int(idx), f"unknown_{idx}") for idx in val_global_indices]

    val_predictions_df = pd.DataFrame({
        "experiment": experiment_name,
        "family": "feature_text_graph",
        "split": "val",
        "user_id": val_uids[:len(val_labels)],
        "true_label": val_labels.tolist(),
        "pred_label": val_preds.tolist(),
        "bot_probability": val_probs_list.tolist(),
    })

    predictions_df = pd.concat([predictions_df, val_predictions_df], ignore_index=True)

    val_f1 = trainer.best_val_f1
    artifact_path = config.models_dir / f"{experiment_name}.pt"
    torch.save({"state_dict": best_state, "best_val_f1": val_f1}, artifact_path)

    metrics_rows = [{
        "experiment": experiment_name,
        "family": "feature_text_graph",
        "split": "test",
        "best_threshold": 0.5,
        "selected_threshold": 0.5,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "auc_roc": float("nan"),
    }, {
        "experiment": experiment_name,
        "family": "feature_text_graph",
        "split": "val",
        "best_threshold": 0.5,
        "selected_threshold": 0.5,
        "accuracy": trainer.best_val_metrics["accuracy"],
        "precision": trainer.best_val_metrics["precision"],
        "recall": trainer.best_val_metrics["recall"],
        "f1": trainer.best_val_metrics["f1"],
        "auc_roc": float("nan"),
    }]

    return {
        "metrics_rows": metrics_rows,
        "predictions": predictions_df,
        "best_val_f1": val_f1,
        "artifact_path": artifact_path,
        "training_history": pd.DataFrame(trainer.history_rows).assign(experiment=experiment_name),
        "ablation_mode": ablation_mode,
        "removed_modality": BOTDGT_ABLATION_LABELS[ablation_mode],
    }
