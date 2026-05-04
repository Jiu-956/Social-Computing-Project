from __future__ import annotations

import logging
import os
import pickle
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ...config import ProjectConfig
from ..botdgt.data import BotDGTDataset, _DATA_DIR
from .loss import composite_loss
from .model import TIGNv2Model

LOGGER = logging.getLogger(__name__)


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
    cross_modal_weight: float
    temporal_inv_weight: float
    specific_decorr_weight: float
    embedding_dropout: float
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


class TIGNv2Trainer:
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

        self.model = TIGNv2Model(self.args)
        self.model.to(self.args.device)

        params = [
            {"params": self.model.modality_encoders.parameters(),
             "lr": self.args.structural_learning_rate},
            {"params": self.model.channel_fusion.parameters(),
             "lr": self.args.structural_learning_rate},
            {"params": self.model.structural_layer.parameters(),
             "lr": self.args.structural_learning_rate},
            {"params": self.model.pos_clustering.parameters(),
             "lr": self.args.structural_learning_rate},
            {"params": self.model.pos_blr.parameters(),
             "lr": self.args.structural_learning_rate},
            {"params": self.model.invariant_gru.parameters(),
             "lr": self.args.temporal_learning_rate},
            {"params": self.model.specific_gru.parameters(),
             "lr": self.args.temporal_learning_rate},
            {"params": self.model.cross_stream_attn.parameters(),
             "lr": self.args.temporal_learning_rate},
            {"params": self.model.cross_stream_norm.parameters(),
             "lr": self.args.temporal_learning_rate},
            {"params": self.model.temporal_fusion.parameters(),
             "lr": self.args.temporal_learning_rate},
            {"params": self.model.temporal_pos_embedding.parameters(),
             "lr": self.args.temporal_learning_rate},
            {"params": self.model.output_mlp.parameters(),
             "lr": self.args.temporal_learning_rate},
            {"params": self.model.output_head.parameters(),
             "lr": self.args.temporal_learning_rate},
        ]
        self.optimizer = torch.optim.AdamW(params, weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.epoch, eta_min=0,
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

        output, aux = self.model(
            des_tensor_list, tweet_tensor_list, num_prop_list, category_prop_list,
            edge_index_list, clustering_coefficient_list, bidirectional_links_ratio_list,
            exist_nodes_list, batch_size,
        )
        # output: [T, B, 2], aux: {"invariant_stacks": [B, T, 4, block_dim], ...}
        loss, loss_components = composite_loss(
            self.criterion, output, label_list, exist_nodes_list,
            aux["invariant_stacks"], aux["specific_stacks"],
            coefficient=self.args.coefficient,
            cross_modal_weight=self.args.cross_modal_weight,
            temporal_inv_weight=self.args.temporal_inv_weight,
            specific_decorr_weight=self.args.specific_decorr_weight,
        )
        return output, loss, loss_components, label_list, exist_nodes_list

    def forward_one_epoch(self, right, n_id, edge_index, exist_nodes, clustering_coefficient,
                           bidirectional_links_ratio):
        all_label = []
        all_output = []
        all_exist_nodes = []
        total_loss = 0.0
        for batch_size, batch_n_id, batch_edge_index, batch_exist_nodes, \
            batch_clustering_coefficient, batch_bidirectional_links_ratio in \
                zip(right, n_id, edge_index, exist_nodes, clustering_coefficient, bidirectional_links_ratio):
            output, loss, _, label_list, exist_nodes_list = self.forward_one_batch(
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
        total_components = {}
        num_batches = 0
        for batch_size, batch_n_id, batch_edge_index, batch_exist_nodes, \
            batch_clustering_coefficient, batch_bidirectional_links_ratio in \
                zip(self.train_right, self.train_n_id, self.train_edge_index, self.train_exist_nodes,
                    self.train_clustering_coefficient, self.train_bidirectional_links_ratio):
            self.optimizer.zero_grad()
            output, loss, loss_components, label_list, exist_nodes_list = self.forward_one_batch(
                batch_size, batch_n_id, batch_edge_index, batch_exist_nodes,
                batch_clustering_coefficient, batch_bidirectional_links_ratio,
            )
            total_loss += loss.item() / self.args.window_size / len(self.train_right)
            for k, v in loss_components.items():
                total_components[k] = total_components.get(k, 0.0) + v
            num_batches += 1
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

        if num_batches > 0:
            ce = total_components.get("ce_loss", 0) / num_batches
            cm = total_components.get("cross_modal_inv", 0) / num_batches
            ct = total_components.get("temporal_inv", 0) / num_batches
            sd = total_components.get("specific_decorr", 0) / num_batches
            LOGGER.debug("train CE:%.4f CM:%.4f CT:%.4f SD:%.4f", ce, cm, ct, sd)
        return metrics

    @torch.no_grad()
    def val_per_epoch(self):
        self.model.eval()
        metrics = self.forward_one_epoch(
            self.val_right, self.val_n_id, self.val_edge_index, self.val_exist_nodes,
            self.val_clustering_coefficient, self.val_bidirectional_links_ratio,
        )
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
            train_metrics = self.train_per_epoch(current_epoch)
            self.scheduler.step()
            val_metrics = self.val_per_epoch()

            self.history_rows.append({
                "epoch": current_epoch,
                "train_loss": train_metrics["loss"],
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

            patience_left = self.args.patience - validate_score_non_improvement_count
            LOGGER.info(
                "[feature_text_graph_tignv2] epoch %03d/%03d train_loss=%.4f val_loss=%.4f train_f1=%.4f val_f1=%.4f best_val_f1=%.4f patience_left=%d",
                current_epoch, self.args.epoch,
                train_metrics["loss"], val_metrics["loss"],
                train_metrics["f1"], val_metrics["f1"],
                self.best_val_f1, patience_left,
            )

            if self.args.early_stop and validate_score_non_improvement_count >= self.args.patience:
                LOGGER.info("[feature_text_graph_tignv2] Early stopping at epoch %d", current_epoch)
                break

        best_state = self.test_state_dict_list[-1] if self.test_state_dict_list else self.last_state_dict
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        test_metrics = self.test_model(state_dict=best_state)

        model_dir = self.args.dataset_name
        os.makedirs(model_dir, exist_ok=True)
        model_name = f"tignv2_{self.args.interval}_{self.args.seed}_{test_metrics['accuracy']:.4f}_{test_metrics['f1']:.4f}.pt"
        torch.save(best_state, os.path.join(model_dir, model_name))
        LOGGER.info("Saved model: %s", model_name)

        return test_metrics, best_state


def run_tignv2(config: ProjectConfig) -> dict:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    LOGGER.info("=== TIGN-v2: Loading BotDGT dataset (calendar-month snapshots) ===")
    dataset = BotDGTDataset(config, interval=config.tignv2_interval,
                            batch_size=config.tignv2_batch_size)

    args = _Args(
        hidden_dim=config.gnn_hidden_dim,
        structural_head_config=config.tignv2_structural_heads,
        structural_drop=config.tignv2_structural_dropout,
        temporal_head_config=config.tignv2_temporal_heads,
        temporal_drop=config.tignv2_temporal_dropout,
        window_size=dataset.window_size,
        temporal_module_type=config.botdgt_temporal_module,
        structural_learning_rate=config.tignv2_structural_lr,
        temporal_learning_rate=config.tignv2_temporal_lr,
        weight_decay=config.tignv2_weight_decay,
        epoch=config.tignv2_epochs,
        seed=config.random_state,
        device=device_str,
        dataset_name="Twibot-20",
        interval=config.tignv2_interval,
        batch_size=config.tignv2_batch_size,
        coefficient=config.tignv2_loss_coefficient,
        cross_modal_weight=config.tignv2_cross_modal_weight,
        temporal_inv_weight=config.tignv2_temporal_invariance_weight,
        specific_decorr_weight=config.tignv2_specific_decorr_weight,
        embedding_dropout=config.tignv2_embedding_dropout,
        early_stop=True,
        patience=config.tignv2_patience,
    )

    LOGGER.info("=== TIGN-v2: Starting training ===")
    LOGGER.info(
        "structural_lr=%.1e temporal_lr=%.1e epochs=%d batch_size=%d interval=%s window_size=%d",
        args.structural_learning_rate, args.temporal_learning_rate,
        args.epoch, args.batch_size, args.interval, args.window_size,
    )
    LOGGER.info(
        "cross_modal_weight=%.4f temporal_inv_weight=%.4f specific_decorr_weight=%.4f",
        args.cross_modal_weight, args.temporal_inv_weight, args.specific_decorr_weight,
    )
    trainer = TIGNv2Trainer(args, dataset)
    test_metrics, best_state = trainer.train()

    LOGGER.info(
        "=== TIGN-v2: Test results — acc=%.5f f1=%.5f precision=%.5f recall=%.5f ===",
        test_metrics["accuracy"], test_metrics["f1"],
        test_metrics["precision"], test_metrics["recall"],
    )

    # Build predictions
    with open(str(_DATA_DIR / "processed_data" / "uid2global_index.pkl"), "rb") as f:
        uid2global = pickle.load(f)
    global2uid = {v: k for k, v in uid2global.items()}

    model = trainer.model
    model.eval()

    def _build_predictions(split_name):
        right = getattr(dataset, f"{split_name}_right")
        n_id = getattr(dataset, f"{split_name}_n_id")
        edge_index = getattr(dataset, f"{split_name}_edge_index")
        exist_nodes = getattr(dataset, f"{split_name}_exist_nodes")
        clustering = getattr(dataset, f"{split_name}_clustering_coefficient")
        blr = getattr(dataset, f"{split_name}_bidirectional_links_ratio")
        idx = getattr(dataset, f"{split_name}_idx")

        with torch.no_grad():
            all_output, all_label, all_exist = [], [], []
            for batch_size, batch_n_id, batch_edge_index, batch_exist_nodes, \
                batch_clustering_coefficient, batch_bidirectional_links_ratio in \
                    zip(right, n_id, edge_index, exist_nodes, clustering, blr):
                output, _, _, label_list, exist_nodes_list = trainer.forward_one_batch(
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

        preds_np = preds.cpu().numpy()
        probs_np = probs[:, 1].cpu().numpy()
        labels_np = labels.cpu().numpy()

        global_indices = idx.cpu().numpy()
        uids = [global2uid.get(int(idx), f"unknown_{idx}") for idx in global_indices]

        return pd.DataFrame({
            "experiment": "feature_text_graph_tignv2",
            "family": "feature_text_graph",
            "split": split_name,
            "user_id": uids[:len(labels_np)],
            "true_label": labels_np.tolist(),
            "pred_label": preds_np.tolist(),
            "bot_probability": probs_np.tolist(),
        })

    test_preds = _build_predictions("test")
    val_preds = _build_predictions("val")
    predictions_df = pd.concat([test_preds, val_preds], ignore_index=True)

    val_f1 = trainer.best_val_f1
    artifact_path = config.models_dir / "feature_text_graph_tignv2.pt"
    torch.save({"state_dict": best_state, "best_val_f1": val_f1}, artifact_path)

    metrics_rows = [{
        "experiment": "feature_text_graph_tignv2",
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
        "experiment": "feature_text_graph_tignv2",
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
        "training_history": pd.DataFrame(trainer.history_rows),
    }
