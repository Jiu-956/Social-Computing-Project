from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from torch_geometric.loader import NeighborLoader

from ...config import ProjectConfig

LOGGER = logging.getLogger(__name__)

_DATA_DIR = Path("data")


def _load_graphs(interval: str) -> tuple[list, list[str]]:
    interval_dict = {
        "year": 12, "month": 1, "three_months": 3, "six_months": 6,
        "9_months": 9, "18_months": 18, "15_months": 15,
        "21_months": 21, "24_months": 24,
    }
    assert interval in interval_dict, f"Unknown interval: {interval}"

    graph_dir = _DATA_DIR / "graph_data" / "graphs"
    files = sorted(os.listdir(str(graph_dir)))
    selected = []
    for idx in range(-1, -len(files) - 1, -interval_dict[interval]):
        selected.append(files[idx])
    selected.reverse()
    LOGGER.info("Selected %d snapshots (interval=%s): %s ... %s",
                 len(selected), interval, selected[0], selected[-1])
    graphs = [torch.load(str(graph_dir / f), weights_only=False) for f in selected]
    return graphs, selected


def _load_split_index() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pdir = _DATA_DIR / "processed_data"
    train_idx = torch.load(str(pdir / "train_idx.pt"), weights_only=True)
    val_idx = torch.load(str(pdir / "val_idx.pt"), weights_only=True)
    test_idx = torch.load(str(pdir / "test_idx.pt"), weights_only=True)
    return train_idx, val_idx, test_idx


def _load_labels() -> torch.Tensor:
    return torch.load(str(_DATA_DIR / "processed_data" / "label.pt"), weights_only=True)


def _load_features() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pdir = _DATA_DIR / "processed_data"
    des_tensor = torch.load(str(pdir / "des_tensor.pt"), weights_only=True)
    tweets_tensor = torch.load(str(pdir / "tweets_tensor.pt"), weights_only=True)
    num_prop = torch.load(str(pdir / "num_properties_tensor.pt"), weights_only=True)
    category_prop = torch.load(str(pdir / "cat_properties_tensor.pt"), weights_only=True)
    return des_tensor, tweets_tensor, num_prop, category_prop


def _load_cached_batches(interval: str, batch_size: int, seed: int, split: str):
    cache_dir = _DATA_DIR / "final_data" / interval / f"batch-size-{batch_size}" / f"seed-{seed}" / split
    if not cache_dir.exists():
        return None
    result = {}
    for name in ["right", "n_id", "edge_index", "edge_type", "exist_nodes",
                  "clustering_coefficient", "bidirectional_links_ratio"]:
        path = cache_dir / f"all_{name}.pt"
        if path.exists():
            result[name] = torch.load(str(path), weights_only=False)
        else:
            return None
    LOGGER.info("Loaded cached %s batches from %s", split, cache_dir)
    return result


BOTDGT_ABLATION_MODES = ("full", "no_profile", "no_text", "no_graph")


class BotDGTDataset:
    def __init__(self, config: ProjectConfig, *, interval: str | None = None,
                 batch_size: int | None = None, window_size: int | None = None,
                 ablation_mode: str = "full"):
        if ablation_mode not in BOTDGT_ABLATION_MODES:
            raise ValueError(f"Unknown BotDGT ablation mode: {ablation_mode}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.interval = interval if interval is not None else config.botdgt_interval
        self.batch_size = batch_size if batch_size is not None else config.botdgt_batch_size
        self.seed = config.random_state
        self.window_size = -1
        self.ablation_mode = ablation_mode

        # Load graphs with interval selection
        self.graphs, _ = _load_graphs(self.interval)
        self.graphs = [g.to(self.device) for g in self.graphs]

        _ws = window_size if window_size is not None else config.botdgt_window_size
        if _ws > 0 and len(self.graphs) > _ws:
            self.graphs = self.graphs[-_ws:]
            self.window_size = _ws
        else:
            self.window_size = len(self.graphs)

        # Load splits, labels, features
        self.train_idx = torch.load(
            str(_DATA_DIR / "processed_data" / "train_idx.pt"), weights_only=True).to(self.device)
        self.val_idx = torch.load(
            str(_DATA_DIR / "processed_data" / "val_idx.pt"), weights_only=True).to(self.device)
        self.test_idx = torch.load(
            str(_DATA_DIR / "processed_data" / "test_idx.pt"), weights_only=True).to(self.device)

        self.labels = _load_labels()
        self.des_tensor, self.tweets_tensor, self.num_prop, self.category_prop = _load_features()

        # Pad labels for all 229580 nodes (matching reference: unlabeled = 3)
        if len(self.labels) < 229580:
            self.labels = torch.cat(
                (self.labels, 3 * torch.ones(229580 - len(self.labels), dtype=torch.long)), dim=0)

        self.labels = self.labels.to(self.device)
        self.des_tensor = self.des_tensor.to(self.device)
        self.tweets_tensor = self.tweets_tensor.to(self.device)
        self.num_prop = self.num_prop.to(self.device)
        self.category_prop = self.category_prop.to(self.device)
        self._apply_feature_ablation()

        LOGGER.info(
            "BotDGTDataset: %d nodes, %d snapshots, interval=%s, train=%d val=%d test=%d",
            229580, self.window_size, self.interval,
            len(self.train_idx), len(self.val_idx), len(self.test_idx),
        )

        # Load or build NeighborLoader batches
        self._load_or_build_batches("train")
        self._load_or_build_batches("val")
        self._load_or_build_batches("test")
        self._apply_graph_ablation()

    def _apply_feature_ablation(self) -> None:
        if self.ablation_mode == "no_profile":
            self.num_prop = torch.zeros_like(self.num_prop)
            self.category_prop = torch.zeros_like(self.category_prop)
        elif self.ablation_mode == "no_text":
            self.des_tensor = torch.zeros_like(self.des_tensor)
            self.tweets_tensor = torch.zeros_like(self.tweets_tensor)

    def _apply_graph_ablation(self) -> None:
        if self.ablation_mode != "no_graph":
            return
        for split in ("train", "val", "test"):
            edge_batches = getattr(self, f"{split}_edge_index")
            clustering_batches = getattr(self, f"{split}_clustering_coefficient")
            bidirectional_batches = getattr(self, f"{split}_bidirectional_links_ratio")
            setattr(
                self,
                f"{split}_edge_index",
                [
                    [torch.empty((2, 0), dtype=edge_index.dtype) for edge_index in batch]
                    for batch in edge_batches
                ],
            )
            setattr(
                self,
                f"{split}_clustering_coefficient",
                [[torch.zeros_like(values) for values in batch] for batch in clustering_batches],
            )
            setattr(
                self,
                f"{split}_bidirectional_links_ratio",
                [[torch.zeros_like(values) for values in batch] for batch in bidirectional_batches],
            )

    def _load_or_build_batches(self, split: str):
        cached = _load_cached_batches(self.interval, self.batch_size, self.seed, split)
        if cached is not None:
            for key, val in cached.items():
                setattr(self, f"{split}_{key}", val)
            return

        LOGGER.info("Building NeighborLoader batches for %s (no cache found)...", split)
        input_nodes = getattr(self, f"{split}_idx")
        shuffle = (split == "train")
        num_neighbors = [2560, 2560] if split == "train" else [-1, -1]

        loaders = [
            NeighborLoader(
                graph, shuffle=shuffle,
                generator=torch.Generator().manual_seed(self.seed),
                batch_size=self.batch_size, input_nodes=input_nodes,
                num_neighbors=num_neighbors,
            )
            for graph in self.graphs
        ]
        iters = [iter(loader) for loader in loaders]

        all_right = []
        all_n_id = []
        all_edge_index = []
        all_edge_type = []
        all_exist_nodes = []
        all_clustering = []
        all_blr = []

        total = len(input_nodes)
        for i in range(0, total, self.batch_size):
            right = min(self.batch_size, total - i)
            all_right.append(right)
            try:
                subgraphs = [next(it) for it in iters]
            except StopIteration:
                break
            all_n_id.append([sg.n_id.to("cpu") for sg in subgraphs])
            all_edge_index.append([sg.edge_index.to("cpu") for sg in subgraphs])
            all_edge_type.append([sg.edge_type.to("cpu") for sg in subgraphs])
            all_exist_nodes.append([sg.exist_nodes.to("cpu") for sg in subgraphs])
            all_clustering.append([sg.clustering_coefficient.to("cpu") for sg in subgraphs])
            all_blr.append([sg.bidirectional_links_ratio.to("cpu") for sg in subgraphs])

        setattr(self, f"{split}_right", all_right)
        setattr(self, f"{split}_n_id", all_n_id)
        setattr(self, f"{split}_edge_index", all_edge_index)
        setattr(self, f"{split}_edge_type", all_edge_type)
        setattr(self, f"{split}_exist_nodes", all_exist_nodes)
        setattr(self, f"{split}_clustering_coefficient", all_clustering)
        setattr(self, f"{split}_bidirectional_links_ratio", all_blr)
