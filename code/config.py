from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


RAW_DATA_FILES = ("edge.csv", "label.csv", "split.csv", "node.json")


def _contains_raw_dataset_files(path: Path) -> bool:
    return all((path / file_name).exists() for file_name in RAW_DATA_FILES)


def resolve_data_dir(data_dir: Path) -> Path:
    candidate = data_dir.expanduser()
    if _contains_raw_dataset_files(candidate):
        return candidate
    nested_raw_dir = candidate / "raw"
    if _contains_raw_dataset_files(nested_raw_dir):
        return nested_raw_dir
    return candidate


@dataclass(slots=True)
class ProjectConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("artifacts")
    max_graph_users: int | None = None
    max_tweets_per_user: int = 12

    tfidf_max_features: int = 10000
    tfidf_min_df: int = 3
    dense_text_max_features: int = 12000
    dense_text_svd_dim: int = 64

    use_transformer: bool = True
    transformer_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    transformer_batch_size: int = 32
    transformer_max_length: int = 128

    run_node2vec: bool = True
    node2vec_dimensions: int = 64
    node2vec_walk_length: int = 20
    node2vec_num_walks: int = 6
    node2vec_window: int = 5
    node2vec_epochs: int = 5
    node2vec_return_p: float = 1.0
    node2vec_inout_q: float = 2.0
    node2vec_workers: int = 4

    run_gnn: bool = True
    gnn_hidden_dim: int = 128
    gnn_epochs: int = 50
    gnn_patience: int = 10
    gnn_learning_rate: float = 1e-3
    gnn_weight_decay: float = 5e-4
    gnn_dropout: float = 0.1
    botsai_invariant_weight: float = 0.05
    botsai_attention_heads: int = 4
    botdgt_snapshot_count: int = 8
    botdgt_min_keep_ratio: float = 0.15
    botdgt_temporal_module: str = "attention"
    botdgt_temporal_heads: int = 4
    botdgt_temporal_smoothness_weight: float = 0.05
    botdgt_temporal_consistency_weight: float = 0.03
    botdgt_structural_lr: float = 1e-4
    botdgt_temporal_lr: float = 1e-5
    botdgt_structural_dropout: float = 0.0
    botdgt_temporal_dropout: float = 0.5
    botdgt_weight_decay: float = 1e-2
    botdgt_loss_coefficient: float = 1.1
    botdgt_epochs: int = 20
    # BotDGT new-module parameters (matching reference)
    botdgt_batch_size: int = 64
    botdgt_interval: str = "year"
    botdgt_structural_heads: int = 4
    botdgt_window_size: int = -1
    botdgt_embedding_dropout: float = 0.3
    tign_num_age_buckets: int = 3
    tign_intra_class_weight: float = 0.02

    visualization_sample_size: int = 3000
    random_state: int = 1234

    cache_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    tables_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = resolve_data_dir(Path(self.data_dir))
        self.output_dir = Path(self.output_dir)
        self.cache_dir = self.output_dir / "cache"
        self.logs_dir = self.output_dir / "logs"
        self.models_dir = self.output_dir / "models"
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir = self.output_dir / "figures"

    def ensure_directories(self) -> None:
        for path in (self.output_dir, self.cache_dir, self.logs_dir, self.models_dir, self.tables_dir, self.figures_dir):
            path.mkdir(parents=True, exist_ok=True)


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()
