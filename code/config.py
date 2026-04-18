from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
    transformer_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
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
    gnn_learning_rate: float = 1e-2
    gnn_weight_decay: float = 5e-2
    gnn_dropout: float = 0.1
    run_dynamic_fusion: bool = True
    adaptive_gate_hidden_dim: int = 96
    adaptive_fusion_temperature: float = 1.0
    temporal_snapshot_count: int = 4
    temporal_transformer_heads: int = 4
    temporal_transformer_layers: int = 2

    visualization_sample_size: int = 3000
    random_state: int = 42

    cache_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    tables_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.cache_dir = self.output_dir / "cache"
        self.models_dir = self.output_dir / "models"
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir = self.output_dir / "figures"

    def ensure_directories(self) -> None:
        for path in (self.output_dir, self.cache_dir, self.models_dir, self.tables_dir, self.figures_dir):
            path.mkdir(parents=True, exist_ok=True)


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()
