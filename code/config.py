from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PipelineConfig:
    data_dir: Path
    output_dir: Path = Path("result")
    max_labeled_users: int | None = None
    max_tweets_per_user: int = 8
    random_state: int = 42
    logreg_max_iter: int = 4000
    tfidf_max_features: int = 8000
    tfidf_min_df: int = 3
    transformer_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    transformer_batch_size: int = 32
    transformer_max_length: int = 128
    deepwalk_dimensions: int = 64
    deepwalk_walk_length: int = 20
    deepwalk_num_walks: int = 5
    deepwalk_window: int = 5
    deepwalk_epochs: int = 5
    node2vec_dimensions: int = 64
    node2vec_walk_length: int = 20
    node2vec_num_walks: int = 5
    node2vec_window: int = 5
    node2vec_epochs: int = 5
    node2vec_return_p: float = 1.0
    node2vec_inout_q: float = 2.0
    betweenness_sample_k: int = 128
    harmonic_sample_sources: int = 128
    tsne_sample_size: int = 3000
    gnn_hidden_dim: int = 128
    gnn_epochs: int = 150
    gnn_patience: int = 20
    gnn_learning_rate: float = 0.01
    gnn_weight_decay: float = 5e-4
    gnn_dropout: float = 0.35

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "cache"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    def ensure_dirs(self) -> None:
        for path in (self.output_dir, self.cache_dir, self.models_dir, self.figures_dir, self.tables_dir):
            path.mkdir(parents=True, exist_ok=True)
