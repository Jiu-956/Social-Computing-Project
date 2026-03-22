from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PipelineConfig:
    data_dir: Path
    output_dir: Path = Path("artifacts")
    max_labeled_users: int | None = None
    max_tweets_per_user: int = 8
    random_state: int = 42
    tfidf_max_features: int = 8000
    tfidf_min_df: int = 3
    deepwalk_dimensions: int = 64
    deepwalk_walk_length: int = 20
    deepwalk_num_walks: int = 5
    deepwalk_window: int = 5
    deepwalk_epochs: int = 5
    betweenness_sample_k: int = 128
    harmonic_sample_sources: int = 128
    tsne_sample_size: int = 3000

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
