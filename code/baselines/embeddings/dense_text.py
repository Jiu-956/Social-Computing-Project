from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

if TYPE_CHECKING:
    from ...config import ProjectConfig

LOGGER = logging.getLogger(__name__)


def compute_dense_text_embeddings(
    config: "ProjectConfig",
    text_df: pd.DataFrame,
    cache_name: str,
    prefix: str,
) -> pd.DataFrame:
    cache_path = config.cache_dir / f"{cache_name}.joblib"
    if cache_path.exists():
        cached = joblib.load(cache_path)
        if isinstance(cached, pd.DataFrame) and _is_valid_embedding_cache(cached, text_df["user_id"], prefix=prefix):
            return cached
        LOGGER.warning("Existing dense text cache does not match the current prepared dataset. Recomputing embeddings.")

    vectorizer = TfidfVectorizer(
        max_features=config.dense_text_max_features,
        min_df=2 if len(text_df) > 100 else 1,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(text_df["text"].fillna(""))
    if matrix.shape[1] == 0:
        dense = np.zeros((len(text_df), 1), dtype=np.float32)
    else:
        n_components = min(config.dense_text_svd_dim, max(1, matrix.shape[1] - 1))
        if n_components <= 1:
            dense = matrix[:, :1].toarray().astype(np.float32)
        else:
            dense = TruncatedSVD(n_components=n_components, random_state=config.random_state).fit_transform(matrix).astype(np.float32)

    frame = pd.DataFrame(dense, columns=[f"{prefix}{index}" for index in range(dense.shape[1])])
    frame.insert(0, "user_id", text_df["user_id"].to_numpy())
    joblib.dump(frame, cache_path)
    return frame


def _is_valid_embedding_cache(
    frame: pd.DataFrame,
    expected_user_ids: pd.Series,
    prefix: str,
) -> bool:
    if "user_id" not in frame.columns:
        return False
    expected_ids = set(expected_user_ids.astype(str).tolist())
    cached_ids = set(frame["user_id"].astype(str).tolist())
    if expected_ids != cached_ids:
        return False
    if frame["user_id"].duplicated().any():
        return False
    embedding_columns = [column for column in frame.columns if column.startswith(prefix)]
    if embedding_columns and frame[embedding_columns].isna().any().any():
        return False
    return True
