from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import joblib
import pandas as pd

if TYPE_CHECKING:
    from ...config import ProjectConfig

from ...config import safe_slug

LOGGER = logging.getLogger(__name__)


def compute_transformer_embeddings(
    config: "ProjectConfig",
    text_df: pd.DataFrame,
    cache_name: str,
    prefix: str,
) -> pd.DataFrame | None:
    cache_path = config.cache_dir / f"{cache_name}_{safe_slug(config.transformer_model_name)}.joblib"
    if cache_path.exists():
        try:
            cached = joblib.load(cache_path)
        except Exception:
            LOGGER.warning("Failed to load transformer cache (likely pandas version incompatibility). Recomputing embeddings.")
            cache_path.unlink(missing_ok=True)
            cached = None
        if cached is not None and isinstance(cached, pd.DataFrame) and _is_valid_embedding_cache(cached, text_df["user_id"], prefix=prefix):
            LOGGER.info("Transformer cache hit: %s (%d users)", cache_name, len(cached))
            return cached
        # Cache exists but validation failed - try a more lenient check to avoid unnecessary recompute
        if cached is not None and isinstance(cached, pd.DataFrame) and "user_id" in cached.columns:
            cached_ids = set(cached["user_id"].astype(str).tolist())
            expected_ids = set(text_df["user_id"].astype(str).tolist())
            if cached_ids == expected_ids:
                emb_cols = [c for c in cached.columns if c.startswith(prefix)]
                has_nan = False
                if emb_cols:
                    try:
                        has_nan = cached[emb_cols].isna().any().any()
                    except Exception:
                        has_nan = True
                if not has_nan:
                    LOGGER.info("Transformer cache hit (lenient): %s (%d users)", cache_name, len(cached))
                    return cached
        LOGGER.warning("Existing transformer cache does not match the current prepared dataset. Recomputing embeddings.")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("SentenceTransformer is unavailable: %s", exc)
        return None

    try:
        model = SentenceTransformer(config.transformer_model_name)
        embeddings = model.encode(
            text_df["text"].fillna("").tolist(),
            batch_size=config.transformer_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Skipping transformer experiment because the model could not be loaded: %s", exc)
        return None

    frame = pd.DataFrame(embeddings, columns=[f"{prefix}{index}" for index in range(embeddings.shape[1])])
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
    embedding_columns = [column for column in frame.columns if column.startswith(prefix)]
    expected_ids = set(expected_user_ids.astype(str).tolist())
    cached_ids = set(frame["user_id"].astype(str).tolist())
    if expected_ids != cached_ids:
        return False
    if frame["user_id"].duplicated().any():
        return False
    if embedding_columns and frame[embedding_columns].isna().any().any():
        return False
    return True
