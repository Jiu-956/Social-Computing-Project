from __future__ import annotations

import logging
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)


def _slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")
    return slug or "transformer"


def _text_embedding_cache_path(config: PipelineConfig) -> Path:
    slug = _slugify_model_name(config.transformer_model_name)
    return config.cache_dir / f"text_embeddings_{slug}.joblib"


def load_or_compute_text_embeddings(config: PipelineConfig, users: pd.DataFrame) -> pd.DataFrame:
    cache_path = _text_embedding_cache_path(config)
    user_ids = users["user_id"].astype(str).tolist()

    if cache_path.exists():
        payload = joblib.load(cache_path)
        if payload.get("user_id") == user_ids:
            frame = pd.DataFrame(payload["embeddings"], columns=payload["columns"])
            frame.insert(0, "user_id", payload["user_id"])
            return frame

    LOGGER.info("Encoding texts with transformer model %s", config.transformer_model_name)
    embeddings = compute_transformer_embeddings(
        texts=users["combined_text"].fillna("").astype(str).tolist(),
        model_name=config.transformer_model_name,
        batch_size=config.transformer_batch_size,
        max_length=config.transformer_max_length,
    )
    columns = [f"st_{idx}" for idx in range(embeddings.shape[1])]
    payload = {
        "model_name": config.transformer_model_name,
        "user_id": user_ids,
        "columns": columns,
        "embeddings": embeddings.astype(np.float32),
    }
    joblib.dump(payload, cache_path)

    frame = pd.DataFrame(embeddings, columns=columns)
    frame.insert(0, "user_id", user_ids)
    return frame


def compute_transformer_embeddings(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    output_batches: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        output_batches.append(pooled.cpu().numpy().astype(np.float32))

    return np.vstack(output_batches)


def mean_pool(hidden_states, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    masked = hidden_states * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts
