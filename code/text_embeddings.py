from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TextEmbeddingResult:
    frame: pd.DataFrame
    encoder_name: str


def _slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")
    return slug or "transformer"


def _text_embedding_cache_path(config: PipelineConfig) -> Path:
    slug = _slugify_model_name(config.transformer_model_name)
    return config.cache_dir / f"text_embeddings_{slug}.joblib"


def load_or_compute_text_embeddings(config: PipelineConfig, users: pd.DataFrame) -> TextEmbeddingResult:
    cache_path = _text_embedding_cache_path(config)
    user_ids = users["user_id"].astype(str).tolist()

    if cache_path.exists():
        payload = joblib.load(cache_path)
        encoder_name = str(payload.get("encoder_name", "unknown"))
        if payload.get("user_id") == user_ids and encoder_name.startswith("transformer:"):
            frame = pd.DataFrame(payload["embeddings"], columns=payload["columns"])
            frame.insert(0, "user_id", payload["user_id"])
            return TextEmbeddingResult(frame=frame, encoder_name=encoder_name)

    texts = users["combined_text"].fillna("").astype(str).tolist()
    LOGGER.info("Encoding texts with transformer model %s", config.transformer_model_name)
    embeddings = compute_transformer_embeddings(
        texts=texts,
        model_name=config.transformer_model_name,
        batch_size=config.transformer_batch_size,
        max_length=config.transformer_max_length,
    )
    encoder_name = f"transformer:{config.transformer_model_name}"

    columns = [f"te_{idx}" for idx in range(embeddings.shape[1])]
    payload = {
        "encoder_name": encoder_name,
        "model_name": config.transformer_model_name,
        "user_id": user_ids,
        "columns": columns,
        "embeddings": embeddings.astype(np.float32),
    }
    joblib.dump(payload, cache_path)

    frame = pd.DataFrame(embeddings, columns=columns)
    frame.insert(0, "user_id", user_ids)
    return TextEmbeddingResult(frame=frame, encoder_name=encoder_name)


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

    model_source = resolve_transformer_source(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_source))
    model = AutoModel.from_pretrained(str(model_source)).to(device)
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


def resolve_transformer_source(model_name: str) -> Path:
    explicit_path = Path(model_name)
    if explicit_path.exists():
        return explicit_path

    cached_path = resolve_cached_transformer_source(model_name)
    if cached_path is not None:
        return cached_path

    from huggingface_hub import snapshot_download

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            LOGGER.info("Downloading transformer model %s from Hugging Face (attempt %s/3)", model_name, attempt)
            return Path(snapshot_download(repo_id=model_name))
        except Exception as exc:
            last_error = exc
            LOGGER.warning("Transformer model download attempt %s/3 failed: %s", attempt, exc)
            if attempt < 3:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"Failed to download transformer model '{model_name}' after 3 attempts. "
        "Check network connectivity to Hugging Face, or configure a proxy / mirror such as HF_ENDPOINT."
    ) from last_error


def resolve_cached_transformer_source(model_name: str) -> Path | None:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    candidates = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), key=lambda item: item.name, reverse=True)
    for candidate in candidates:
        if has_complete_transformer_artifacts(candidate):
            return candidate
    return None


def has_complete_transformer_artifacts(path: Path) -> bool:
    files = {file.name for file in path.rglob("*") if file.is_file()}
    has_config = "config.json" in files
    has_tokenizer = bool({"tokenizer.json", "vocab.txt", "sentencepiece.bpe.model"} & files)
    has_weights = bool({"model.safetensors", "pytorch_model.bin", "tf_model.h5", "flax_model.msgpack"} & files)
    return has_config and has_tokenizer and has_weights


def mean_pool(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    masked = hidden_states * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts
