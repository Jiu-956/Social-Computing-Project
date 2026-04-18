from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dafbot.utils import clean_text, dump_json


def _compose_text(description: str, tweet_text: str, empty_token: str, separator: str) -> str:
    description = clean_text(description)
    tweet_text = clean_text(tweet_text)
    if not description:
        description = empty_token
    if not tweet_text:
        tweet_text = empty_token
    tweets = separator.join(part.strip() for part in tweet_text.split(separator) if part.strip()) if separator in tweet_text else tweet_text
    return f"[DESC] {description} [TWEETS] {tweets}".strip()


def _encode_with_sentence_transformer(texts: list[str], config: dict[str, object]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config["model_name"])
    embeddings = model.encode(
        texts,
        batch_size=int(config["batch_size"]),
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _encode_with_tfidf_svd(texts: list[str], config: dict[str, object]) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=int(config["max_features"]),
        min_df=int(config["min_df"]),
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape[1] == 0:
        return np.zeros((len(texts), 1), dtype=np.float32)

    max_components = min(int(config["svd_dim"]), max(1, matrix.shape[1] - 1))
    if max_components <= 1:
        return matrix[:, :1].toarray().astype(np.float32)

    svd = TruncatedSVD(n_components=max_components, random_state=42)
    dense = svd.fit_transform(matrix).astype(np.float32)
    return dense


def build_text_inputs_and_features(config: dict[str, object], user_table: pd.DataFrame) -> torch.Tensor:
    processed_dir = Path(config["paths"]["processed_dir"])
    preprocess_cfg = config["preprocess"]
    text_cfg = config["text"]

    texts = [
        _compose_text(
            row.description_text,
            row.tweet_text,
            empty_token=preprocess_cfg["empty_token"],
            separator=preprocess_cfg["tweet_separator"],
        )
        for row in user_table.itertuples(index=False)
    ]
    text_payload = [{"user_id": user_id, "text": text} for user_id, text in zip(user_table["user_id"], texts)]
    with (processed_dir / "text_inputs.pkl").open("wb") as handle:
        pickle.dump(text_payload, handle)

    encoder_name = text_cfg["encoder_type"]
    fallback_name = text_cfg["fallback_encoder_type"]
    active_encoder = encoder_name
    try:
        if encoder_name == "sentence_transformer":
            text_features = _encode_with_sentence_transformer(texts, text_cfg)
        elif encoder_name == "tfidf_svd":
            text_features = _encode_with_tfidf_svd(texts, text_cfg)
        else:
            raise ValueError(f"Unsupported text encoder: {encoder_name}")
    except Exception:
        if fallback_name == encoder_name:
            raise
        active_encoder = fallback_name
        if fallback_name == "tfidf_svd":
            text_features = _encode_with_tfidf_svd(texts, text_cfg)
        else:
            raise ValueError(f"Unsupported fallback text encoder: {fallback_name}")

    text_tensor = torch.tensor(text_features, dtype=torch.float32)
    torch.save(text_tensor, processed_dir / "text_features.pt")
    dump_json(
        {
            "encoder": active_encoder,
            "requested_encoder": encoder_name,
            "feature_dim": int(text_tensor.shape[1]),
            "text_count": len(texts),
        },
        processed_dir / "text_feature_meta.json",
    )
    return text_tensor
