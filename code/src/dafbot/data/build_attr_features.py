from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dafbot.utils import dump_json, safe_div


NUMERIC_FEATURE_COLUMNS = [
    "followers_count",
    "friends_count",
    "listed_count",
    "favourites_count",
    "statuses_count",
    "followers_friends_ratio",
    "friends_followers_ratio",
    "favourites_statuses_ratio",
    "account_age_days",
    "posting_intensity",
    "username_length",
    "username_digit_ratio",
    "description_length",
    "tweet_num",
]

BOOLEAN_FEATURE_COLUMNS = [
    "verified",
    "default_profile",
    "default_profile_image",
    "has_extended_profile",
    "protected",
    "geo_enabled",
    "url_is_empty",
    "description_is_empty",
]


def _build_feature_frame(user_table: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=user_table.index)
    frame["followers_count"] = pd.to_numeric(user_table["followers_count"], errors="coerce")
    frame["friends_count"] = pd.to_numeric(user_table["friends_count"], errors="coerce")
    frame["listed_count"] = pd.to_numeric(user_table["listed_count"], errors="coerce")
    frame["favourites_count"] = pd.to_numeric(user_table["favourites_count"], errors="coerce")
    frame["statuses_count"] = pd.to_numeric(user_table["statuses_count"], errors="coerce")
    frame["followers_friends_ratio"] = [safe_div(a, b + 1.0) for a, b in zip(frame["followers_count"], frame["friends_count"])]
    frame["friends_followers_ratio"] = [safe_div(a, b + 1.0) for a, b in zip(frame["friends_count"], frame["followers_count"])]
    frame["favourites_statuses_ratio"] = [safe_div(a, b + 1.0) for a, b in zip(frame["favourites_count"], frame["statuses_count"])]
    frame["account_age_days"] = pd.to_numeric(user_table["account_age_days"], errors="coerce")
    frame["posting_intensity"] = [safe_div(a, max(float(b), 1.0)) for a, b in zip(frame["statuses_count"], frame["account_age_days"])]
    frame["username_length"] = pd.to_numeric(user_table["username_length"], errors="coerce")
    frame["username_digit_ratio"] = pd.to_numeric(user_table["username_digit_ratio"], errors="coerce")
    frame["description_length"] = pd.to_numeric(user_table["description_length"], errors="coerce")
    frame["tweet_num"] = pd.to_numeric(user_table["post_count"], errors="coerce")

    for column in BOOLEAN_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(user_table[column], errors="coerce")
    return frame


def build_attr_features(config: dict[str, object], user_table: pd.DataFrame) -> torch.Tensor:
    processed_dir = Path(config["paths"]["processed_dir"])
    feature_frame = _build_feature_frame(user_table).fillna(0.0)

    numeric_frame = feature_frame[NUMERIC_FEATURE_COLUMNS].copy()
    numeric_frame = numeric_frame.clip(lower=0.0)
    medians = numeric_frame.median()
    numeric_frame = numeric_frame.fillna(medians)
    log_numeric = np.log1p(numeric_frame.to_numpy(dtype=np.float32))
    means = log_numeric.mean(axis=0)
    stds = log_numeric.std(axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    scaled_numeric = (log_numeric - means) / stds

    bool_matrix = feature_frame[BOOLEAN_FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
    attr_matrix = np.concatenate([scaled_numeric.astype(np.float32), bool_matrix], axis=1)
    attr_tensor = torch.tensor(attr_matrix, dtype=torch.float32)
    torch.save(attr_tensor, processed_dir / "attr_features.pt")
    dump_json(
        {
            "numeric_feature_columns": NUMERIC_FEATURE_COLUMNS,
            "boolean_feature_columns": BOOLEAN_FEATURE_COLUMNS,
            "numeric_log1p_mean": means.tolist(),
            "numeric_log1p_std": stds.tolist(),
            "numeric_median_fill": medians.astype(float).tolist(),
            "feature_dim": int(attr_tensor.shape[1]),
        },
        processed_dir / "attr_feature_meta.json",
    )
    return attr_tensor
