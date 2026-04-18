from __future__ import annotations

import json
import pickle
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

TWITTER_TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_twitter_datetime(value: Any) -> datetime | None:
    if value in (None, "", "None"):
        return None
    try:
        return datetime.strptime(str(value).strip(), TWITTER_TIME_FORMAT).astimezone(timezone.utc)
    except ValueError:
        return None


def parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "none":
        return ""
    return " ".join(text.split())


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(data: Any, path: str | Path) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def dump_pickle(data: Any, path: str | Path) -> None:
    path = ensure_parent(path)
    with path.open("wb") as handle:
        pickle.dump(data, handle)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
