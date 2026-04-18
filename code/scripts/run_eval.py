from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dafbot.config import ensure_directories, load_config
from dafbot.data.dataset import load_processed_dataset
from dafbot.train.trainer import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint and export interpretability reports.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    dataset = load_processed_dataset(config)
    metrics = evaluate_checkpoint(config, dataset, args.checkpoint)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
