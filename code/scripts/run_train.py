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
from dafbot.train.trainer import EXPERIMENT_VARIANTS, train_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Dynamic-Aware Adaptive Fusion experiment.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--experiment",
        type=str,
        default="dynamic_graph_adaptive",
        choices=sorted(EXPERIMENT_VARIANTS),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    dataset = load_processed_dataset(config)
    artifacts = train_experiment(config, dataset, args.experiment)
    print(f"Checkpoint saved to {artifacts.checkpoint_path}")


if __name__ == "__main__":
    main()
