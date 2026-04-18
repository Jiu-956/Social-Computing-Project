from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dafbot.config import ensure_directories, load_config
from dafbot.data.dataset import load_processed_dataset
from dafbot.train.trainer import train_experiment


ABLATION_EXPERIMENTS = [
    "dynamic_graph_adaptive",
    "ablation_no_temporal",
    "ablation_no_adaptive_fusion",
    "ablation_no_quality",
    "dynamic_graph_concat",
    "dynamic_graph_only",
    "attr_only",
    "text_only",
    "static_graph_only",
    "attr_text_concat",
    "attr_text_static_graph_concat",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline and ablation experiments.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    dataset = load_processed_dataset(config)

    frames = []
    for experiment in ABLATION_EXPERIMENTS:
        artifacts = train_experiment(config, dataset, experiment)
        frames.append(artifacts.metrics_frame)

    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(Path(config["paths"]["table_dir"]) / "ablation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
