from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dafbot.config import ensure_directories, load_config
from dafbot.pipeline import run_preprocess_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Dynamic Graph preprocessing artifacts for TwiBot-20.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    run_preprocess_pipeline(config)
    print("TwiBot-20 dynamic graph preprocessing completed.")


if __name__ == "__main__":
    main()
