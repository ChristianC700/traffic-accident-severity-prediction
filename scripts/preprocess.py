#!/usr/bin/env python
# File: scripts/preprocess.py
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Make "src" importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from traffic_severity.preprocess import run_preprocess  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess accident severity dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    run_preprocess(Path(args.config))


if __name__ == "__main__":
    main()
