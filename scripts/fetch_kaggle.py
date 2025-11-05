#!/usr/bin/env python
"""
Download & unzip the US Accidents dataset from Kaggle into data/raw/.
Why: Single, reproducible entrypoint for teammates/CI.
Requires: kaggle API credentials at ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY envs.
"""
from __future__ import annotations
import sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "sobhanmoosavi/us-accidents"
OUTDIR = Path("data/raw")

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()  # reads ~/.kaggle/kaggle.json or envs

    print(f"[fetch] downloading {DATASET} → {OUTDIR} (unzipping)…")
    api.dataset_download_files(DATASET, path=str(OUTDIR), unzip=True, quiet=False)

    # Display what we pulled so you can copy the CSV name into configs/default.yaml
    print("\n[fetch] contents of data/raw/:")
    for p in sorted(OUTDIR.glob("*")):
        print(" -", p.name)
    print("\n[fetch] Next:")
    print("  1) Open configs/default.yaml")
    print('  2) Set raw_csv_path to the main CSV (e.g., "data/raw/US_Accidents_March23.csv")')
    print("  3) Run: make preprocess")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[fetch][error] {e}", file=sys.stderr)
        sys.exit(1)