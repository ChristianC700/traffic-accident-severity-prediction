from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import numpy as np
import pandas as pd

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))

def read_csv_fast(path: str | Path, use_arrow: bool = True, n_rows: Optional[int] = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=n_rows, low_memory=False)

def to_parquet(df: pd.DataFrame, path: str | Path) -> None:
    df.to_parquet(path, index=False)

def to_csv_series(y: Iterable, path: str | Path) -> None:
    pd.Series(y).to_csv(path, index=False, header=False)