from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator

class PreprocessConfig(BaseModel):
    raw_csv_path: str
    output_dir: str
    reports_dir: str

    target_column: str
    id_columns: List[str] = Field(default_factory=list)
    datetime_columns: List[str] = Field(default_factory=list)
    drop_columns: List[str] = Field(default_factory=list)
    selected_features: List[str] = Field(default_factory=list)
    allowlist_only: bool = False

    make_duration: bool = True
    time_features: Dict[str, bool] = Field(default_factory=lambda: {"hour": True, "dow": True, "month": True})
    cyclical_encode: bool = True

    max_missing_col_fraction: float = 0.6
    drop_rows_with_missing_target: bool = True

    sample_n_rows: Optional[int] = 1_000_000
    sample_fraction: Optional[float] = None

    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
    stratify: bool = True

    scale_numeric: bool = True
    ohe_min_freq: int = 0

    n_jobs: int = -1

    # Data sampling configuration
    data_sampling: Optional[Dict] = Field(default_factory=lambda: {
        "minority_oversample": False
    })

    # Class merging configuration
    merge_classes: Optional[Dict] = Field(default_factory=lambda: {
        "enabled": False,
        "mapping": {1: 2, 4: 3}
    })

    @validator("max_missing_col_fraction")
    def _range(cls, v):
        if not (0.0 <= v <= 1.0): raise ValueError("max_missing_col_fraction must be in [0,1]")
        return v

    @validator("test_size", "val_size")
    def _range_split(cls, v):
        if not (0.0 < v < 1.0): raise ValueError("split fractions must be in (0,1)")
        return v

    def output_paths(self) -> dict:
        od = Path(self.output_dir)
        return {
            "X_train": od / "X_train.parquet",
            "X_val": od / "X_val.parquet",
            "X_test": od / "X_test.parquet",
            "y_train": od / "y_train.csv",
            "y_val": od / "y_val.csv",
            "y_test": od / "y_test.csv",
            "pipeline": od / "pipeline.joblib",
            "features": od / "features.json",
            "meta": od / "meta.json",
        }
