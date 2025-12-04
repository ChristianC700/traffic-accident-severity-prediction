from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse  # add at top

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import PreprocessConfig
from .utils import ensure_dir, read_csv_fast, save_json, to_csv_series

@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

def load_config(path: str | Path) -> PreprocessConfig:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return PreprocessConfig(**cfg_dict)

def basic_eda_reports(df: pd.DataFrame, target_col: str, reports_dir: Path) -> None:
    reports_dir = ensure_dir(reports_dir)
    overview = {
        "timestamp": datetime.utcnow().isoformat(),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "target": target_col,
    }
    save_json(overview, reports_dir / "eda_overview.json")
    df.isna().mean().sort_values(ascending=False).rename("missing_frac").to_csv(reports_dir / "missingness.csv")
    if target_col in df.columns:
        df[target_col].value_counts(dropna=False).rename("count").to_csv(reports_dir / "class_distribution.csv")
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        df[num_cols].describe().to_csv(reports_dir / "numeric_stats.csv")

def _apply_allowlist(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    if cfg.allowlist_only and cfg.selected_features:
        keep = [c for c in cfg.selected_features if c in df.columns]
        return df[keep].copy()
    return df

def _make_duration(df: pd.DataFrame, start_col: str, end_col: str) -> pd.DataFrame:
    if start_col in df.columns and end_col in df.columns:
        st = pd.to_datetime(df[start_col], errors="coerce")
        et = pd.to_datetime(df[end_col], errors="coerce")
        df["duration_min"] = (et - st).dt.total_seconds() / 60.0
    return df

def _time_cyclical(df: pd.DataFrame, start_col: str, use_hour: bool, use_dow: bool, use_month: bool, cyclical: bool) -> pd.DataFrame:
    if start_col not in df.columns: return df
    t = pd.to_datetime(df[start_col], errors="coerce")
    if use_hour:
        h = t.dt.hour
        if cyclical:
            df["hour_sin"] = np.sin(2*np.pi*h/24)
            df["hour_cos"] = np.cos(2*np.pi*h/24)
        else:
            df["hour"] = h
    if use_dow:
        d = t.dt.dayofweek
        if cyclical:
            df["dow_sin"] = np.sin(2*np.pi*d/7)
            df["dow_cos"] = np.cos(2*np.pi*d/7)
        else:
            df["dow"] = d
    if use_month:
        m = t.dt.month
        if cyclical:
            df["month_sin"] = np.sin(2*np.pi*m/12)
            df["month_cos"] = np.cos(2*np.pi*m/12)
        else:
            df["month"] = m
    return df

def select_and_engineer(df: pd.DataFrame, cfg: PreprocessConfig) -> Tuple[pd.DataFrame, List[str], List[str]]:
    # Optional allowlist to keep project-specified features
    df = _apply_allowlist(df, cfg)

    # Drop rows if target missing
    if cfg.drop_rows_with_missing_target and cfg.target_column in df.columns:
        df = df[~df[cfg.target_column].isna()].copy()

    # Parse time, duration, and engineered time features
    start_col = cfg.datetime_columns[0] if cfg.datetime_columns else None
    end_col = cfg.datetime_columns[1] if len(cfg.datetime_columns) > 1 else None
    if cfg.make_duration and start_col and end_col:
        df = _make_duration(df, start_col, end_col)
    if start_col:
        df = _time_cyclical(
            df,
            start_col=start_col,
            use_hour=cfg.time_features.get("hour", True),
            use_dow=cfg.time_features.get("dow", True),
            use_month=cfg.time_features.get("month", True),
            cyclical=cfg.cyclical_encode,
        )
        # Drop original datetime columns to avoid leakage; keep if needed later
        df = df.drop(columns=[c for c in [start_col, end_col] if c in df.columns])

    # Drop high-missing columns + ids + forced
    to_drop_missing = df.columns[df.isna().mean() > cfg.max_missing_col_fraction].tolist()
    to_drop = list({*(cfg.id_columns or []), *to_drop_missing, *(cfg.drop_columns or [])})
    to_drop = [c for c in to_drop if c in df.columns and c != cfg.target_column]
    df = df.drop(columns=to_drop, errors="ignore")

    # Split features / target
    y = df[cfg.target_column].copy()
    X = df.drop(columns=[cfg.target_column])
    
    # Apply class merging if enabled
    if cfg.merge_classes and cfg.merge_classes.get("enabled", False):
        mapping = cfg.merge_classes.get("mapping", {1: 2, 4: 3})
        y = y.replace(mapping)

    # Numeric / Categorical separation
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    return pd.concat([X, y], axis=1), num_cols, cat_cols

def build_transformer(num_cols: List[str], cat_cols: List[str], scale_numeric: bool, min_freq: int) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=None if min_freq == 0 else min_freq)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        n_jobs=None,
    )

def split_data(X: pd.DataFrame, y: pd.Series, cfg: PreprocessConfig) -> DatasetSplits:
    X_tr_tmp, X_te, y_tr_tmp, y_te = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y if cfg.stratify else None
    )
    val_ratio = cfg.val_size / (1.0 - cfg.test_size)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr_tmp, y_tr_tmp, test_size=val_ratio, random_state=cfg.random_state, stratify=y_tr_tmp if cfg.stratify else None
    )
    return DatasetSplits(X_tr, X_va, X_te, y_tr.values, y_va.values, y_te.values)

def fit_transform_and_save(
    splits: DatasetSplits,
    transformer: ColumnTransformer,
    out_paths: dict,
    reports_dir: Path,
    cfg: PreprocessConfig,
) -> None:
    X_train_tf = transformer.fit_transform(splits.X_train)
    X_val_tf = transformer.transform(splits.X_val)
    X_test_tf = transformer.transform(splits.X_test)

    # Feature names
    feature_names: List[str] = []
    for name, trans, cols in transformer.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            ohe = trans.named_steps["ohe"]
            feature_names.extend(ohe.get_feature_names_out(cols).tolist())

    # Save X matrices:
    # - If sparse -> .npz (CSR)
    # - If dense  -> Parquet
    def save_X(x, path_parquet: Path, tag: str):
        if sparse.issparse(x):
            npz_path = path_parquet.with_suffix(".npz")
            sparse.save_npz(npz_path, x.tocsr())
            return {"path": str(npz_path), "format": "npz", "shape": list(x.shape)}
        else:
            df = pd.DataFrame(x, columns=feature_names)
            df.to_parquet(path_parquet, index=False)
            return {"path": str(path_parquet), "format": "parquet", "shape": [int(df.shape[0]), int(df.shape[1])]}

    meta_x = {
        "train": save_X(X_train_tf, out_paths["X_train"], "train"),
        "val":   save_X(X_val_tf, out_paths["X_val"], "val"),
        "test":  save_X(X_test_tf, out_paths["X_test"], "test"),
    }

    # y vectors
    to_csv_series(splits.y_train, out_paths["y_train"])
    to_csv_series(splits.y_val, out_paths["y_val"])
    to_csv_series(splits.y_test, out_paths["y_test"])

    # Persist pipeline and metadata
    joblib.dump(transformer, out_paths["pipeline"])
    save_json({"feature_names": feature_names}, out_paths["features"])
    
    # Include class merging info in metadata
    meta_data = {
        "created_at": datetime.utcnow().isoformat(),
        "rows": {"train": int(len(splits.y_train)), "val": int(splits.X_val.shape[0]), "test": int(splits.X_test.shape[0])},
        "X_artifacts": meta_x,
        "merge_classes": cfg.merge_classes or {},
    }
    save_json(meta_data, out_paths["meta"])


def run_preprocess(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.reports_dir)

    # Check if we should use balanced class sampling
    data_sampling_cfg = cfg.data_sampling or {}
    minority_oversample = data_sampling_cfg.get("minority_oversample", False)
    samples_per_class = data_sampling_cfg.get("samples_per_class", None)
    
    if minority_oversample and samples_per_class:
        # Load with balanced class sampling - collects specified count per class
        # This stops early once all class targets are met
        df = read_csv_fast(
            cfg.raw_csv_path,
            n_rows=None,  # Ignored when samples_per_class is set
            target_col=cfg.target_column,
            samples_per_class=samples_per_class,
            random_state=cfg.random_state
        )
    elif minority_oversample:
        # Legacy behavior: collect all minority classes (for backward compatibility)
        df = read_csv_fast(
            cfg.raw_csv_path,
            n_rows=None,
            target_col=cfg.target_column,
            minority_classes=[1, 4],  # Collect all samples of these classes
            random_state=cfg.random_state
        )
        # If sample_n_rows is set, keep all minority, sample from majority
        if cfg.sample_n_rows and len(df) > cfg.sample_n_rows:
            minority = df[df[cfg.target_column].isin([1, 4])]
            majority = df[~df[cfg.target_column].isin([1, 4])]
            n_needed = cfg.sample_n_rows - len(minority)
            if n_needed > 0:
                majority = majority.sample(n=min(n_needed, len(majority)), random_state=cfg.random_state)
            df = pd.concat([minority, majority], ignore_index=True)
            df = df.sample(frac=1.0, random_state=cfg.random_state).reset_index(drop=True)
            print(f"[preprocess] Final dataset after limiting: {len(df):,} rows")
    else:
        # Original behavior: simple random sampling
        n_rows = cfg.sample_n_rows if cfg.sample_n_rows else None
        df = read_csv_fast(cfg.raw_csv_path, n_rows=n_rows)
    
    if cfg.sample_fraction is not None:
        df = df.sample(frac=cfg.sample_fraction, random_state=cfg.random_state)

    basic_eda_reports(df, cfg.target_column, Path(cfg.reports_dir))

    df2, num_cols, cat_cols = select_and_engineer(df, cfg)
    X, y = df2.drop(columns=[cfg.target_column]), df2[cfg.target_column]

    splits = split_data(X, y, cfg)
    transformer = build_transformer(num_cols, cat_cols, cfg.scale_numeric, cfg.ohe_min_freq)
    fit_transform_and_save(splits, transformer, cfg.output_paths(), Path(cfg.reports_dir), cfg)
