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

def read_csv_fast(
    path: str | Path, 
    use_arrow: bool = True, 
    n_rows: Optional[int] = None,
    target_col: Optional[str] = None,
    samples_per_class: Optional[Dict[int, int]] = None,
    minority_classes: Optional[list] = None,  # Legacy parameter for backward compatibility
    random_state: int = 42
) -> pd.DataFrame:
    """
    Read CSV with optional chunked reading and balanced class sampling.
    
    If samples_per_class is specified:
    - Reads CSV in chunks and collects up to the specified count for each class
    - Stops reading once all class targets are met (early stopping)
    - Returns balanced and shuffled dataframe
    
    Args:
        path: Path to CSV file
        use_arrow: Whether to use arrow (currently not used, kept for compatibility)
        n_rows: Maximum number of rows to return (ignored if samples_per_class is set)
        target_col: Name of target column for filtering
        samples_per_class: Dict mapping class value -> target count (e.g., {1: 30000, 2: 30000, 3: 30000, 4: 30000})
        minority_classes: Legacy parameter - list of class values to collect all samples of (for backward compatibility)
        random_state: Random seed for sampling
        
    Returns:
        DataFrame with balanced class distribution
    """
    # Legacy support: if minority_classes is specified but not samples_per_class, use old behavior
    if minority_classes is not None and samples_per_class is None:
        # Original minority_classes behavior (collect all samples)
        if target_col is None:
            return pd.read_csv(path, nrows=n_rows, low_memory=False)
        
        chunk_size = 100000
        minority_rows = []
        majority_rows = []
        total_read = 0
        
        print(f"[read_csv] Reading CSV in chunks, collecting all samples of classes {minority_classes}...")
        
        for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
            if target_col not in chunk.columns:
                raise ValueError(f"Target column '{target_col}' not found in CSV")
            
            minority_chunk = chunk[chunk[target_col].isin(minority_classes)]
            majority_chunk = chunk[~chunk[target_col].isin(minority_classes)]
            
            minority_rows.append(minority_chunk)
            majority_rows.append(majority_chunk)
            
            total_read += len(chunk)
            print(f"[read_csv] Processed {total_read:,} rows, found {sum(len(r) for r in minority_rows):,} minority samples", end='\r')
            
            if n_rows:
                n_minority = sum(len(r) for r in minority_rows)
                n_majority = sum(len(r) for r in majority_rows)
                if n_minority + min(n_majority, n_rows - n_minority) >= n_rows:
                    break
        
        print()
        
        if minority_rows:
            minority_df = pd.concat(minority_rows, ignore_index=True)
            n_minority = len(minority_df)
            print(f"[read_csv] Collected {n_minority:,} minority class samples (classes {minority_classes})")
        else:
            minority_df = pd.DataFrame()
            n_minority = 0
        
        if majority_rows:
            majority_df = pd.concat(majority_rows, ignore_index=True)
            if n_rows and n_minority < n_rows:
                n_majority_needed = n_rows - n_minority
                if n_majority_needed > 0:
                    majority_df = majority_df.sample(
                        n=min(n_majority_needed, len(majority_df)), 
                        random_state=random_state
                    )
                    print(f"[read_csv] Sampled {len(majority_df):,} majority class samples")
            else:
                print(f"[read_csv] Using all {len(majority_df):,} majority class samples")
        else:
            majority_df = pd.DataFrame()
        
        if len(minority_df) > 0 or len(majority_df) > 0:
            result = pd.concat([minority_df, majority_df], ignore_index=True)
            result = result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            print(f"[read_csv] Final dataset: {len(result):,} rows")
            if target_col:
                print(f"[read_csv] Class distribution: {result[target_col].value_counts().sort_index().to_dict()}")
            return result
        else:
            raise ValueError("No data collected from CSV file")
    
    # New behavior: samples_per_class
    if samples_per_class is None or target_col is None:
        # Original behavior: simple read
        return pd.read_csv(path, nrows=n_rows, low_memory=False)
    
    # Chunked reading for large files
    chunk_size = 100000  # Read 100k rows at a time
    class_rows = {cls: [] for cls in samples_per_class.keys()}
    total_read = 0
    
    print(f"[read_csv] Reading CSV in chunks, collecting {samples_per_class} samples per class...")
    
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        if target_col not in chunk.columns:
            raise ValueError(f"Target column '{target_col}' not found in CSV")
        
        # Separate by class and collect up to target
        for cls, target_count in samples_per_class.items():
            class_chunk = chunk[chunk[target_col] == cls]
            current_count = sum(len(r) for r in class_rows[cls])
            needed = target_count - current_count
            
            if needed > 0:
                # Take only what we need
                if len(class_chunk) <= needed:
                    class_rows[cls].append(class_chunk)
                else:
                    # Sample randomly from this chunk
                    class_rows[cls].append(class_chunk.sample(n=needed, random_state=random_state))
        
        total_read += len(chunk)
        
        # Check if we've reached all targets
        all_targets_met = all(
            sum(len(r) for r in class_rows[cls]) >= target_count
            for cls, target_count in samples_per_class.items()
        )
        
        current_counts = {cls: sum(len(r) for r in rows) for cls, rows in class_rows.items()}
        print(f"[read_csv] Processed {total_read:,} rows | Collected: {current_counts}", end='\r')
        
        if all_targets_met:
            print()  # New line
            print(f"[read_csv] All class targets met! Stopping early at {total_read:,} rows")
            break
    
    print()  # New line after progress
    
    # Combine all classes
    all_dfs = []
    for cls, rows in class_rows.items():
        if rows:
            cls_df = pd.concat(rows, ignore_index=True)
            # Ensure we don't exceed target (shouldn't happen, but safety check)
            if len(cls_df) > samples_per_class[cls]:
                cls_df = cls_df.sample(n=samples_per_class[cls], random_state=random_state)
            all_dfs.append(cls_df)
            print(f"[read_csv] Collected {len(cls_df):,} samples of class {cls} (target: {samples_per_class[cls]:,})")
    
    if not all_dfs:
        raise ValueError("No data collected from CSV file")
    
    # Combine and shuffle
    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    print(f"[read_csv] Final dataset: {len(result):,} rows")
    if target_col:
        print(f"[read_csv] Class distribution: {result[target_col].value_counts().sort_index().to_dict()}")
    
    return result

def to_parquet(df: pd.DataFrame, path: str | Path) -> None:
    df.to_parquet(path, index=False)

def to_csv_series(y: Iterable, path: str | Path) -> None:
    pd.Series(y).to_csv(path, index=False, header=False)