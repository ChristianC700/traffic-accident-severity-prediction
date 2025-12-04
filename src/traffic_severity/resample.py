"""Resampling utilities for handling class imbalance."""
from __future__ import annotations
from typing import Optional, Tuple, Union
import time
import numpy as np
from scipy import sparse
from scipy.sparse import issparse

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


def apply_resampling(
    X_train,
    y_train: np.ndarray,
    method: Optional[str] = None,
    k_neighbors: int = 5,
    random_state: int = 42,
    n_jobs: int = 1,
    sampling_strategy: Union[str, dict, float] = "auto",
) -> Tuple[np.ndarray | sparse.spmatrix, np.ndarray]:
    """
    Apply resampling to training data to handle class imbalance.
    
    Args:
        X_train: Training features (sparse or dense)
        y_train: Training labels
        method: Resampling method - "smote", "adasyn", "borderline_smote", or None
        k_neighbors: Number of neighbors for SMOTE variants
        random_state: Random seed
        n_jobs: Number of parallel jobs (-1 for all cores)
        sampling_strategy: Oversampling strategy. Can be:
            - "auto": Automatically balance to majority class
            - "minority": Only oversample minority classes
            - dict: Target counts per class, e.g. {1: 50000, 4: 50000} to oversample classes 1 and 4 to 50k each
            - float: Ratio to oversample (e.g., 0.5 = 50% of majority class size)
        
    Returns:
        Resampled (X_train_resampled, y_train_resampled)
    """
    if method is None or method.lower() == "none":
        return X_train, y_train
    
    if not IMBLEARN_AVAILABLE:
        raise ImportError("imbalanced-learn is required for resampling. Install with: pip install imbalanced-learn")
    
    # Convert sparse to dense if needed (SMOTE requires dense arrays)
    is_sparse_input = issparse(X_train)
    if is_sparse_input:
        print(f"[resample] Converting sparse matrix ({X_train.shape}) to dense...")
        t0_conv = time.time()
        X_train_dense = X_train.toarray()
        conv_time = time.time() - t0_conv
        print(f"[resample] Dense matrix shape: {X_train_dense.shape}, memory: ~{X_train_dense.nbytes / 1e9:.2f} GB, conversion time: {conv_time:.1f}s")
    else:
        X_train_dense = X_train
    
    # Show class distribution before resampling
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"[resample] Class distribution before: {dict(zip(unique, counts))}")
    
    # Select resampling method
    # Note: n_jobs is not supported by SMOTE/ADASYN/BorderlineSMOTE in imbalanced-learn
    # The underlying k-NN computation may use threading internally, but it's not configurable
    method_lower = method.lower()
    if method_lower == "smote":
        resampler = SMOTE(
            k_neighbors=k_neighbors, 
            random_state=random_state,
            sampling_strategy=sampling_strategy
        )
    elif method_lower == "adasyn":
        resampler = ADASYN(
            n_neighbors=k_neighbors, 
            random_state=random_state,
            sampling_strategy=sampling_strategy
        )
    elif method_lower == "borderline_smote":
        resampler = BorderlineSMOTE(
            k_neighbors=k_neighbors, 
            random_state=random_state,
            sampling_strategy=sampling_strategy
        )
    else:
        raise ValueError(f"Unknown resampling method: {method}. Choose from: smote, adasyn, borderline_smote")
    
    # Apply resampling
    print(f"[resample] Starting {method.upper()} resampling (k_neighbors={k_neighbors})...")
    t0 = time.time()
    X_resampled, y_resampled = resampler.fit_resample(X_train_dense, y_train)
    elapsed = time.time() - t0
    print(f"[resample] Resampling completed in {elapsed:.1f} seconds")
    
    # Show class distribution after resampling
    unique, counts = np.unique(y_resampled, return_counts=True)
    print(f"[resample] Class distribution after: {dict(zip(unique, counts))}")
    
    # Convert back to sparse if input was sparse (though resampling typically increases size)
    # For now, keep as dense since resampling usually increases data size significantly
    return X_resampled, y_resampled

