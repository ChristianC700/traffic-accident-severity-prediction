#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train classification models on preprocessed artifacts and evaluate on both VAL + TEST.
Saves:
  - reports/metrics/<model>_<split>.json
  - reports/predictions/<model>_test_predictions.csv
  - reports/figures/cm_<model>_<split>.png
  - reports/baselines_summary.csv
"""

from __future__ import annotations
import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, issparse

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import matplotlib.pyplot as plt
import yaml
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


# ---------- IO helpers ----------

def _load_X(path: Path):
    if path.suffix == ".npz":
        return load_npz(path)
    elif path.suffix == ".parquet":
        return pd.read_parquet(path).values  # dense ndarray
    else:
        raise ValueError(f"Unsupported X format: {path.suffix}")


def _load_y(path: Path) -> np.ndarray:
    # Saved as single-column CSV without header
    return np.loadtxt(path, delimiter=",")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _maybe_dense(X, tag: str = "[model]"):
    """Convert sparse to dense for models that require/benefit from dense arrays."""
    if issparse(X):
        print(f"{tag} Converting sparse CSR to dense (may use more RAM)…")
        return X.toarray()
    return X


def _get_torch_device():
    """Pick MPS on Apple Silicon if available, else CPU."""
    if torch.backends.mps.is_available():
        print("[torch] Using MPS (Apple GPU) backend.")
        return torch.device("mps")
    else:
        print("[torch] MPS not available; using CPU.")
        return torch.device("cpu")


def _cleanup_memory():
    """Clear PyTorch caches and force garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ---------- Class imbalance helpers ----------

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is the predicted probability for the true class.
    """
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of true class
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            # Apply alpha weighting
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_weights(
    y: np.ndarray, 
    label_order: np.ndarray | None = None,
    strategy: str = "inverse_freq"
) -> torch.Tensor:
    """
    Compute class weights using different strategies.
    
    Args:
        y: Array of labels
        label_order: Ordered list of unique labels
        strategy: "inverse_freq", "balanced", or "uniform"
        
    Returns:
        Tensor of class weights aligned with label_order
    """
    labels, counts = np.unique(y, return_counts=True)
    counts = counts.astype(np.float32)
    if label_order is None:
        label_order = labels
    else:
        label_order = np.asarray(label_order)
    
    count_map = {int(lbl): float(cnt) for lbl, cnt in zip(labels, counts)}
    weights = []
    
    if strategy == "inverse_freq":
        # Inverse frequency weighting (current default)
        total = counts.sum()
        num_classes = len(label_order)
        for lbl in label_order:
            c = count_map.get(int(lbl), 0.0)
            if c == 0.0:
                weights.append(0.0)
            else:
                weights.append(total / (num_classes * c))
    elif strategy == "balanced":
        # sklearn-style balanced weights
        total = counts.sum()
        num_classes = len(label_order)
        for lbl in label_order:
            c = count_map.get(int(lbl), 0.0)
            if c == 0.0:
                weights.append(0.0)
            else:
                weights.append(total / (num_classes * c))
    elif strategy == "uniform":
        # Uniform weights (no weighting)
        weights = [1.0] * len(label_order)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from: inverse_freq, balanced, uniform")
    
    return torch.tensor(weights, dtype=torch.float32)


# ---------- Metrics / Visualization ----------

def pretty_model_name(key: str) -> str:
    """Map internal model keys to human-readable titles."""
    return {
        "dummy": "Dummy Classifier (Most Frequent)",
        "logreg": "Logistic Regression",
        "linearsvc": "Linear SVM",
        "mlp": "Multi-layer Perceptron (PyTorch)",
        "ffnn": "Feed-Forward Neural Network (2-Layer, PyTorch)",
        "rf": "Random Forest",
    }.get(key, key)


def compute_cost(y_true, y_pred, cost_matrix: np.ndarray | None = None, label_order: np.ndarray | None = None):
    """Compute total cost given a cost matrix."""
    if cost_matrix is None:
        return None
    
    if label_order is None:
        label_order = np.unique(np.concatenate([y_true, y_pred]))
    
    total_cost = 0.0
    for true_cls, pred_cls in zip(y_true, y_pred):
        true_idx = np.where(label_order == true_cls)[0][0]
        pred_idx = np.where(label_order == pred_cls)[0][0]
        total_cost += cost_matrix[true_idx, pred_idx]
    
    return total_cost


def build_metric_payload(y_true, y_pred, label_order: np.ndarray | None = None, y_proba: np.ndarray | None = None, cost_matrix: np.ndarray | None = None):
    """Return metrics dict + confusion matrix for the provided split.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_order: Ordered list of unique labels
        y_proba: Predicted probabilities (optional, for AUC computation)
    """
    if label_order is None:
        label_order = np.unique(y_true)
    label_order = np.asarray(label_order, dtype=int)
    
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    
    precision, recall, f1s, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_order, zero_division=0
    )
    
    # Per-class AUC (one-vs-rest) if probabilities available
    per_class_auc = {}
    if y_proba is not None and y_proba.shape[1] == len(label_order):
        try:
            # Convert labels to binary for each class
            for i, cls in enumerate(label_order):
                y_binary = (y_true == cls).astype(int)
                if len(np.unique(y_binary)) > 1:  # Need both classes present
                    per_class_auc[int(cls)] = float(roc_auc_score(y_binary, y_proba[:, i]))
        except Exception:
            pass  # Skip AUC if computation fails
    
    # Top-2 accuracy
    top2_acc = None
    if y_proba is not None:
        try:
            top2_acc = float(top_k_accuracy_score(y_true, y_proba, k=2, labels=label_order))
        except Exception:
            pass
    
    metrics = {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "cohens_kappa": kappa,
        "matthews_corrcoef": mcc,
        "top2_accuracy": top2_acc,
        "per_class": [
            {
                "class": int(cls),
                "precision": float(pi),
                "recall": float(ri),
                "f1": float(fi),
                "support": int(si),
                "auc": per_class_auc.get(int(cls), None),
            }
            for cls, pi, ri, fi, si in zip(label_order, precision, recall, f1s, support)
        ],
        "per_class_auc": per_class_auc,
        "classification_report": classification_report(
            y_true, y_pred, labels=label_order, digits=4
        ),
    }
    
    # Cost-sensitive evaluation
    if cost_matrix is not None:
        total_cost = compute_cost(y_true, y_pred, cost_matrix, label_order)
        metrics["total_cost"] = total_cost
        metrics["average_cost"] = total_cost / len(y_true) if len(y_true) > 0 else 0.0
    
    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    return metrics, cm


def save_confusion_png(cm: np.ndarray, labels: np.ndarray, out_png: Path, title: str):
    """Save normalized confusion matrix with values in cells and proper title."""
    # Row-normalize so each row sums to 1
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_plot = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_plot, display_labels=labels)
    disp.plot(ax=ax, include_values=True, colorbar=False, values_format=".2f", cmap="Blues")
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ---------- Training / Evaluation ----------

def train_and_eval(
    model_name: str,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    random_state: int,
    linear_class_weight: str | None,
    class_weight_tensor: torch.Tensor | None,
    label_order: np.ndarray | None,
    focal_loss_gamma: float | None = None,
    logreg_max_iter: int = 1000,
    logreg_tol: float = 1e-4,
    tune_linearsvc: bool = False,
    linearsvc_cv_folds: int = 3,
    tune_rf: bool = False,
    rf_n_iter: int = 20,
    rf_cv_folds: int = 3,
    mlp_config: Dict | None = None,
    cost_matrix: np.ndarray | None = None,
) -> Tuple[Dict, np.ndarray, Dict, np.ndarray, Dict[str, np.ndarray]]:
    """Train model and return metrics for validation + test splits."""

    if label_order is None:
        label_order = np.unique(y_train)
    label_order = np.asarray(label_order, dtype=int)
    label_to_idx = {int(lbl): idx for idx, lbl in enumerate(label_order)}

    params: Dict = {}
    y_pred_val: np.ndarray
    y_pred_test: np.ndarray

    # ---- Dummy baseline ----
    if model_name == "dummy":
        clf = DummyClassifier(strategy="most_frequent")
        params = {"strategy": "most_frequent"}

        dummy_train = np.zeros((len(y_train), 1))
        dummy_val = np.zeros((len(y_val), 1))
        dummy_test = np.zeros((len(y_test), 1))

        t0 = time.time()
        clf.fit(dummy_train, y_train)
        train_time = time.time() - t0

        y_pred_val = clf.predict(dummy_val)
        y_pred_test = clf.predict(dummy_test)

    # ---- Logistic Regression (PyTorch GPU-accelerated) ----
    elif model_name == "logreg":
        X_train_dense = _maybe_dense(X_train, tag="[logreg]")
        X_val_dense = _maybe_dense(X_val, tag="[logreg]")
        X_test_dense = _maybe_dense(X_test, tag="[logreg]")

        device = _get_torch_device()
        
        # Set random seed for reproducibility
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS doesn't have manual_seed, but we set CPU seed which should be sufficient
            pass

        # Map labels to contiguous indices
        y_train_idx = np.vectorize(label_to_idx.get)(y_train)
        y_val_idx = np.vectorize(label_to_idx.get)(y_val)
        y_test_idx = np.vectorize(label_to_idx.get)(y_test)

        # Convert to torch tensors
        X_train_t = torch.from_numpy(X_train_dense.astype(np.float32))
        y_train_t = torch.from_numpy(y_train_idx.astype(np.int64))
        X_val_t = torch.from_numpy(X_val_dense.astype(np.float32))
        y_val_t = torch.from_numpy(y_val_idx.astype(np.int64))
        X_test_t = torch.from_numpy(X_test_dense.astype(np.float32))
        y_test_t = torch.from_numpy(y_test_idx.astype(np.int64))

        num_features = X_train_t.shape[1]
        num_classes = len(label_order)

        # Simple linear layer (logistic regression = linear layer + softmax)
        class LogisticRegressionTorch(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.linear = nn.Linear(in_dim, out_dim)
            
            def forward(self, x):
                return self.linear(x)

        model = LogisticRegressionTorch(num_features, num_classes).to(device)

        # Use class weights if provided (from class_weight_tensor)
        weights = class_weight_tensor.to(device) if class_weight_tensor is not None else None
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Adam optimizer with L2 regularization (weight_decay=0.1 equivalent to C=0.1 in sklearn)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.1)

        # Batch size for GPU efficiency
        batch_size = 4096
        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Move validation and test data to device
        X_val_t = X_val_t.to(device)
        y_val_t = y_val_t.to(device)
        X_test_t = X_test_t.to(device)
        y_test_t = y_test_t.to(device)

        print(f"[logreg] Data prepared: train={X_train_t.shape}, val={X_val_t.shape}, test={X_test_t.shape}")
        print(f"[logreg] Starting training for up to {logreg_max_iter} epochs...")

        # Training with early stopping based on tolerance
        max_epochs = logreg_max_iter
        best_val_loss = float('inf')
        patience_counter = 0
        patience_threshold = 3  # Stop if no improvement for 3 epochs
        best_model_state = None

        t0 = time.time()
        model.train()
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            avg_loss = epoch_loss / len(train_ds)
            
            # Check validation loss for early stopping
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()
            model.train()
            
            # Early stopping based on tolerance
            if val_loss < best_val_loss - logreg_tol:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience_threshold:
                    print(f"[logreg] Early stopping at epoch {epoch+1} (loss change < {logreg_tol})")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f"[logreg] Epoch {epoch+1}/{max_epochs} - train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")
            elif epoch == 0:
                print(f"[logreg] Epoch 1/{max_epochs} - train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")

        train_time = time.time() - t0

        # Evaluation
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t)
            logits_test = model(X_test_t)
            y_proba_val = torch.softmax(logits_val, dim=1).cpu().numpy()
            y_proba_test = torch.softmax(logits_test, dim=1).cpu().numpy()
            y_pred_val_idx = torch.argmax(logits_val, dim=1).cpu().numpy()
            y_pred_test_idx = torch.argmax(logits_test, dim=1).cpu().numpy()
            y_pred_val = label_order[y_pred_val_idx]
            y_pred_test = label_order[y_pred_test_idx]

        params = {
            "type": "torch_logreg",
            "device": str(device),
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "lr": 0.01,
            "weight_decay": 0.1,
            "tol": logreg_tol,
            "class_weights": weights.cpu().numpy().tolist() if weights is not None else None,
            "random_state": random_state,
        }
        
        # Cleanup: delete dense arrays and tensors
        del X_train_dense, X_val_dense, X_test_dense
        del X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t
        del model, optimizer, train_loader, train_ds
        _cleanup_memory()

    # ---- Linear SVM ----
    elif model_name == "linearsvc":
        if tune_linearsvc:
            # Hyperparameter grid for LinearSVC
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'loss': ['hinge', 'squared_hinge'],
                'max_iter': [1000, 5000, 10000],
            }
            base_clf = LinearSVC(
                random_state=random_state,
                class_weight=linear_class_weight,
                dual=False,  # False for n_samples > n_features
            )
            print(f"[linearsvc] Running GridSearchCV with {linearsvc_cv_folds}-fold CV...")
            t0 = time.time()
            clf = GridSearchCV(
                base_clf,
                param_grid,
                cv=linearsvc_cv_folds,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=1,
            )
            clf.fit(X_train, y_train)
            train_time = time.time() - t0
            print(f"[linearsvc] Best params: {clf.best_params_}")
            params = {
                "best_params": clf.best_params_,
                "best_score": float(clf.best_score_),
                "class_weight": linear_class_weight,
                "random_state": random_state,
            }
            clf = clf.best_estimator_
        else:
            clf = LinearSVC(
                random_state=random_state,
                class_weight=linear_class_weight,
                max_iter=1000,  # Prevent infinite runs
                dual=False,  # Efficient for sparse data when n_samples > n_features
            )
            params = {
                "class_weight": linear_class_weight,
                "max_iter": 1000,
                "dual": False,
                "random_state": random_state,
            }
            t0 = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - t0

        y_pred_val = clf.predict(X_val)
        y_pred_test = clf.predict(X_test)
        # Get probabilities for AUC if available
        # LinearSVC doesn't support predict_proba, use decision_function for scores
        if hasattr(clf, 'predict_proba'):
            y_proba_val = clf.predict_proba(X_val)
            y_proba_test = clf.predict_proba(X_test)
        else:
            # LinearSVC only has decision_function, not predict_proba
            y_proba_val = None
            y_proba_test = None

    # ---- Random Forest ----
    elif model_name == "rf":
        if tune_rf:
            # Hyperparameter distribution for RandomizedSearchCV
            param_dist = {
                'n_estimators': [200, 300, 500, 1000],
                'max_depth': [10, 15, 20, 25, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5],
                'max_features': ['sqrt', 'log2', 0.3, 0.5],
                'class_weight': ['balanced', 'balanced_subsample'],
            }
            base_clf = RandomForestClassifier(
                n_jobs=-1,
                random_state=random_state,
            )
            print(f"[rf] Running RandomizedSearchCV with {rf_n_iter} iterations, {rf_cv_folds}-fold CV...")
            t0 = time.time()
            clf = RandomizedSearchCV(
                base_clf,
                param_dist,
                n_iter=rf_n_iter,
                cv=rf_cv_folds,
                scoring='f1_macro',
                n_jobs=-1,
                random_state=random_state,
                verbose=1,
            )
            clf.fit(X_train, y_train)
            train_time = time.time() - t0
            print(f"[rf] Best params: {clf.best_params_}")
            params = {
                "best_params": clf.best_params_,
                "best_score": float(clf.best_score_),
                "n_jobs": -1,
                "random_state": random_state,
            }
            clf = clf.best_estimator_
        else:
            clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            )
            params = {
                "n_estimators": 300,
                "max_depth": 20,
                "min_samples_leaf": 5,
                "class_weight": "balanced",
                "n_jobs": -1,
                "random_state": random_state,
            }
            t0 = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - t0

        y_pred_val = clf.predict(X_val)
        y_pred_test = clf.predict(X_test)
        # Get probabilities for AUC if available
        try:
            y_proba_val = clf.predict_proba(X_val)
            y_proba_test = clf.predict_proba(X_test)
        except:
            y_proba_val = None
            y_proba_test = None

    # ---- Neural Network (single-hidden-layer MLP, PyTorch) ----
    elif model_name == "mlp":
        X_train_dense = _maybe_dense(X_train, tag="[mlp]")
        X_val_dense = _maybe_dense(X_val, tag="[mlp]")
        X_test_dense = _maybe_dense(X_test, tag="[mlp]")

        device = _get_torch_device()

        # Map labels to contiguous indices
        y_train_idx = np.vectorize(label_to_idx.get)(y_train)
        y_val_idx = np.vectorize(label_to_idx.get)(y_val)
        y_test_idx = np.vectorize(label_to_idx.get)(y_test)

        # Convert to torch tensors
        X_train_t = torch.from_numpy(X_train_dense.astype(np.float32))
        y_train_t = torch.from_numpy(y_train_idx.astype(np.int64))
        X_val_t = torch.from_numpy(X_val_dense.astype(np.float32))
        y_val_t = torch.from_numpy(y_val_idx.astype(np.int64))
        X_test_t = torch.from_numpy(X_test_dense.astype(np.float32))
        y_test_t = torch.from_numpy(y_test_idx.astype(np.int64))

        num_features = X_train_t.shape[1]
        num_classes = len(label_order)

        # Read MLP config or use defaults
        mlp_cfg = mlp_config or {}
        hidden_dims = mlp_cfg.get("hidden_dims", [128])
        dropout = mlp_cfg.get("dropout", 0.2)
        activation = mlp_cfg.get("activation", "relu")
        batch_norm = mlp_cfg.get("batch_norm", False)
        batch_size = mlp_cfg.get("batch_size", 1024)
        epochs = mlp_cfg.get("epochs", 10)
        lr = mlp_cfg.get("lr", 1e-3)
        weight_decay = mlp_cfg.get("weight_decay", 0.0)
        lr_schedule = mlp_cfg.get("lr_schedule", "none")
        early_stopping = mlp_cfg.get("early_stopping", {})
        es_enabled = early_stopping.get("enabled", False)
        es_patience = early_stopping.get("patience", 5)
        es_min_delta = early_stopping.get("min_delta", 0.001)

        # Build activation function
        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "swish":
            act_fn = nn.SiLU()  # SiLU is Swish
        else:
            act_fn = nn.ReLU()

        class MLP(nn.Module):
            def __init__(self, in_dim, hidden_dims, out_dim, dropout, activation, batch_norm):
                super().__init__()
                layers = []
                prev_dim = in_dim
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    if batch_norm:
                        layers.append(nn.BatchNorm1d(hidden_dim))
                    layers.append(activation)
                    layers.append(nn.Dropout(dropout))
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, out_dim))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        model = MLP(num_features, hidden_dims, num_classes, dropout, act_fn, batch_norm).to(device)

        # Dataloaders - using standard shuffling (balanced_sampling disabled for balanced datasets)
        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        X_val_t = X_val_t.to(device)
        y_val_t = y_val_t.to(device)
        X_test_t = X_test_t.to(device)
        y_test_t = y_test_t.to(device)

        weights = class_weight_tensor.to(device) if class_weight_tensor is not None else None
        # Support focal loss if gamma is provided
        if focal_loss_gamma is not None and focal_loss_gamma > 0:
            criterion = FocalLoss(alpha=weights, gamma=focal_loss_gamma)
            loss_type = "focal"
        else:
            criterion = nn.CrossEntropyLoss(weight=weights)
            loss_type = "cross_entropy"
        
        # Optimizer
        if weight_decay > 0:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Learning rate scheduler
        if lr_schedule == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        elif lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            scheduler = None

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        t0 = time.time()
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            avg_loss = epoch_loss / len(train_ds)
            
            # Validation loss for early stopping
            if es_enabled:
                model.eval()
                with torch.no_grad():
                    val_logits = model(X_val_t)
                    val_loss = criterion(val_logits, y_val_t).item()
                model.train()
                
                if val_loss < best_val_loss - es_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= es_patience:
                        print(f"[mlp] Early stopping at epoch {epoch+1}")
                        model.load_state_dict(best_model_state)
                        break
            
            # Learning rate scheduling
            if scheduler:
                if lr_schedule == "plateau":
                    scheduler.step(avg_loss)
                else:
                    scheduler.step()
            
            print(f"[mlp] Epoch {epoch+1}/{epochs} - loss={avg_loss:.4f}")

        train_time = time.time() - t0

        # Evaluation
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t)
            logits_test = model(X_test_t)
            y_proba_val = torch.softmax(logits_val, dim=1).cpu().numpy()
            y_proba_test = torch.softmax(logits_test, dim=1).cpu().numpy()
            y_pred_val_idx = torch.argmax(logits_val, dim=1).cpu().numpy()
            y_pred_test_idx = torch.argmax(logits_test, dim=1).cpu().numpy()
            y_pred_val = label_order[y_pred_val_idx]
            y_pred_test = label_order[y_pred_test_idx]

        params = {
            "type": "torch_mlp",
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "activation": activation,
            "batch_norm": batch_norm,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "lr_schedule": lr_schedule,
            "early_stopping": early_stopping,
            "device": str(device),
            "loss": loss_type,
            "focal_gamma": focal_loss_gamma if focal_loss_gamma else None,
            "class_weights": weights.cpu().numpy().tolist() if weights is not None else None,
        }
        
        # Cleanup: delete dense arrays and tensors
        del X_train_dense, X_val_dense, X_test_dense
        del X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t
        del model, optimizer, train_loader, train_ds
        if scheduler is not None:
            del scheduler
        _cleanup_memory()

    # ---- Two-Layer Feed-Forward Neural Network (PyTorch) ----
    elif model_name == "ffnn":
        X_train_dense = _maybe_dense(X_train, tag="[ffnn]")
        X_val_dense = _maybe_dense(X_val, tag="[ffnn]")
        X_test_dense = _maybe_dense(X_test, tag="[ffnn]")

        device = _get_torch_device()

        # Map labels to contiguous indices
        y_train_idx = np.vectorize(label_to_idx.get)(y_train)
        y_val_idx = np.vectorize(label_to_idx.get)(y_val)
        y_test_idx = np.vectorize(label_to_idx.get)(y_test)

        # Convert to torch tensors
        X_train_t = torch.from_numpy(X_train_dense.astype(np.float32))
        y_train_t = torch.from_numpy(y_train_idx.astype(np.int64))
        X_val_t = torch.from_numpy(X_val_dense.astype(np.float32))
        y_val_t = torch.from_numpy(y_val_idx.astype(np.int64))
        X_test_t = torch.from_numpy(X_test_dense.astype(np.float32))
        y_test_t = torch.from_numpy(y_test_idx.astype(np.int64))

        num_features = X_train_t.shape[1]
        num_classes = len(label_order)

        class FeedForwardNN(nn.Module):
            def __init__(self, in_dim, hidden1, hidden2, out_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden1),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden1, hidden2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden2, out_dim),
                )

            def forward(self, x):
                return self.net(x)

        model = FeedForwardNN(
            in_dim=num_features,
            hidden1=256,
            hidden2=128,
            out_dim=num_classes,
        ).to(device)

        # Dataloaders with balanced sampling if enabled
        batch_size = 1024
        train_ds = TensorDataset(X_train_t, y_train_t)
        # For FFNN, balanced sampling can be enabled via config if needed
        # For now, use standard shuffling
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        X_val_t = X_val_t.to(device)
        y_val_t = y_val_t.to(device)
        X_test_t = X_test_t.to(device)
        y_test_t = y_test_t.to(device)

        weights = class_weight_tensor.to(device) if class_weight_tensor is not None else None
        # Support focal loss if gamma is provided
        if focal_loss_gamma is not None and focal_loss_gamma > 0:
            criterion = FocalLoss(alpha=weights, gamma=focal_loss_gamma)
            loss_type = "focal"
        else:
            criterion = nn.CrossEntropyLoss(weight=weights)
            loss_type = "cross_entropy"
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        epochs = 15

        t0 = time.time()
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)

            avg_loss = running_loss / len(train_ds)
            print(f"[ffnn] Epoch {epoch+1}/{epochs} - loss={avg_loss:.4f}")

        train_time = time.time() - t0

        # Evaluation
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t)
            logits_test = model(X_test_t)
            y_proba_val = torch.softmax(logits_val, dim=1).cpu().numpy()
            y_proba_test = torch.softmax(logits_test, dim=1).cpu().numpy()
            y_pred_val_idx = torch.argmax(logits_val, dim=1).cpu().numpy()
            y_pred_test_idx = torch.argmax(logits_test, dim=1).cpu().numpy()
            y_pred_val = label_order[y_pred_val_idx]
            y_pred_test = label_order[y_pred_test_idx]

        params = {
            "type": "torch_ffnn",
            "hidden_layers": [256, 128],
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "device": str(device),
            "loss": loss_type,
            "focal_gamma": focal_loss_gamma if focal_loss_gamma else None,
            "class_weights": weights.cpu().numpy().tolist() if weights is not None else None,
        }
        
        # Cleanup: delete dense arrays and tensors
        del X_train_dense, X_val_dense, X_test_dense
        del X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t
        del model, optimizer, train_loader, train_ds
        _cleanup_memory()

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # ---- Common metrics ----
    # Get probabilities if available (for AUC computation)
    # Note: y_proba_val and y_proba_test are set in each model branch above
    metrics_val, cm_val = build_metric_payload(y_val, y_pred_val, label_order, locals().get('y_proba_val', None), cost_matrix)
    metrics_test, cm_test = build_metric_payload(y_test, y_pred_test, label_order, locals().get('y_proba_test', None), cost_matrix)

    report_val = {
        "model": model_name,
        "split": "val",
        "params": params,
        "train_time_sec": round(train_time, 3),
        "metrics": metrics_val,
    }
    report_test = {
        "model": model_name,
        "split": "test",
        "params": params,
        "train_time_sec": round(train_time, 3),
        "metrics": metrics_test,
    }

    preds = {
        "y_val": y_val,
        "y_pred_val": y_pred_val,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
    }

    return report_val, cm_val, report_test, cm_test, preds


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Run classifiers on preprocessed artifacts (VAL evaluation)."
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="data/processed/meta.json",
        help="Path to meta.json written by preprocessing.",
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports",
        help="Reports root directory.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="(Optional) config to log split strategy.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="dummy,logreg,linearsvc,mlp,ffnn,rf",
        help="Comma-separated model keys to run.",
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        default="balanced",
        help="Class weight for LR/SVMs: None|balanced",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    X_paths = meta["X_artifacts"]
    X_train_path = Path(X_paths["train"]["path"])
    X_val_path = Path(X_paths["val"]["path"])
    X_test_path = Path(X_paths["test"]["path"])
    y_base_dir = Path(args.meta).parent
    y_train_path = Path(y_base_dir / "y_train.csv")
    y_val_path = Path(y_base_dir / "y_val.csv")
    y_test_path = Path(y_base_dir / "y_test.csv")

    # Load data
    X_train = _load_X(X_train_path)
    X_val = _load_X(X_val_path)
    X_test = _load_X(X_test_path)
    y_train = _load_y(y_train_path).astype(int)
    y_val = _load_y(y_val_path).astype(int)
    y_test = _load_y(y_test_path).astype(int)

    # Report dirs
    figures_dir = _ensure_dir(Path(args.reports_dir) / "figures")
    metrics_dir = _ensure_dir(Path(args.reports_dir) / "metrics")
    preds_dir = _ensure_dir(Path(args.reports_dir) / "predictions")

    # Optional: capture split strategy and class weight config from config
    split_info = {}
    class_weight_strategy = "inverse_freq"
    focal_loss_gamma = None
    try:
        cfg = yaml.safe_load(Path(args.config).read_text())
        split_info = {
            "test_size": float(cfg.get("test_size", 0.15)),
            "val_size": float(cfg.get("val_size", 0.15)),
            "stratify": bool(cfg.get("stratify", True)),
        }
        # Read class weight config
        class_weights_cfg = cfg.get("class_weights", {})
        class_weight_strategy = class_weights_cfg.get("strategy", "inverse_freq")
        focal_loss_gamma = class_weights_cfg.get("focal_loss_gamma")
        if focal_loss_gamma is not None:
            focal_loss_gamma = float(focal_loss_gamma)
        # Read logreg config
        logreg_cfg = cfg.get("logreg", {})
        logreg_max_iter = int(logreg_cfg.get("max_iter", 1000))
        logreg_tol = float(logreg_cfg.get("tol", 1e-4))
        # Read hyperparameter tuning config
        tuning_cfg = cfg.get("hyperparameter_tuning", {})
        linearsvc_tuning = tuning_cfg.get("linearsvc", {})
        tune_linearsvc = linearsvc_tuning.get("enabled", False)
        linearsvc_cv_folds = linearsvc_tuning.get("cv_folds", 3)
        rf_tuning = tuning_cfg.get("rf", {})
        tune_rf = rf_tuning.get("enabled", False)
        rf_n_iter = rf_tuning.get("n_iter", 20)
        rf_cv_folds = rf_tuning.get("cv_folds", 3)
        # Read MLP config
        mlp_config = cfg.get("mlp", {})
        # Read cost matrix config
        cost_cfg = cfg.get("cost_matrix", {})
        cost_matrix = None
        if cost_cfg.get("enabled", False):
            cost_matrix = np.array(cost_cfg.get("matrix", []))
    except Exception:
        logreg_max_iter = 1000
        logreg_tol = 1e-4
        tune_linearsvc = False
        linearsvc_cv_folds = 3
        tune_rf = False
        rf_n_iter = 20
        rf_cv_folds = 3
        mlp_config = {}
        cost_matrix = None

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cw = None if (args.class_weight is None or args.class_weight.lower() == "none") else args.class_weight

    label_order = np.unique(np.concatenate([y_train, y_val, y_test]))
    class_weight_tensor = compute_class_weights(y_train, label_order=label_order, strategy=class_weight_strategy)

    summary_rows = []

    for m in models:
        print(f"[run] {m} …")
        torch_weights = class_weight_tensor if m in {"mlp", "ffnn"} else None
        report_val, cm_val, report_test, cm_test, preds = train_and_eval(
            model_name=m,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            random_state=args.seed,
            linear_class_weight=cw if m in {"logreg", "linearsvc"} else None,
            class_weight_tensor=torch_weights,
            label_order=label_order,
            focal_loss_gamma=focal_loss_gamma,
            logreg_max_iter=logreg_max_iter if m == "logreg" else 1000,
            logreg_tol=logreg_tol if m == "logreg" else 1e-4,
            tune_linearsvc=tune_linearsvc if m == "linearsvc" else False,
            linearsvc_cv_folds=linearsvc_cv_folds if m == "linearsvc" else 3,
            tune_rf=tune_rf if m == "rf" else False,
            rf_n_iter=rf_n_iter if m == "rf" else 20,
            rf_cv_folds=rf_cv_folds if m == "rf" else 3,
            mlp_config=mlp_config if m == "mlp" else None,
            cost_matrix=cost_matrix,
        )
        
        # Cleanup memory after each model
        if m in {"logreg", "mlp", "ffnn"}:
            _cleanup_memory()
        
        report_val["split_info"] = split_info
        report_test["split_info"] = split_info

        # Save metrics JSON
        metrics_val_path = metrics_dir / f"{m}_val.json"
        metrics_test_path = metrics_dir / f"{m}_test.json"
        metrics_val_path.write_text(json.dumps(report_val, indent=2))
        metrics_test_path.write_text(json.dumps(report_test, indent=2))

        # Save confusion matrices
        pretty = pretty_model_name(m)
        cm_val_path = figures_dir / f"cm_{m}_val.png"
        cm_test_path = figures_dir / f"cm_{m}_test.png"
        save_confusion_png(
            cm_val, labels=label_order, out_png=cm_val_path, title=f"{pretty} — Validation"
        )
        save_confusion_png(
            cm_test, labels=label_order, out_png=cm_test_path, title=f"{pretty} — Test"
        )

        # Save test predictions
        preds_path = preds_dir / f"{m}_test_predictions.csv"
        pd.DataFrame(
            {
                "y_true": preds["y_test"].astype(int),
                "y_pred": preds["y_pred_test"].astype(int),
            }
        ).to_csv(preds_path, index=False)

        # Add to summary
        summary_rows.append(
            {
                "model": m,
                "split": "val",
                "accuracy": report_val["metrics"]["accuracy"],
                "balanced_accuracy": report_val["metrics"].get("balanced_accuracy", None),
                "macro_f1": report_val["metrics"]["macro_f1"],
                "cohens_kappa": report_val["metrics"].get("cohens_kappa", None),
                "mcc": report_val["metrics"].get("matthews_corrcoef", None),
                "class_1_f1": next((p["f1"] for p in report_val["metrics"]["per_class"] if p["class"] == 1), None),
                "class_4_f1": next((p["f1"] for p in report_val["metrics"]["per_class"] if p["class"] == 4), None),
                "metrics_json": str(metrics_val_path),
                "cm_png": str(cm_val_path),
                "predictions_csv": "",
            }
        )
        summary_rows.append(
            {
                "model": m,
                "split": "test",
                "accuracy": report_test["metrics"]["accuracy"],
                "balanced_accuracy": report_test["metrics"].get("balanced_accuracy", None),
                "macro_f1": report_test["metrics"]["macro_f1"],
                "cohens_kappa": report_test["metrics"].get("cohens_kappa", None),
                "mcc": report_test["metrics"].get("matthews_corrcoef", None),
                "class_1_f1": next((p["f1"] for p in report_test["metrics"]["per_class"] if p["class"] == 1), None),
                "class_4_f1": next((p["f1"] for p in report_test["metrics"]["per_class"] if p["class"] == 4), None),
                "metrics_json": str(metrics_test_path),
                "cm_png": str(cm_test_path),
                "predictions_csv": str(preds_path),
            }
        )

    # Write summary CSV
    pd.DataFrame(summary_rows).to_csv(
        Path(args.reports_dir) / "baselines_summary.csv", index=False
    )
    print(f"[done] Wrote reports to: {args.reports_dir}")


if __name__ == "__main__":
    main()
