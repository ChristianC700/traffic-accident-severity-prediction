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
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, issparse

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.svm import LinearSVC

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


# ---------- Class imbalance helpers ----------

def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency weights for CrossEntropyLoss."""
    labels, counts = np.unique(y, return_counts=True)
    counts = counts.astype(np.float32)
    weights = counts.sum() / (len(labels) * counts)
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


def build_metric_payload(y_true, y_pred, label_order: np.ndarray | None = None):
    """Return metrics dict + confusion matrix for the provided split."""
    if label_order is None:
        label_order = np.unique(y_true)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    precision, recall, f1s, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_order, zero_division=0
    )
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": [
            {
                "class": int(cls),
                "precision": float(pi),
                "recall": float(ri),
                "f1": float(fi),
                "support": int(si),
            }
            for cls, pi, ri, fi, si in zip(label_order, precision, recall, f1s, support)
        ],
        "classification_report": classification_report(
            y_true, y_pred, labels=label_order, digits=4
        ),
    }
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
) -> Tuple[Dict, np.ndarray, Dict, np.ndarray, Dict[str, np.ndarray]]:
    """Train model and return metrics for validation + test splits."""

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

    # ---- Logistic Regression ----
    elif model_name == "logreg":
        clf = LogisticRegression(
            solver="saga",
            penalty="l2",
            C=0.1,
            max_iter=5000,
            tol=1e-3,
            random_state=random_state,
            class_weight=linear_class_weight,
            n_jobs=-1,
        )
        params = {
            "solver": "saga",
            "penalty": "l2",
            "max_iter": 5000,
            "tol": 1e-3,
            "class_weight": linear_class_weight,
            "random_state": random_state,
            "n_jobs": -1,
        }

        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred_val = clf.predict(X_val)
        y_pred_test = clf.predict(X_test)

    # ---- Linear SVM ----
    elif model_name == "linearsvc":
        clf = LinearSVC(
            random_state=random_state,
            class_weight=linear_class_weight,
        )
        params = {
            "class_weight": linear_class_weight,
            "random_state": random_state,
        }

        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred_val = clf.predict(X_val)
        y_pred_test = clf.predict(X_test)

    # ---- Random Forest ----
    elif model_name == "rf":
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

    # ---- Neural Network (single-hidden-layer MLP, PyTorch) ----
    elif model_name == "mlp":
        X_train_dense = _maybe_dense(X_train, tag="[mlp]")
        X_val_dense = _maybe_dense(X_val, tag="[mlp]")
        X_test_dense = _maybe_dense(X_test, tag="[mlp]")

        device = _get_torch_device()

        # Convert to torch tensors
        X_train_t = torch.from_numpy(X_train_dense.astype(np.float32))
        y_train_t = torch.from_numpy(y_train.astype(np.int64))
        X_val_t = torch.from_numpy(X_val_dense.astype(np.float32))
        y_val_t = torch.from_numpy(y_val.astype(np.int64))
        X_test_t = torch.from_numpy(X_test_dense.astype(np.float32))
        y_test_t = torch.from_numpy(y_test.astype(np.int64))

        num_features = X_train_t.shape[1]
        num_classes = int(np.unique(y_train).shape[0])

        class MLP(nn.Module):
            def __init__(self, in_dim, hidden_dim, out_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, out_dim),
                )

            def forward(self, x):
                return self.net(x)

        model = MLP(num_features, hidden_dim=128, out_dim=num_classes).to(device)

        # Dataloaders
        batch_size = 1024
        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        X_val_t = X_val_t.to(device)
        y_val_t = y_val_t.to(device)
        X_test_t = X_test_t.to(device)
        y_test_t = y_test_t.to(device)

        weights = class_weight_tensor.to(device) if class_weight_tensor is not None else None
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        epochs = 10

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
            print(f"[mlp] Epoch {epoch+1}/{epochs} - loss={avg_loss:.4f}")

        train_time = time.time() - t0

        # Evaluation
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t)
            logits_test = model(X_test_t)
            y_pred_val = torch.argmax(logits_val, dim=1).cpu().numpy()
            y_pred_test = torch.argmax(logits_test, dim=1).cpu().numpy()

        params = {
            "type": "torch_mlp",
            "hidden_dim": 128,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": 1e-3,
            "device": str(device),
            "class_weights": weights.cpu().numpy().tolist() if weights is not None else None,
        }

    # ---- Two-Layer Feed-Forward Neural Network (PyTorch) ----
    elif model_name == "ffnn":
        X_train_dense = _maybe_dense(X_train, tag="[ffnn]")
        X_val_dense = _maybe_dense(X_val, tag="[ffnn]")
        X_test_dense = _maybe_dense(X_test, tag="[ffnn]")

        device = _get_torch_device()

        # Convert to torch tensors
        X_train_t = torch.from_numpy(X_train_dense.astype(np.float32))
        y_train_t = torch.from_numpy(y_train.astype(np.int64))
        X_val_t = torch.from_numpy(X_val_dense.astype(np.float32))
        y_val_t = torch.from_numpy(y_val.astype(np.int64))
        X_test_t = torch.from_numpy(X_test_dense.astype(np.float32))
        y_test_t = torch.from_numpy(y_test.astype(np.int64))

        num_features = X_train_t.shape[1]
        num_classes = int(np.unique(y_train).shape[0])

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

        # Dataloaders
        batch_size = 1024
        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        X_val_t = X_val_t.to(device)
        y_val_t = y_val_t.to(device)
        X_test_t = X_test_t.to(device)
        y_test_t = y_test_t.to(device)

        weights = class_weight_tensor.to(device) if class_weight_tensor is not None else None
        criterion = nn.CrossEntropyLoss(weight=weights)
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
            y_pred_val = torch.argmax(logits_val, dim=1).cpu().numpy()
            y_pred_test = torch.argmax(logits_test, dim=1).cpu().numpy()

        params = {
            "type": "torch_ffnn",
            "hidden_layers": [256, 128],
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "device": str(device),
            "class_weights": weights.cpu().numpy().tolist() if weights is not None else None,
        }

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # ---- Common metrics ----
    metrics_val, cm_val = build_metric_payload(y_val, y_pred_val, label_order)
    metrics_test, cm_test = build_metric_payload(y_test, y_pred_test, label_order)

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

    # Optional: capture split strategy from config
    split_info = {}
    try:
        cfg = yaml.safe_load(Path(args.config).read_text())
        split_info = {
            "test_size": float(cfg.get("test_size", 0.15)),
            "val_size": float(cfg.get("val_size", 0.15)),
            "stratify": bool(cfg.get("stratify", True)),
        }
    except Exception:
        pass

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cw = None if (args.class_weight is None or args.class_weight.lower() == "none") else args.class_weight

    label_order = np.unique(np.concatenate([y_train, y_val, y_test]))
    class_weight_tensor = compute_class_weights(y_train)

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
        )
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
                "macro_f1": report_val["metrics"]["macro_f1"],
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
                "macro_f1": report_test["metrics"]["macro_f1"],
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
