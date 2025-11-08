#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train quick baselines on preprocessed artifacts and evaluate on VAL (hold out TEST).
Saves:
  - reports/figures/cm_<model>.png   (normalized confusion matrix)
  - reports/baselines/<model>.json
  - reports/baselines_summary.csv
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, issparse

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt
import yaml


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


def _maybe_dense_for_rf(X):
    """RandomForest works best with dense arrays. Convert if sparse."""
    if issparse(X):
        print("[rf] Converting sparse CSR to dense for RandomForest (may use more RAM)…")
        return X.toarray()
    return X


# ---------- Visualization ----------

def pretty_model_name(key: str) -> str:
    """Map internal model keys to human-readable titles."""
    return {
        "majority": "Dummy (Majority Class)",
        "logreg_liblinear": "Logistic Regression (liblinear)",
        "logreg_saga": "Logistic Regression (saga)",
        "linearsvc": "Linear SVM",
        "rf": "Random Forest",
    }.get(key, key)


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
    random_state: int,
    class_weight: str | None,
) -> Tuple[Dict, np.ndarray]:
    """Train model and return metrics + confusion matrix."""
    if model_name == "majority":
        clf = DummyClassifier(strategy="most_frequent", random_state=random_state)
        params = {"strategy": "most_frequent", "random_state": random_state}

    elif model_name == "logreg_liblinear":
        clf = LogisticRegression(
            solver="liblinear",
            max_iter=200,
            random_state=random_state,
            class_weight=class_weight,
        )
        params = {"solver": "liblinear", "max_iter": 200, "class_weight": class_weight, "random_state": random_state}

    elif model_name == "logreg_saga":
        clf = LogisticRegression(
            solver="saga",
            max_iter=300,
            penalty="l2",
            random_state=random_state,
            class_weight=class_weight,
        )
        params = {"solver": "saga", "penalty": "l2", "max_iter": 300, "class_weight": class_weight, "random_state": random_state}

    elif model_name == "linearsvc":
        clf = LinearSVC(
            random_state=random_state,
            class_weight=class_weight,
        )
        params = {"class_weight": class_weight, "random_state": random_state}

    elif model_name == "rf":
        X_train = _maybe_dense_for_rf(X_train)
        X_val = _maybe_dense_for_rf(X_val)
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state,
        )
        params = {"n_estimators": 200, "max_depth": None, "n_jobs": -1, "random_state": random_state}

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = clf.predict(X_val)

    acc = float(accuracy_score(y_val, y_pred))
    macro_f1 = float(f1_score(y_val, y_pred, average="macro"))
    p, r, f1s, support = precision_recall_fscore_support(y_val, y_pred, labels=np.unique(y_val), zero_division=0)

    report = {
        "model": model_name,
        "params": params,
        "train_time_sec": round(train_time, 3),
        "metrics": {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "per_class": [
                {"class": int(cls), "precision": float(pi), "recall": float(ri), "f1": float(fi), "support": int(si)}
                for cls, pi, ri, fi, si in zip(np.unique(y_val), p, r, f1s, support)
            ],
            "classification_report": classification_report(y_val, y_pred, digits=4),
        },
    }

    cm = confusion_matrix(y_val, y_pred, labels=np.unique(y_val))
    return report, cm


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Run baseline classifiers on preprocessed artifacts (VAL evaluation).")
    parser.add_argument("--meta", type=str, default="data/processed/meta.json", help="Path to meta.json written by preprocessing.")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Reports root directory.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="(Optional) config to log split strategy.")
    parser.add_argument("--models", type=str, default="majority,logreg_liblinear,logreg_saga,linearsvc,rf",
                        help="Comma-separated: majority,logreg_liblinear,logreg_saga,linearsvc,rf")
    parser.add_argument("--class_weight", type=str, default="balanced", help="Class weight for LR/LinearSVC: None|balanced")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    X_paths = meta["X_artifacts"]
    X_train_path = Path(X_paths["train"]["path"])
    X_val_path = Path(X_paths["val"]["path"])
    y_train_path = Path(Path(args.meta).parent / "y_train.csv")
    y_val_path = Path(Path(args.meta).parent / "y_val.csv")

    # Load data
    X_train = _load_X(X_train_path)
    X_val = _load_X(X_val_path)
    y_train = _load_y(y_train_path)
    y_val = _load_y(y_val_path)

    # Report dirs
    figures_dir = _ensure_dir(Path(args.reports_dir) / "figures")
    json_dir = _ensure_dir(Path(args.reports_dir) / "baselines")

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

    labels = np.unique(y_val)
    summary_rows = []

    for m in models:
        print(f"[run] {m} …")
        report, cm = train_and_eval(
            model_name=m,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            random_state=args.seed,
            class_weight=cw if m in {"logreg_liblinear", "logreg_saga", "linearsvc"} else None,
        )
        report["split"] = split_info

        # Save JSON
        out_json = json_dir / f"{m}.json"
        out_json.write_text(json.dumps(report, indent=2))

        # Save a single normalized confusion matrix
        pretty = pretty_model_name(m)
        out_png = figures_dir / f"cm_{m}.png"
        save_confusion_png(cm, labels=labels, out_png=out_png, title=f"Confusion Matrix — {pretty}")

        # Add to summary
        summary_rows.append({
            "model": m,
            "accuracy": report["metrics"]["accuracy"],
            "macro_f1": report["metrics"]["macro_f1"],
            "json": str(out_json),
            "cm_png": str(out_png),
        })

    # Write summary CSV
    pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False).to_csv(
        Path(args.reports_dir) / "baselines_summary.csv", index=False
    )
    print(f"[done] Wrote reports to: {args.reports_dir}")


if __name__ == "__main__":
    main()
