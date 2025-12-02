# WORKSTREAM1_implementation_plan.md

## Overview
Workstream 1 extends `models/baselines.py` so every baseline (dummy, logistic regression, LinearSVC, PyTorch MLP, PyTorch FFNN, tree-based model) can be re-trained with consistent metrics, class-imbalance handling, and artifacts saved for report use. All code changes stay inside this file unless otherwise noted.

---

## Section 1 — New Baselines

### 1.1 Imports
Add these near the other scikit-learn imports:

```python
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
# Optional: only if we choose XGBoost over RF
# from xgboost import XGBClassifier
```

### 1.2 DummyClassifier Integration
1. Extend `pretty_model_name()` mapping:

```python
"dummy": "Dummy Classifier (Majority)"
```

2. Inside `train_and_eval`, add an `elif model_name == "dummy":` block:

```python
elif model_name == "dummy":
    clf = DummyClassifier(strategy="most_frequent")
    params = {"strategy": "most_frequent"}
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    y_pred_val = clf.predict(X_val)
    y_pred_test = clf.predict(X_test)
```

3. Note: Dummy ignores `class_weight`; document this in the `params` dict for traceability.

### 1.3 RandomForestClassifier (or XGBClassifier)
Choose RF unless GPU acceleration is available. For RF:

```python
elif model_name == "rf":
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
```

- `class_weight="balanced"` helps minority classes without oversampling.
- Store `params` with the hyperparameters above.
- `y_pred_val` / `y_pred_test` mirrors other branches.

If XGBoost is preferred, mirror the branch but use:

```python
elif model_name == "xgb":
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        tree_method="hist",          # or "gpu_hist" if CUDA
        random_state=random_state,
    )
```

Also add `"rf": "Random Forest"` (or `"xgb": "Extreme Gradient Boosting"`) to `pretty_model_name`, include the key in the CLI `--models` default list, and update README/baseline docs later.

### 1.4 Wiring into CLI
- Update the `parser.add_argument("--models", ...)` default to include `dummy` and `rf` (`logreg,linearsvc,mlp,ffnn,dummy,rf`).
- Allow CLI flag `--models dummy,logreg,...` to control runs; the new branches will automatically be picked up.

---

## Section 2 — Class Weights for PyTorch Models

### 2.1 Helper Function
Place above `train_and_eval`:

```python
def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    labels, counts = np.unique(y, return_counts=True)
    freq = counts.astype(np.float32)
    weights = freq.sum() / (len(labels) * freq)
    return torch.tensor(weights, dtype=torch.float32)
```

Return both the tensor and the `labels` order if needed for logging.

### 2.2 Loading and Logging Weights
- Compute once in `main()` after loading `y_train`:

```python
class_weights = compute_class_weights(y_train)
```

- Pass into `train_and_eval` as an optional parameter (`class_weight_tensor`).

- Store `class_weights.tolist()` inside each model’s `report["class_weights"]` for reproducibility.

### 2.3 Applying Weights in PyTorch Branches
1. Move weight tensor to the model device:

```python
weights = class_weight_tensor.to(device) if class_weight_tensor is not None else None
criterion = nn.CrossEntropyLoss(weight=weights)
```

2. Include this for both `mlp` and `ffnn` branches.

3. Document in `params`:

```python
"loss": "CrossEntropyLoss",
"class_weights": weights.cpu().numpy().tolist() if weights is not None else None
```

---

## Section 3 — Metrics & Artifact Saving

### 3.1 Updated Function Signature
Let `train_and_eval` return `(report_val, cm_val, report_test, cm_test, preds_dict)` where `preds_dict` holds `(y_val, y_pred_val, y_test, y_pred_test)`.

### 3.2 Evaluate on Validation and Test
- After fitting, compute predictions for both splits:

```python
y_pred_val = clf.predict(X_val)
y_pred_test = clf.predict(X_test)
```

- Use shared helper to compute metrics and confusion matrices:

```python
def build_metric_payload(y_true, y_pred):
    acc = accuracy_score(...)
    macro_f1 = f1_score(..., average="macro")
    per_cls = precision_recall_fscore_support(...)
    cm = confusion_matrix(...)
    return metrics_dict, cm
```

### 3.3 Saving Artifacts
Inside the run loop in `main()`:

```python
metrics_dir = _ensure_dir(Path(args.reports_dir) / "metrics")
preds_dir = _ensure_dir(Path(args.reports_dir) / "predictions")
figures_dir = _ensure_dir(Path(args.reports_dir) / "figures")

for split, metrics in {"val": report_val, "test": report_test}.items():
    (metrics_dir / f"{model}_{split}.json").write_text(json.dumps(metrics, indent=2))
    save_confusion_png(cm_split, labels, figures_dir / f"{model}_{split}_cm.png", ...)
```

For predictions:

```python
pd.DataFrame({"y_true": y_test, "y_pred": y_pred_test}).to_csv(
    preds_dir / f"{model}_test_predictions.csv", index=False
)
```

### 3.4 File Naming Convention
- Metrics: `reports/metrics/{model}_{split}.json`
- Confusion matrices: `reports/figures/cm_{model}_{split}.png`
- Predictions: `reports/predictions/{model}_{split}_predictions.csv`
- Optional CSV versions of metrics: `reports/metrics/{model}_{split}.csv` if needed for Excel.

---

## Section 4 — Unified Metric Summary

### 4.1 Aggregation Helper
At the end of `main()`:

```python
summary_rows.append({
    "model": model_key,
    "split": "val",
    "accuracy": report_val["metrics"]["accuracy"],
    "macro_f1": report_val["metrics"]["macro_f1"],
    "json": str(metrics_dir / f"{model_key}_val.json"),
    "cm_png": str(figures_dir / f"cm_{model_key}_val.png"),
})
summary_rows.append({... split "test" ...})
```

### 4.2 Writing CSV
Use pandas to create `reports/baselines_summary.csv` with columns:

- `model`
- `split`
- `accuracy`
- `macro_f1`
- `per_class_json` (optional path)
- `cm_png`
- `predictions_csv` (only for test splits)

This file becomes the canonical table referenced in the report (Implementation/Evaluation sections) and can be directly pasted into LaTeX.

---

## Section 5 — Hooks for the Report

The Implementation/Evaluation chapters can now cite:

- `reports/baselines_summary.csv` for the master metrics table (both val and test).
- `reports/figures/cm_{model}_{val|test}.png` for confusion matrices (use the best-performing model’s test matrix in the Error Analysis section).
- `reports/predictions/{model}_test_predictions.csv` for downstream qualitative error analysis (teammate can sample misclassified severity-4 cases).
- Each per-model JSON metrics file for textual descriptions of training setup, including class weights, loss, optimizer, epochs, batch size, and training time.
- Logged class weights (stored in JSON `params` blocks) to describe imbalance handling.
- Training configs (LR solver, SVM penalties, RF depth, MLP/FFNN layer sizes, dropout, epochs, LR, optimizer choices) taken from the `params` dictionaries.

These artifacts fully support the report’s Implementation, Evaluation, and Error Analysis requirements without re-running code.

