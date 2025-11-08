# Traffic Accident Severity Prediction — Baselines & Evaluation

This document describes the **baseline modeling**, **evaluation metrics**, and **report generation** processes following preprocessing.

## Baselines & Evaluation

After preprocessing is complete, you can train and evaluate simple baseline models
to validate the processed artifacts and establish initial performance benchmarks.

### Command
```bash
make baselines
```

### What This Step Does
Running `make baselines` will:
- Load preprocessed datasets from **`data/processed/`**
- Train a set of quick, lightweight models for comparison:
  - **Dummy (Majority Class)** — predicts the most frequent severity class
  - **Logistic Regression (liblinear)** — fast, interpretable linear model
  - **Logistic Regression (saga)** — supports larger sparse datasets
  - **Linear SVM** — strong baseline for linear separation
  - **Random Forest** — simple ensemble tree-based model
- Evaluate each model on the **validation set only** (the **test set remains untouched** for final evaluation)
- Compute and record:
  - Overall **Accuracy**
  - **Macro F1-Score** (equal weight across all classes)
  - **Per-Class Precision, Recall, and F1**
  - A **Normalized Confusion Matrix** (row-wise percentages showing prediction quality per class)
- Record the **split strategy** used (train/val/test ratios, stratified sampling)
  as defined in `configs/default.yaml`.

---

### Outputs & Where to Find Results

All generated reports and evaluation artifacts are saved under the `reports/` directory:

```
reports/
  baselines/
    majority.json
    logreg_liblinear.json
    logreg_saga.json
    linearsvc.json
    rf.json
  figures/
    cm_majority.png
    cm_logreg_liblinear.png
    cm_logreg_saga.png
    cm_linearsvc.png
    cm_rf.png
  baselines_summary.csv
```

#### JSON Reports
Each file in `reports/baselines/` contains:
- Model parameters (e.g., solver, class weight, random seed)
- Accuracy and Macro F1 scores
- Per-class precision, recall, and F1
- The split configuration (train/val/test ratios, stratified flag)
  
These JSON files serve as **machine-readable logs** for your experiments and can be used later for automated performance tracking or plotting.

#### Summary CSV
`reports/baselines_summary.csv` provides a compact, sortable table comparing all models by **accuracy** and **macro-F1**.
You can open this file in Excel, Google Sheets, or Pandas for a quick performance overview.

Example:
| model | accuracy | macro_f1 |
|--------|-----------|----------|
| logreg_saga | 0.74 | 0.68 |
| rf | 0.81 | 0.72 |

#### Confusion Matrices
Each model’s confusion matrix (normalized) is saved in `reports/figures/` as `cm_<model>.png`.

- Each cell value shows the **percentage of samples per true class** predicted as another class.
- Darker diagonals indicate stronger per-class accuracy.
- This makes it easy to identify which severity levels the model struggles to distinguish (e.g., severe accidents mislabeled as minor).

---
