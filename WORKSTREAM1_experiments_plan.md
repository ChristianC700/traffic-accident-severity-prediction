# WORKSTREAM1_experiments_plan.md

## Section 1 — Experiments to Run

| Experiment | Config/Input | Key Hyperparameters | Notes / Runtime |
| --- | --- | --- | --- |
| DummyClassifier (`dummy`) | `data/processed/` artifacts | `strategy="most_frequent"` | Seconds; baseline sanity check |
| Logistic Regression (`logreg`) | `configs/default.yaml` | `solver="saga"`, `C=0.1`, `class_weight="balanced"`, `max_iter=5000` | ~2–3 min on CPU; sparse-friendly |
| LinearSVC (`linearsvc`) | `configs/default.yaml` | default `LinearSVC` + `class_weight="balanced"` | ~3–5 min |
| PyTorch MLP (`mlp`) | `configs/default.yaml` | 1 hidden layer (128), dropout 0.2, epochs 10, Adam lr 1e-3, batch 1024, **class weights applied** | ~10 min on CPU (faster if MPS) |
| PyTorch FFNN (`ffnn`) | `configs/default.yaml` | Hidden layers 256/128, dropout 0.3, epochs 15, AdamW lr 1e-3, wd 1e-4, **class weights applied** | ~15 min |
| Random Forest (`rf`) | `configs/default.yaml` | `n_estimators=300`, `max_depth=20`, `min_samples_leaf=5`, `class_weight="balanced"`, `n_jobs=-1` | ~10 min depending on cores |
| (Optional) XGBoost (`xgb`) | `configs/default.yaml` | `n_estimators=400`, `max_depth=8`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8` | Requires `pip install xgboost`; GPU optional |

All experiments read the same preprocessed artifacts under `data/processed/`. No alternate configs are required unless running feature ablations later.

---

## Section 2 — How to Run Them

1. **Environment**
   ```powershell
   pip install -r requirements.txt
   # If using XGBoost
   pip install xgboost
   ```

2. **Preprocessing (only if artifacts stale)**
   ```powershell
    make preprocess
   ```

3. **Run all models in one sweep**
   ```powershell
   python models/baselines.py ^
     --meta data/processed/meta.json ^
     --reports_dir reports ^
     --config configs/default.yaml ^
     --models dummy,logreg,linearsvc,mlp,ffnn,rf
   ```

4. **Run individual model (rerun/failures)**
   ```powershell
   python models/baselines.py --models mlp
   ```

5. **Validation + Test evaluation**
   - Script automatically evaluates both splits; inspect outputs in `reports/metrics/` and `reports/figures/`.

6. **Predictions for downstream analysis**
   - `reports/predictions/{model}_test_predictions.csv` is created for each model; hand off to teammates for qualitative/error analysis.

---

## Section 3 — Output Checklist

- [ ] `reports/baselines_summary.csv`
- [ ] `reports/metrics/{model}_val.json` and `reports/metrics/{model}_test.json`
- [ ] `reports/figures/cm_{model}_val.png` and `reports/figures/cm_{model}_test.png`
- [ ] `reports/predictions/{model}_test_predictions.csv`
- [ ] Console logs (optional) captured for training time per model

Verify counts: 6 models × 2 splits → 12 metric JSONs + 12 confusion PNGs minimum, plus 6 prediction CSVs.

---

## Section 4 — Report Hooks

- **Implementation**: Use the hyperparameters/optimizers recorded in the per-model JSON metrics (`params` block) to describe setup and class-imbalance handling (class weights, `class_weight="balanced"` for traditional models).
- **Evaluation**: Pull main comparison table from `reports/baselines_summary.csv`; include both validation and test rows. Reference confusion matrices for the best-performing model (val for tuning rationale, test for final results).
- **Error Analysis**: Teammates can load `reports/predictions/{model}_test_predictions.csv` to inspect misclassified severity-4 cases or compute per-condition error summaries. Confusion matrices (test split) go directly into the Error Analysis section.
- **Progress documentation**: Mention that Dummy and RandomForest baselines were added, class weights applied to neural nets, and both splits are evaluated consistently with saved artifacts.

