# Manual Baseline Feature Engineering

This folder contains manual/hand-crafted baseline feature engineering assets for
RelBench tasks. The notebooks show how to build baseline features and then train
baseline models (LightGBM today, with pointers on how to swap in other models).

## 1) User attendance baseline (rel-event-user-attendance)

**Files**
- `feats (2).sql`: SQL template for creating a feature table with monthly
  rolling-window aggregates.
- `user-attendance.ipynb`: notebook that renders the SQL template and trains a
  baseline model.

**How to run**
1. Open `user-attendance.ipynb` in Jupyter.
2. Adjust the SQL path in the notebook to point to the SQL template in this
   folder (e.g. `baselines/feats (2).sql`).
3. Ensure you have a DuckDB database with the `rel-event` dataset loaded (the
   notebook assumes a local `event/event.db`).
4. Run the cells to create `user_attendance_{train,val,test}_feats` tables and
   then train/evaluate the baseline model using the feature tables.

**Using other models**
- The notebook already materializes a `torch_frame` TensorFrame, so you can swap
  the model block to use `XGBoost` from `torch_frame.gbdt`, TabPFN, or any
  scikit-learn model that accepts NumPy arrays (e.g. `RandomForestClassifier`).
- RealMLP can be used as well, but it does not accept missing values. You need to
  impute NaNs (e.g. fill with 0 or use a simple imputer) before fitting.

## 2) Study outcome baseline (rel-trial-study-outcome)

**File**
- `outcome_feature_eng.ipynb`: builds manual features from `studies`,
  `designs`, `eligibilities`, and historical trial success stats.

**How to run**
1. Open `outcome_feature_eng.ipynb`.
2. The notebook downloads `rel-trial` via `relbench` and derives features from
   the relational tables.
3. Run the model section to fit a baseline and evaluate on validation/test.

**Using other models**
- The notebook uses `torch_frame` to infer feature types and build a
  `Dataset/TensorFrame`. You can replace `LightGBM` with
  `torch_frame.gbdt.XGBoost`, TabPFN, or extract `.tensor_frame` data to feed any
  other baseline model (e.g. sklearn logistic regression).
- RealMLP can be used as well, but it requires NaN-free inputs, so add an
  imputation step before training.

## 3) Site success baseline (rel-trial-site-success)

**File**
- `site_feature_eng.ipynb`: builds site history features (counts, phases,
  historical success/adverse events, etc.) and trains a baseline regressor.

**How to run**
1. Open `site_feature_eng.ipynb`.
2. The notebook downloads `rel-trial` via `relbench` and constructs
   site-level historical features.
3. Run the model section to fit a baseline and evaluate on validation/test.

**Using other models**
- Same approach as above: swap the `LightGBM` block for `XGBoost`, TabPFN, or
  export the TensorFrame to NumPy/pandas for alternative regressors.
- RealMLP can be used as well, but it requires NaN-free inputs, so add an
  imputation step before training.

## Dependencies

Install the project dependencies plus at least one baseline model library, for
example:

```bash
pip install -r requirements.txt
pip install lightgbm xgboost scikit-learn tabpfn realmlp
```

If you use the notebooks directly, ensure Jupyter is installed and available in
your environment.
