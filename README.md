# MaintenAI Lab

MaintenAI Lab is a Streamlit workspace for leakage-aware exploration and machine-learning experimentation with the AI4I 2020 Predictive Maintenance dataset.

## Current implementation

Milestone 1 provides a modular application shell, CSV loading, AI4I schema validation, dataset status reporting, and Plotly-backed exploratory data analysis. Milestones 2–4 add a tested ML engine, interactive Model Lab, diagnostics, explainability fallbacks, clustering, and PCA. Milestone 5 adds durable SQLite-backed experiment metadata and comparison history.

## ML engine

`core/ml_engine.py` provides leakage-aware scikit-learn pipelines for classification (Logistic Regression, Decision Tree, Random Forest, KNN, SVM, and Naive Bayes), regression (Linear Regression, Ridge, Decision Tree, Random Forest, and SVR), clustering (K-Means, DBSCAN, and Agglomerative), and PCA.

Supervised runs use a held-out test set and can add K-fold or Stratified K-fold validation on training data. Preprocessing is fitted inside each pipeline/fold. UDI, Product ID, and every AI4I outcome-derived label are rejected as predictors. Optional random oversampling is applied only to training partitions. Model Lab and Clustering & PCA require a dataset that passes AI4I schema validation; loading another dataset or changing the originating configuration clears stale results. Saved experiment metadata is stored in `data/experiments.db`; fitted model objects and predictions are never persisted. Use Experiment History to compare compatible saved runs, export CSV/JSON, or permanently delete a record after confirmation.

## Setup

Use Python 3.10 through 3.13 where possible. From this project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Run automated tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

For an isolated pytest temporary directory, create its parent once before using a nested `--basetemp` path:

```powershell
New-Item -ItemType Directory -Force .\test-temp | Out-Null
.\.venv\Scripts\python.exe -m pytest -q --basetemp .\test-temp\pytest-run
```

## Dataset

Download the official **AI4I 2020 Predictive Maintenance Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), then either upload its CSV in the sidebar or provide a local path. The app does not bundle or invent dataset values.

The loader expects the core AI4I columns: `UDI`, `Product ID`, `Type`, the five sensor/process values, and `Machine failure`. Failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are optional for loading but enrich failure-type EDA.

## Leakage safeguard

When predicting `Machine failure`, `Machine failure` itself and the five failure-mode labels are prohibited as model features. These columns are outcome-derived and would leak target information.

## Project structure

```
app.py              Streamlit entry point and page routing
core/data_loader.py CSV loading, schema validation, leakage checks
core/eda.py         Plotly EDA figure builders
core/ml_engine.py   Reusable model, clustering, PCA, validation engine
core/model_lab.py   Task-specific Model Lab configuration helpers
core/diagnostics.py Real diagnostic data and Plotly figure builders
core/explainability.py Native feature importance and optional SHAP helpers
core/experiment_storage.py SQLite experiment metadata repository
data/               AI4I CSV and local SQLite experiment history
tests/              Validation, leakage, workflow, diagnostics, and storage tests
```
