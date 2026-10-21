# MaintenAI-Lab

MaintenAI-Lab is an end-to-end, leakage-free machine learning laboratory application and empirical analysis suite for predictive maintenance, built upon the AI4I 2020 dataset.

---

## What It Does
MaintenAI-Lab provides an interactive Streamlit workspace and automated benchmarking pipeline designed specifically for high-stakes industrial predictive maintenance:
- **Exploratory Data Analysis (EDA):** Inspects class imbalance, sensor distributions, failure subtypes, and descriptive correlations with anti-leakage warnings.
- **Model Lab:** Interactive classification, clustering, and PCA experimentation across multiple machine learning algorithms with configurable train/test splits, stratified cross-validation, and class imbalance strategies (class-weighted loss and fold-isolated random oversampling).
- **Domain Feature Engineering:** Physics-based domain interaction terms (`Temp difference [K]`, `Power [W]`, `Overstrain [minNm]`, `Temp ratio`) derived from milling thermodynamics and mechanics.
- **Interactive Decision Thresholding:** Dynamic threshold slider (0.05 to 0.95) allowing operators to tune operating points between precision and high-recall failure capture with live metric and confusion matrix updates.
- **Diagnostics & Explainability:** Full diagnostic suite including confusion matrices, ROC and Precision-Recall curves, comparison metrics with a trivial baseline reference line, and global feature importance.
- **Experiment Persistence:** SQLite-backed metadata tracking (`data/experiments.db`) enabling reproducible cross-run comparisons and CSV/JSON export.

---

## Dataset
The project is built upon the **AI4I 2020 Predictive Maintenance Dataset**, available via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (Dataset ID: 601).
- **Origin:** Synthetic benchmark dataset modelled on real milling-machine telemetry, developed by Stephan Matzka (2020).
- **Scale:** 10,000 observations, each representing an operational equipment cycle.
- **Features:** Categorical machine quality `Type` (L, M, H) and 5 continuous sensor measurements: `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]`.
- **Target Variable:** `Machine failure` (binary 0/1).
- **Class Imbalance:** 9,661 healthy cycles (96.61%) and 339 failure events (3.39%).
- **Failure Subtypes:** 5 specific failure modes are recorded for post-hoc analysis: Tool Wear Failure (`TWF`: 46), Heat Dissipation Failure (`HDF`: 115), Power Failure (`PWF`: 95), Overstrain Failure (`OSF`: 98), and Random Failure (`RNF`: 19). Exactly 9 failures exhibit no failure-mode flag.
- **Leakage Prevention:** Identifiers (`UDI`, `Product ID`), `Machine failure`, and the five failure subtype flags are strictly barred from model predictor sets to guarantee valid generalization.

---

## How to Install Dependencies and Run the Streamlit App

### Prerequisites
Python 3.10 through 3.13 on Windows, macOS, or Linux.

### Setup and Execution
```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # On Linux/macOS
# or .\.venv\Scripts\Activate.ps1 on Windows PowerShell

# 2. Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Launch the Streamlit application
streamlit run app.py
```
Open your browser at `http://localhost:8501`. In the sidebar, select **Load from > Local CSV path** and enter `data/ai4i2020.csv` (or upload an AI4I CSV file).

---

## How to Run the Analysis Script
To execute the automated end-to-end experimental benchmarking pipeline from the repository root:

```bash
python analysis/run_analysis.py
```

This single command executes:
1. Class balance and trivial majority baseline verification (`results/class_balance.csv`, `results/baseline.csv`).
2. Raw-feature model comparison across 6 algorithms and 3 imbalance strategies (`results/model_comparison_raw.csv`, `results/figures/model_comparison_raw_f1.png`).
3. Physics-based feature engineering and ablation comparison (`results/model_comparison_engineered.csv`, `results/ablation_comparison.csv`, `results/figures/model_comparison_engineered_f1.png`).
4. Out-of-fold threshold tuning on the training set (`results/threshold_tuning.csv`, `results/figures/pr_curve.png`, `results/figures/roc_curve.png`, `results/figures/confusion_matrix_default.png`, `results/figures/confusion_matrix_tuned.png`).
5. Test-set permutation feature importance with 20 repeats (`results/feature_importance.csv`, `results/figures/feature_importance.png`).
6. Per-failure-mode recall analysis on the test holdout (`results/recall_by_mode.csv`).
7. Programmatic markdown summary compilation (`results/summary.md`).

To run the complete test suite:
```bash
pytest -q
```

---

## Key Results
All metrics below are drawn directly from the verified experimental result files in `results/`:

### 1. Baseline Performance
- **Trivial Baseline Classifier (Always Predict 0):** Accuracy = **0.9661** (96.61%), Failure Precision = 0.0000, Failure Recall = 0.0000, Failure F1 = 0.0000.
- Accuracy is demonstrably deceptive because predicting constant health achieves 96.61% accuracy while failing completely as a maintenance monitoring tool.

### 2. Model Selection and Feature Engineering Impact
Across an 80:20 stratified train/test split (test set: 68 failures, 1,932 normal) with 5-fold stratified cross-validation on the training set:
- **Best Raw-Feature Model:** Random Forest (`class_weight="balanced"`) achieved **0.6602 +/- 0.0406** CV F1, **0.6822** Test F1, and **0.7551** Test PR-AUC. (Unweighted Random Forest holdout recall = 0.5147, holdout PR-AUC = 0.7885).
- **Best Engineered-Feature Model:** Random Forest (`none` imbalance handling) achieved **0.8911 +/- 0.0301** CV F1, **0.9930** Test Accuracy, **0.9655** Test Failure Precision, **0.8235** Test Failure Recall, **0.8889** Test Failure F1, **0.9831** Test ROC-AUC, and **0.8979** Test PR-AUC.
- **Engineering Gain:** Physics-based feature engineering delivered an absolute improvement of **+0.2309** in cross-validated failure F1-score and **+0.2067** in holdout test F1-score.

### 3. Out-of-Fold Threshold Tuning (Held-Out Test Set: 68 Failures, 1,932 Normal)
- **Default Threshold (tau = 0.50):** 0.9930 Accuracy, 0.9655 Precision, 0.8235 Recall, 0.8889 F1 (56 TP, 2 FP, 12 FN, 1,930 TN).
- **Tuned F1-Optimal (tau = 0.48):** 0.9930 Accuracy, 0.9655 Precision, 0.8235 Recall, 0.8889 F1 (56 TP, 2 FP, 12 FN, 1,930 TN).
- **High-Recall Target (tau = 0.07):** 0.9595 Accuracy, 0.4511 Precision, 0.8824 Recall, 0.5970 F1 (60 TP, 73 FP, 8 FN, 1,859 TN). Missed failures are reduced by 33.3% at the cost of 73 false alarms.

### 4. Failure Mode Detection Recall
- **Heat Dissipation Failure (HDF):** 96.55% (28/29 detected)
- **Power Failure (PWF):** 100.0% (13/13 detected)
- **Overstrain Failure (OSF):** 100.0% (16/16 detected)
- **Tool Wear Failure (TWF):** 10.0% (1/10 detected)
- **Random Failure (RNF):** 0.0% (0/0 in test split; synthetic random noise by design)

---

## Project Structure

```
MaintenAI-Lab/
├── analysis/
│   └── run_analysis.py           # Reproducible end-to-end benchmarking script
├── core/
│   ├── data_loader.py            # AI4I schema validation, CSV loader, leakage guard
│   ├── diagnostics.py           # Diagnostic figures (confusion matrix, ROC, PR, baseline line)
│   ├── eda.py                   # Plotly exploratory data analysis visualizers
│   ├── experiment_storage.py    # SQLite durable experiment repository
│   ├── explainability.py        # Permutation and native feature importance
│   ├── features.py              # Physics-derived feature engineering (add_engineered_features)
│   ├── ml_engine.py             # Leakage-aware scikit-learn pipeline execution engine
│   └── model_lab.py             # UI configuration and algorithm parameter dispatchers
├── data/
│   └── ai4i2020.csv             # AI4I 2020 predictive maintenance dataset (10,000 rows)
├── report/
│   ├── generate_docx.py         # Automated DOCX university report exporter
│   ├── report.docx              # Final university project report (Word format)
│   └── report.md                # Final academic lab report (Markdown format)
├── results/
│   ├── ablation_comparison.csv  # Raw vs. engineered feature performance gains
│   ├── baseline.csv             # Trivial majority baseline metrics (0.9661 accuracy)
│   ├── class_balance.csv        # Class imbalance and failure subtype statistics
│   ├── feature_importance.csv   # Test-set permutation feature importance
│   ├── model_comparison_engineered.csv # 6-model comparison on engineered features
│   ├── model_comparison_raw.csv # 6-model comparison on raw features
│   ├── recall_by_mode.csv       # Test-set recall broken down by failure subtype
│   ├── summary.md               # Programmatic headline quantitative summary
│   ├── threshold_tuning.csv     # Out-of-fold threshold optimization metrics
│   └── figures/                 # High-resolution benchmark figures
│       ├── confusion_matrix_default.png
│       ├── confusion_matrix_tuned.png
│       ├── feature_importance.png
│       ├── model_comparison_engineered_f1.png
│       ├── model_comparison_raw_f1.png
│       ├── pr_curve.png
│       └── roc_curve.png
├── tests/
│   ├── test_app_workflow.py     # Streamlit session state and page workflow tests
│   ├── test_data_loader.py      # Schema validation and leakage check tests
│   ├── test_diagnostics.py      # Diagnostic figure generation tests
│   ├── test_eda.py              # Exploratory data analysis tests
│   ├── test_end_to_end.py       # End-to-end model pipeline tests
│   ├── test_experiment_storage.py # SQLite persistence and export tests
│   ├── test_ml_engine.py        # ML engine pipeline and isolation tests
│   ├── test_model_lab.py        # Model Lab configuration tests
│   └── test_prompt4_enhancements.py # Engineered features, thresholding & scoring tests
├── app.py                       # Streamlit interactive application entry point
├── requirements.txt             # Pinned dependency requirements
└── README.md                    # Project documentation
```
