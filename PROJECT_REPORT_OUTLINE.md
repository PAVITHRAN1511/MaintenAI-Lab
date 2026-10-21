# MaintenAI Lab — Project Report Outline

## 1. Problem statement

- Describe the goal: explore predictive-maintenance data and run reproducible, leakage-aware machine-learning experiments.
- State the practical concern: predicting or understanding machine failures while avoiding outcome-derived fields as predictors.

## 2. Dataset

- Identify the AI4I 2020 Predictive Maintenance dataset and its source.
- Summarize the process/sensor fields, product type, `Machine failure`, and failure-mode labels.
- Explain validation checks: required schema, numeric/binary constraints, missing values, and dataset fingerprinting.

## 3. Methodology and application design

- Describe the Streamlit workflow: Overview, EDA, Model Lab, Results, Explainability, Clustering & PCA, and Experiment History.
- Explain leakage protection: identifiers and outcome-derived labels are excluded from AI4I outcome prediction and unsupervised defaults.
- Document train/test holdout, K-fold, Stratified K-fold, preprocessing in scikit-learn pipelines, seeds, class weighting, and training-fold-only oversampling.

## 4. Algorithms

- Classification: Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Naive Bayes.
- Regression: Linear/Ridge, Decision Tree/Random Forest Regressors, SVR.
- Unsupervised: K-Means, DBSCAN, Agglomerative Clustering, PCA.

## 5. Evaluation and diagnostics

- Classification: accuracy, precision, recall, macro F1, ROC-AUC when valid, confusion matrix, ROC and precision–recall curves.
- Regression: MAE, RMSE, R², actual-vs-predicted and residual diagnostics.
- Clustering: cluster sizes, silhouette and Calinski–Harabasz when valid.
- PCA: per-component and cumulative explained variance.

## 6. Findings

- Insert only values from saved experiments or exported SQLite history.
- Compare only runs flagged compatible by the application (same dataset fingerprint, target, validation, split, and metric setup).
- Include screenshots/tables of actual EDA and Results output; do not report unexecuted model scores.

## 7. Reproducibility and persistence

- Explain random seeds, stored configuration/metrics/runtime, SQLite transaction handling, dataset fingerprinting, exports, and deletion confirmation.
- Note that fitted models and raw predictions are intentionally not persisted.

## 8. Limitations and future work

- AI4I is synthetic/benchmark data; conclusions should not be generalized to a production plant without external validation.
- Metrics depend on selected split and configuration, and rare failures can make estimates unstable.
- SHAP is optional and currently unavailable unless installed.
- Persisted metadata cannot restore a fitted model for inference.

## 9. Conclusion

- Summarize how MaintenAI Lab supports transparent, reproducible experimentation without fabricating results.
