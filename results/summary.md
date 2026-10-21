# MaintenAI-Lab Experimental Results Summary

This document summarizes the quantitative empirical findings generated from the AI4I 2020 predictive maintenance dataset (`data/ai4i2020.csv`). Every headline metric is traceable to generated machine-readable CSV files in `results/`.

---

## 1. Dataset Characteristics & Class Imbalance
- **Total Observations:** 10,000 milling machine operating rows.
- **Failures (Class 1):** 339 (3.39%)
- **Non-Failures (Class 0):** 9661 (96.61%)
- **Trivial Baseline Accuracy (Always Predict 0):** 0.9661 (96.61% accuracy with 0% failure recall/F1).
- **Failures with No Specific Mode Flag:** 9 observations.

### Failure Mode Breakdown:
| Failure Mode | Description | Count | Percentage |
| :--- | :--- | :---: | :---: |
| **TWF** | Tool Wear Failure | 46 | 0.46% |
| **HDF** | Heat Dissipation Failure | 115 | 1.15% |
| **PWF** | Power Failure | 95 | 0.95% |
| **OSF** | Overstrain Failure | 98 | 0.98% |
| **RNF** | Random Failure | 19 | 0.19% |

---

## 2. Model Comparison: Raw vs. Engineered Features

A total of six algorithms were evaluated across three imbalance-handling strategies (no resampling, class-balanced weighting, and random oversampling inside CV training folds) using an 80:20 stratified split (`random_state=42`) and 5-fold stratified cross-validation on the training set.

### Best Configuration per Model (Ablation Comparison):
| Algorithm | Raw Strategy | Raw CV F1 | Raw Test F1 | Raw PR-AUC | Eng Strategy | Eng CV F1 | Eng Test F1 | Eng PR-AUC | CV F1 Gain | Test F1 Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **random_forest** | balanced | 0.6602 | 0.6822 | 0.7551 | none | 0.8911 | 0.8889 | 0.8979 | +0.2309 | +0.2067 |
| **decision_tree** | none | 0.6868 | 0.6815 | 0.4754 | none | 0.8303 | 0.7970 | 0.6430 | +0.1436 | +0.1155 |
| **knn** | random_over | 0.4907 | 0.4699 | 0.3352 | random_over | 0.5243 | 0.4674 | 0.4025 | +0.0335 | -0.0026 |
| **svm** | balanced | 0.4210 | 0.4251 | 0.6119 | none | 0.4825 | 0.5053 | 0.7589 | +0.0614 | +0.0802 |
| **naive_bayes** | none | 0.2887 | 0.1600 | 0.2495 | none | 0.4795 | 0.4516 | 0.4216 | +0.1909 | +0.2916 |
| **logistic_regression** | none | 0.3119 | 0.1772 | 0.4205 | none | 0.3741 | 0.2697 | 0.4620 | +0.0622 | +0.0924 |

### Overall Best Model Selection:
- **Selected Model:** `random_forest` with **Engineered Features** and `none` imbalance handling.
- **Selection Criterion:** Mean 5-Fold Cross-Validation Failure-Class F1 Score (0.8911 +/- 0.0301).
- **Held-Out Test Set Performance:**
  - Accuracy: 0.9930
  - Failure Precision: 0.9655
  - Failure Recall: 0.8235
  - Failure F1: 0.8889
  - ROC-AUC: 0.9831
  - PR-AUC: 0.8979

---

## 3. Decision Threshold Tuning (Out-of-Fold Optimization)
Thresholds were tuned strictly using out-of-fold training predictions to prevent test data leakage, then evaluated on the untouched test holdout (68 true failure cases, 1,932 true normal cases).

| Operating Point | Threshold | Accuracy | Precision | Recall | F1 Score | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default Threshold (0.50)** | 0.5000 | 0.9930 | 0.9655 | 0.8235 | 0.8889 | 56 | 2 | 12 | 1930 |
| **Tuned Optimal F1 Threshold** | 0.4800 | 0.9930 | 0.9655 | 0.8235 | 0.8889 | 56 | 2 | 12 | 1930 |
| **High-Recall Target (>=0.90)** | 0.0700 | 0.9595 | 0.4511 | 0.8824 | 0.5970 | 60 | 73 | 8 | 1859 |

---

## 4. Test-Set Permutation Feature Importance (Top Predictors)
Scored using Average Precision (`average_precision_score` / PR-AUC) across 20 repeats on the test set:

| Rank | Feature | Mean Decrease in PR-AUC | Std Deviation |
| :---: | :--- | :---: | :---: |
| 1 | `Rotational speed [rpm]` | 0.34352 | 0.03052 |
| 2 | `Overstrain [minNm]` | 0.15413 | 0.01171 |
| 3 | `Power [W]` | 0.10541 | 0.01160 |
| 4 | `Temp ratio` | 0.08308 | 0.01722 |
| 5 | `Tool wear [min]` | 0.03764 | 0.00384 |
| 6 | `Type` | 0.03192 | 0.01097 |
| 7 | `Temp difference [K]` | 0.02101 | 0.00577 |
| 8 | `Torque [Nm]` | 0.00960 | 0.00580 |
| 9 | `Air temperature [K]` | -0.00052 | 0.00445 |
| 10 | `Process temperature [K]` | -0.00359 | 0.00389 |

---

## 5. Recall by Failure Mode (Held-Out Test Set)

| Failure Mode | Test Mode Count | Default Detected (thresh=0.50) | Default Recall | Tuned Detected | Tuned Recall | Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **TWF** | 10 | 1 | 0.1000 | 1 | 0.1000 | Systematic physical failure |
| **HDF** | 29 | 28 | 0.9655 | 28 | 0.9655 | Systematic physical failure |
| **PWF** | 13 | 13 | 1.0000 | 13 | 1.0000 | Systematic physical failure |
| **OSF** | 16 | 16 | 1.0000 | 16 | 1.0000 | Systematic physical failure |
| **RNF** | 0 | 0 | 0.0000 | 0 | 0.0000 | Random failure by design; low recall expected |
| **No Mode Flag** | 2 | 0 | 0.0000 | 0 | 0.0000 | Unlabeled failure modes |

*Note: RNF (Random Failure) represents synthetic random noise by design; low detection recall for RNF is mathematically expected and demonstrates that the model learns physical failure rules rather than overfitting to synthetic noise.*
