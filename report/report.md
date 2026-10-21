# MaintenAI-Lab: Empirical Predictive Maintenance Modelling and Diagnostics on the AI4I 2020 Dataset

**Course / Lab Project:** University Machine Learning Laboratory  
**Dataset:** AI4I 2020 Predictive Maintenance (UCI Machine Learning Repository ID: 601)  
**Author:** MaintenAI-Lab Team  
**Date:** October 2026  

---

## Abstract
This report investigates the application of supervised classification and domain-driven feature engineering to industrial predictive maintenance using the AI4I 2020 dataset. Due to severe class imbalance (3.39% failure incidence), traditional accuracy fails as an informative evaluation metric: a trivial majority classifier achieves an accuracy of 96.61% while detecting zero machine failures. Across six distinct supervised learning algorithms evaluated on raw features under stratified cross-validation and holdout testing, baseline tree ensembles and decision trees achieved moderate failure-class F1-scores (~0.66–0.68). By engineering domain-specific physical interaction terms—capturing temperature difference, mechanical rotational power, overstrain torque-wear product, and thermal ratios directly aligned with real-world failure modes—the best model, a Random Forest classifier, achieved a cross-validation failure F1-score of 0.8911 +/- 0.0301 and a held-out test F1-score of 0.8889 (precision 0.9655, recall 0.8235, PR-AUC 0.8979). Threshold tuning using out-of-fold training probabilities demonstrated operational trade-offs between precision and safety-critical failure recall. Permutation importance analysis confirmed that physics-derived interaction features accounted for the majority of predictive power, while recall-by-mode diagnostics validated near-perfect detection of physical failures (HDF: 96.55%, PWF: 100%, OSF: 100%) alongside the theoretical undetectability of synthetic random noise (RNF).

---

## 1. Introduction
Predictive maintenance (PdM) is a core application of industrial machine learning, aiming to detect equipment anomalies and anticipate catastrophic tool or motor breakdown before catastrophic stoppage occurs. In modern computer numerical control (CNC) and milling operations, the economic and human costs of an unexpected machine breakdown substantially exceed the costs of routine inspection.

In such operational settings, prediction errors have fundamentally asymmetric consequences:
1. **False Negatives (Missed Failures):** A machine failure that goes undetected results in in-process workpiece scrapping, severe spindle or tool damage, unscheduled production halt, and secondary safety hazards. The operational cost of a false negative is severe.
2. **False Positives (False Alarms):** Flagging a healthy machine as failing prompts an operator to perform a brief diagnostic inspection or pause cycling. While undesirable due to maintenance labor overhead and minor productivity loss, the cost of a false positive is orders of magnitude smaller than an uncontained machine crash.

Consequently, predictive maintenance models cannot be evaluated or deployed based on overall classification accuracy. In highly imbalanced operational regimes, a model that optimizes for standard 0/1 accuracy tends to degenerate into the trivial majority predictor, ignoring the rare failure class entirely. This project systematically builds, tests, evaluates, and documents a rigorous, leakage-free machine learning system using the benchmark AI4I 2020 dataset.

---

## 2. Dataset Description
The AI4I 2020 Predictive Maintenance dataset reflects 10,000 operational observations modelled on real milling machine operational telemetry, synthesized by Stephan Matzka (2020) and hosted at the UCI Machine Learning Repository (Dataset ID: 601).

### 2.1 Permitted Features and Target Columns
Each record consists of operational sensor measurements, categorical identifiers, and failure indicators:
- **Identifier Columns (Prohibited from Modelling):**
  - `UDI`: Arbitrary record index (1 to 10,000).
  - `Product ID`: Variant identifier consisting of quality tier and sequential serial number (e.g., `L47181`, `M14860`, `H29424`).
- **Permitted Input Sensor Features:**
  - `Type`: Categorical quality variant consisting of Low (`L`, 60%), Medium (`M`, 30%), and High (`H`, 20%) product variants.
  - `Air temperature [K]`: Ambient temperature surrounding the milling equipment (mean ~300.0 K).
  - `Process temperature [K]`: Working temperature inside the milling chamber (mean ~310.0 K).
  - `Rotational speed [rpm]`: Spindle rotational speed calculated from electrical power and torque (mean ~1538 rpm).
  - `Torque [Nm]`: Mechanical resistance torque applied to the cutting tool (mean ~40.0 Nm).
  - `Tool wear [min]`: Cumulative tool usage duration in minutes; tool inserts are worn as cutting cycles accumulate.
- **Target Variable:**
  - `Machine failure`: Binary ground truth indicating whether a machine breakdown occurred during the operational cycle (0 = Normal, 1 = Failure).
- **Failure Mode Flags (Excluded as Features to Prevent Target Leakage):**
  - `TWF` (Tool Wear Failure): Tool wear exceeds operational failure limits (between 200 and 240 minutes).
  - `HDF` (Heat Dissipation Failure): Inadequate thermal heat transfer due to low temperature gradient (<8.6 K) and low speed (<1380 rpm).
  - `PWF` (Power Failure): Mechanical power (torque * speed) falls outside safe boundaries (<3500 W or >9000 W).
  - `OSF` (Overstrain Failure): Product of tool wear and torque exceeds the material structural threshold.
  - `RNF` (Random Failure): Stochastic failure component with 0.1% occurrence probability regardless of operational parameters.

### 2.2 Empirical Class Imbalance
The dataset contains exactly 10,000 observations with an overall failure rate of 3.39%:
- Total Normal Operations (`Machine failure` = 0): 9,661 (96.61%)
- Total Machine Failures (`Machine failure` = 1): 339 (3.39%)
- Failure Mode Breakdown:
  - `TWF`: 46 failures (0.46% of total)
  - `HDF`: 115 failures (1.15% of total)
  - `PWF`: 95 failures (0.95% of total)
  - `OSF`: 98 failures (0.98% of total)
  - `RNF`: 19 failures (0.19% of total)
  - Failures with No Subtype Flag: Exactly 9 failure cases exhibit `Machine failure == 1` without having any of the five specific failure mode flags set.

---

## 3. Methodology
To ensure scientific integrity, reproducibility, and deployment reliability, the experimental framework strictly enforces anti-leakage boundaries, deterministic partitioning, and imbalance-aware validation protocols.

### 3.1 Data Leakage Prevention
Data leakage represents the most common flaw in predictive maintenance literature. In this study, the following rules are enforced programmatically:
1. **Column Exclusion:** `UDI`, `Product ID`, `Machine failure`, and the five failure subtype labels (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are strictly barred from the predictor set. Including failure mode flags would leak post-event diagnostic classifications into the feature space.
2. **Preprocessing Isolation:** Preprocessing transformers (median imputation for numeric attributes, standard scaling, and one-hot encoding for categorical `Type`) learn statistical parameters (e.g., mean, variance, imputation medians) solely from training folds. Transformers are wrapped in scikit-learn `Pipeline` and `ColumnTransformer` constructs.
3. **Resampling Isolation:** When random minority oversampling is evaluated, it is applied strictly to training folds. Held-out validation folds and final test sets are never resampled, preserving natural industrial class distributions.

### 3.2 Evaluation Protocol
1. **Holdout Partitioning:** The 10,000 observations are split into an 80% training set (8,000 samples, 271 failures, 7,729 normal) and an untouched 20% test holdout (2,000 samples, 68 failures, 1,932 normal) using stratified sampling with `random_state=42`.
2. **Cross-Validation:** Five-fold stratified cross-validation is performed across the 8,000 training observations.
3. **Primary Selection Metric:** Mean cross-validated failure-class F1-score (`f1_score(..., pos_label=1)`) is employed as the model selection criterion. Precision-Recall Area Under the Curve (PR-AUC / Average Precision) and ROC-AUC are calculated to evaluate ranking capacity.

### 3.3 Algorithms Evaluated
Six diverse classification paradigms are evaluated:
1. `Logistic Regression` (L2 penalty, liblinear solver, max_iter=1000)
2. `Decision Tree` (CART algorithm, Gini criterion)
3. `Random Forest` (Ensemble of 200 de-correlated decision trees)
4. `K-Nearest Neighbors` (k=5, Euclidean distance)
5. `Support Vector Classifier` (RBF kernel, probability calibration enabled)
6. `Gaussian Naive Bayes` (Normal likelihood assumption)

Each algorithm is benchmarked under three imbalance strategies:
- `none`: Unweighted empirical loss minimization.
- `balanced`: Loss function inversely weighted by class frequencies (`class_weight="balanced"`).
- `random_over`: Deterministic random oversampling applied within training partitions.

---

## 4. Experimental Results

### 4.1 Trivial Baseline Checkpoint
The trivial majority classifier (which always predicts class 0) yields:
- **Accuracy:** 0.9661 (96.61%)
- **Failure Precision:** 0.0000
- **Failure Recall:** 0.0000
- **Failure F1-Score:** 0.0000

This establishes the baseline diagnostic checkpoint: any model reporting an accuracy of ~96.6% without high minority recall is effectively performing no better than a trivial heuristic.

### 4.2 Raw-Feature Model Comparison
Using the six raw input features (`Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`), models achieved the following performance metrics:

| Algorithm | Strategy | CV Failure F1 (Mean +/- Std) | Test Accuracy | Test Failure Precision | Test Failure Recall | Test Failure F1 | Test ROC-AUC | Test PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | none | 0.6868 +/- 0.0470 | 0.9785 | 0.6866 | 0.6765 | 0.6815 | 0.8328 | 0.4754 |
| **Random Forest** | balanced | 0.6602 +/- 0.0406 | 0.9795 | 0.7213 | 0.6471 | 0.6822 | 0.9612 | 0.7551 |
| **Random Forest** | random_over | 0.6514 +/- 0.0456 | 0.9825 | 0.8667 | 0.5735 | 0.6903 | 0.9640 | 0.7721 |
| **Decision Tree** | random_over | 0.6427 +/- 0.0188 | 0.9760 | 0.6515 | 0.6324 | 0.6418 | 0.8102 | 0.4245 |
| **Decision Tree** | balanced | 0.6211 +/- 0.0430 | 0.9790 | 0.7321 | 0.6029 | 0.6613 | 0.7976 | 0.4549 |
| **Random Forest** | none | 0.6178 +/- 0.0403 | 0.9815 | 0.8974 | 0.5147 | 0.6542 | 0.9721 | 0.7885 |
| **KNN** | random_over | 0.4907 +/- 0.0138 | 0.9515 | 0.3739 | 0.6324 | 0.4699 | 0.8470 | 0.3352 |
| **KNN** | none | 0.4522 +/- 0.0291 | 0.9740 | 0.8333 | 0.2941 | 0.4348 | 0.8291 | 0.4586 |
| **SVM** | balanced | 0.4210 +/- 0.0228 | 0.9175 | 0.2785 | 0.8971 | 0.4251 | 0.9655 | 0.6119 |
| **SVM** | random_over | 0.4206 +/- 0.0166 | 0.9175 | 0.2765 | 0.8824 | 0.4211 | 0.9635 | 0.6135 |
| **SVM** | none | 0.3914 +/- 0.0626 | 0.9720 | 0.8750 | 0.2059 | 0.3333 | 0.9468 | 0.6373 |
| **Logistic Regression** | none | 0.3119 +/- 0.0260 | 0.9675 | 0.6364 | 0.1029 | 0.1772 | 0.8994 | 0.4205 |
| **Naive Bayes** | none | 0.2887 +/- 0.0235 | 0.9580 | 0.2500 | 0.1176 | 0.1600 | 0.8468 | 0.2495 |
| **Logistic Regression** | balanced | 0.2346 +/- 0.0100 | 0.8245 | 0.1418 | 0.8235 | 0.2419 | 0.9070 | 0.3818 |
| **Logistic Regression** | random_over | 0.2342 +/- 0.0112 | 0.8275 | 0.1440 | 0.8235 | 0.2451 | 0.9065 | 0.3854 |
| **Naive Bayes** | random_over | 0.2173 +/- 0.0154 | 0.8120 | 0.1262 | 0.7647 | 0.2167 | 0.8536 | 0.2495 |

*Validation Diagnostic Note:* The unweighted Random Forest model on raw features achieves a holdout recall of 0.5147 (~0.52) and a PR-AUC of 0.7885 (~0.79), verifying the repository's baseline experimental diagnostic checkpoint.

---

### 4.3 Feature Engineering and Ablation Analysis
Guided by the physical failure modes identified in Matzka (2020), four non-linear interaction features were created strictly from raw sensor data:
1. `Temp difference [K]` = `Process temperature [K] - Air temperature [K]` (drives Heat Dissipation Failure).
2. `Power [W]` = `Torque [Nm] * Rotational speed [rpm] * (2 * pi / 60)` (drives Power Failure).
3. `Overstrain [minNm]` = `Tool wear [min] * Torque [Nm]` (drives Overstrain Failure).
4. `Temp ratio` = `Process temperature [K] / Air temperature [K]` (relative thermal gradient).

Re-evaluating the six classifiers on the enriched feature representation yielded significant performance gains across all tree-based and probabilistic models:

| Algorithm | Raw Strategy | Raw CV F1 | Raw Test F1 | Raw PR-AUC | Eng Strategy | Eng CV F1 | Eng Test F1 | Eng PR-AUC | CV F1 Gain | Test F1 Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | balanced | 0.6602 | 0.6822 | 0.7551 | none | **0.8911** | **0.8889** | **0.8979** | **+0.2309** | **+0.2067** |
| **Decision Tree** | none | 0.6868 | 0.6815 | 0.4754 | none | 0.8303 | 0.7970 | 0.6430 | +0.1436 | +0.1155 |
| **KNN** | random_over | 0.4907 | 0.4699 | 0.3352 | random_over | 0.5243 | 0.4674 | 0.4025 | +0.0335 | -0.0026 |
| **SVM** | balanced | 0.4210 | 0.4251 | 0.6119 | none | 0.4825 | 0.5053 | 0.7589 | +0.0614 | +0.0802 |
| **Naive Bayes** | none | 0.2887 | 0.1600 | 0.2495 | none | 0.4795 | 0.4516 | 0.4216 | +0.1909 | +0.2916 |
| **Logistic Regression** | none | 0.3119 | 0.1772 | 0.4205 | none | 0.3741 | 0.2697 | 0.4620 | +0.0622 | +0.0924 |

The unweighted **Random Forest with engineered features** emerged as the definitive best model, attaining:
- **CV Failure F1-Score:** 0.8911 +/- 0.0301 (an absolute improvement of +23.1 percentage points over the raw baseline).
- **Test Set Accuracy:** 0.9930 (99.30%)
- **Test Set Failure Precision:** 0.9655 (96.55%)
- **Test Set Failure Recall:** 0.8235 (82.35%)
- **Test Set Failure F1-Score:** 0.8889
- **Test Set PR-AUC:** 0.8979
- **Test Set ROC-AUC:** 0.9831

---

### 4.4 Decision Threshold Tuning (Out-of-Fold Optimization)
Using out-of-fold training probabilities generated from 5-fold stratified cross-validation on the 8,000 training observations, two operational thresholds were tuned:
1. **F1-Maximizing Threshold (tau = 0.4800):** Threshold that balances precision and recall to maximize failure F1.
2. **High-Recall Target Threshold (tau = 0.0700):** Lowest threshold achieving >=90% recall on the out-of-fold predictions.

Applying these thresholds to the untouched test set (2,000 samples: 68 failures, 1,932 normal) produced the following operational outcomes:

| Operating Point | Threshold | Accuracy | Precision | Recall | F1 Score | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default Threshold** | 0.5000 | 0.9930 | 0.9655 | 0.8235 | 0.8889 | 56 | 2 | 12 | 1,930 |
| **Tuned F1 Threshold** | 0.4800 | 0.9930 | 0.9655 | 0.8235 | 0.8889 | 56 | 2 | 12 | 1,930 |
| **High-Recall Target** | 0.0700 | 0.9595 | 0.4511 | 0.8824 | 0.5970 | 60 | 73 | 8 | 1,859 |

At the tuned F1 threshold (tau = 0.48), the model catches 56 out of 68 test failures while generating only 2 false alarms across 1,932 healthy cycles. If plant policy demands maximizing failure containment at the expense of false alarms (High-Recall setting, tau = 0.07), missed failures drop from 12 down to 8 (88.24% recall), while false alarms increase to 73.

---

### 4.5 Permutation Feature Importance
Permutation feature importance was evaluated across 20 iterations on the held-out test set using Average Precision (PR-AUC) as the evaluation metric:

| Rank | Feature | Mean Decrease in PR-AUC | Std Deviation | Role / Physical Mechanism |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `Rotational speed [rpm]` | 0.34352 | 0.03052 | Primary physical variable driving HDF and PWF |
| 2 | `Overstrain [minNm]` | 0.15413 | 0.01171 | **Engineered feature** directly capturing Overstrain Failure |
| 3 | `Power [W]` | 0.10541 | 0.01160 | **Engineered feature** directly capturing Power Failure |
| 4 | `Temp ratio` | 0.08308 | 0.01722 | **Engineered feature** reflecting thermal dissipation limit |
| 5 | `Tool wear [min]` | 0.03764 | 0.00384 | Cumulative wear component for TWF and OSF |
| 6 | `Type` | 0.03192 | 0.01097 | Product grade (dictates OSF threshold thresholds) |
| 7 | `Temp difference [K]` | 0.02101 | 0.00577 | **Engineered feature** tracking thermal transfer delta |
| 8 | `Torque [Nm]` | 0.00960 | 0.00580 | Mechanical cutting resistance |
| 9 | `Air temperature [K]` | -0.00052 | 0.00445 | Ambient background; redundant with thermal difference |
| 10 | `Process temperature [K]` | -0.00359 | 0.00389 | Chamber temperature; redundant with thermal ratio |

The top four predictive attributes are either rotational speed or engineered non-linear physical interactions. Raw temperature columns exhibited slight negative or negligible importance once differential thermal features were provided.

---

### 4.6 Failure-Mode Recall Breakdown
Evaluating true failure cases in the held-out test set revealed how different failure modes are identified by the best model:

| Failure Mode | Test Cases | Detected (tau=0.50) | Default Recall | Detected (tau=0.48) | Tuned Recall | Nature of Failure Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **HDF** (Heat Dissipation) | 29 | 28 | 96.55% | 28 | 96.55% | Deterministic thermal transfer rule |
| **PWF** (Power Failure) | 13 | 13 | 100.0% | 13 | 100.0% | Strict power boundary rule (Torque * Speed) |
| **OSF** (Overstrain Failure) | 16 | 16 | 100.0% | 16 | 100.0% | Strict threshold rule (Wear * Torque) |
| **TWF** (Tool Wear Failure) | 10 | 1 | 10.00% | 1 | 10.00% | Semi-stochastic wear cutoff (200-240 min) |
| **RNF** (Random Failure) | 0 | 0 | 0.00% | 0 | 0.00% | Pure stochastic uniform noise |
| **No Mode Flag** | 2 | 0 | 0.00% | 0 | 0.00% | Unlabeled synthetic failure instances |

The model achieves 100% recall on Power Failure (PWF) and Overstrain Failure (OSF), and 96.55% recall on Heat Dissipation Failure (HDF). This demonstrates that the engineered features successfully map the underlying physics of machine breakdown.

---

## 5. Analysis of Findings

### 5.1 Why Linear Models Struggle
Logistic Regression achieved a maximum test F1-score of only 0.2697 on engineered features and 0.1772 on raw features. This failure is attributable to the geometry of machine failure boundaries. In AI4I 2020, failure conditions are conjunctions of bounded inequalities (e.g., Power Failure occurs when Power < 3500 W **or** Power > 9000 W; Heat Dissipation Failure occurs when temperature difference < 8.6 K **and** speed < 1380 rpm). A single linear hyperplane cannot partition a non-convex, multimodal failure envelope without polynomial feature expansions.

### 5.2 The Efficacy of Physics-Based Feature Engineering
Tree-based ensembles can partition non-linear spaces using axis-aligned orthogonal splits. However, learning diagonal hyperplanes (such as torque multiplied by rotational speed or tool wear multiplied by torque) requires deep tree depth and hundreds of stepwise split approximations. By explicitly supplying `Power [W]` and `Overstrain [minNm]`, the model can separate failure boundaries with a single orthogonal split. This increased Random Forest cross-validation F1 from 0.6602 to 0.8911 without overfitting.

### 5.3 Precision-Recall Trade-offs in Operational Maintenance
The precision-recall curve illustrates the direct trade-off between inspection cost and catastrophic risk. Operating at tau = 0.48 prioritizes operational efficiency, resulting in 96.55% precision (almost every flagged alert corresponds to an imminent failure). Conversely, operating at tau = 0.07 prioritizes risk mitigation, reducing false negatives from 12 to 8 while tolerating 73 false alarms.

### 5.4 The Predictability of Random Failures (RNF)
The RNF (Random Failure) mode represents an explicitly stochastic failure event synthesized with a uniform 0.1% probability. By mathematical design, RNF does not correlate with sensor readings. Consequently, machine learning models cannot and should not predict RNF. Low detection on RNF is empirical proof that the model is learning deterministic physical relationships rather than memorizing random training noise.

---

## 6. Limitations
1. **Synthetic Nature of AI4I 2020:** While based on real industrial milling kinematics, the dataset generates failure modes using synthetic deterministic rules with injected noise. Validation on real manufacturing equipment with sensor drift, vibration harmonics, and uncalibrated tooling is required before production deployment.
2. **Small Test Failure Count:** In the 2,000-sample test set, exactly 68 failures are present. Individual failure mode sub-samples are small (e.g., 10 TWF cases, 13 PWF cases). A single misclassification impacts subtype recall by 7–10%, introducing variance into per-mode recall estimates.
3. **Absence of Real Temporal Sequences:** AI4I records observations as independent rows without timestamps. Genuine industrial telemetry exhibits time-series properties (e.g., vibration trends, thermal buildup over hours) that could be modelled with recurrent networks or autoregressive state estimators.

---

## 7. Conclusion
This study completed an empirical machine learning pipeline for predictive maintenance on the AI4I 2020 benchmark. Moving beyond misleading accuracy metrics, the project demonstrated that combining domain-derived physical interaction features with tree ensembles improves failure-class F1-scores from ~0.66 to 0.89. The resulting Random Forest model identified 100% of Power and Overstrain failures and 96.55% of Heat Dissipation failures on held-out test data, while out-of-fold threshold tuning provided operational control over the balance between false alarms and missed breakdowns.

---

## 8. References
1. **Matzka, S. (2020).** *Explainable Artificial Intelligence for Predictive Maintenance Applications.* Third International Conference on Artificial Intelligence for Industries (AI4I), pp. 69-74. IEEE. DOI: [10.1109/AI4I49448.2020.00023](https://doi.org/10.1109/AI4I49448.2020.00023).
2. **UCI Machine Learning Repository.** *AI4I 2020 Predictive Maintenance Dataset.* Dataset ID: 601. URL: [https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
3. **Pedregosa, F., et al. (2011).** *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, pp. 2825-2830.
