# MaintenAI Lab — 10-Minute Demonstration Outline

1. **Title and objective — 0:00–0:45**
   - What MaintenAI Lab solves and why leakage-aware predictive-maintenance experiments matter.
2. **Dataset and safeguards — 0:45–1:45**
   - AI4I fields, schema validation, identifier/outcome-label exclusion.
3. **EDA — 1:45–3:00**
   - Feature distribution, class imbalance, failure modes, correlation warning.
4. **Model Lab — 3:00–5:00**
   - Select classification target, safe features, split, validation, imbalance handling, seed, and algorithm.
5. **Real results — 5:00–6:30**
   - Run one experiment and show actual metrics, confusion matrix, ROC/precision–recall where available.
6. **Interpretation — 6:30–7:30**
   - Native feature importance and its model-specific, non-causal limitation; mention optional SHAP status.
7. **Unsupervised workflow — 7:30–8:30**
   - Run K-Means or PCA; show projection and explain why labels are excluded.
8. **Experiment history — 8:30–9:30**
   - Save actual run, restart-safe SQLite history, compatible comparison, CSV/JSON export.
9. **Limitations and close — 9:30–10:00**
   - Split dependence, rare-event uncertainty, benchmark-data caveat, and future model-artifact persistence.
