# MaintenAI Lab — Concise Demo Sequence

1. Start the application with `\.venv\Scripts\python.exe -m streamlit run app.py`.
2. In the sidebar, load `data/ai4i2020.csv` through **Local CSV path** or upload a UCI copy.
3. Open **EDA** and show a sensor distribution, Machine failure percentages, failure-mode counts, and the correlation-label warning.
4. Open **Model Lab**:
   - Choose **Classification** and target **Machine failure**.
   - Confirm that default features exclude identifiers and failure labels.
   - Use a visible split/seed/validation setting and run an algorithm.
5. Open **Results** to show metrics and applicable diagnostic figures. Describe only the displayed measured values.
6. Open **Explainability** and show native feature importance or the honest unavailable message.
7. Open **Clustering & PCA**, run one unsupervised workflow, and show its real projection/variance output.
8. Return to **Model Lab**, choose **Save Experiment**, then open **Experiment History**.
9. Select two compatible saved runs, compare them, and export CSV or JSON. Demonstrate the deletion confirmation only if a disposable run was saved.
