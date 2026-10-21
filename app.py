"""MaintenAI Lab Streamlit entry point."""

from __future__ import annotations

from collections.abc import MutableMapping
from hashlib import sha256
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import streamlit as st

from core.data_loader import load_csv, validate_ai4i_schema
from core.features import add_engineered_features
from core.eda import (
    correlation_figure,
    distribution_feature_columns,
    distribution_figure,
    failure_count_figure,
    failure_class_statistics,
    failure_type_figure,
)
from core.ml_engine import ConfigurationError, ExperimentConfig, ExperimentResult
from core.diagnostics import classification_figures, cluster_figure, experiment_summary, pca_projection_figure, pca_variance_figure, regression_figures
from core.explainability import global_feature_importance, shap_global_importance, shap_status
from core.experiment_storage import ExperimentStorage, StorageError, dataframe_fingerprint
from core.model_lab import (
    TRAIN_TEST_SPLITS,
    algorithms_for,
    default_features_for,
    execute_experiment,
    make_config,
    metric_options,
    split_test_fraction,
    target_options,
)


st.set_page_config(page_title="MaintenAI Lab", page_icon="🛠️", layout="wide")


def _init_state() -> None:
    """Create the session keys used across pages."""
    st.session_state.setdefault("dataset", None)
    st.session_state.setdefault("dataset_name", None)
    st.session_state.setdefault("validation", None)
    st.session_state.setdefault("latest_experiment", None)
    st.session_state.setdefault("latest_metric", None)
    st.session_state.setdefault("latest_config", None)
    st.session_state.setdefault("latest_origin", None)
    st.session_state.setdefault("latest_dataset_fingerprint", None)
    st.session_state.setdefault("loaded_upload_signature", None)


def _clear_latest_result(state: MutableMapping[str, object] | None = None) -> None:
    """Remove outputs that no longer describe the current workflow state."""
    state = st.session_state if state is None else state
    for key in ("latest_experiment", "latest_metric", "latest_config", "latest_origin", "latest_dataset_fingerprint"):
        state[key] = None


def _replace_dataset_state(
    state: MutableMapping[str, object], dataframe: pd.DataFrame, name: str, report: object
) -> None:
    """Install a newly loaded dataset without retaining another dataset's run."""
    _clear_latest_result(state)
    state["dataset"] = dataframe
    state["dataset_name"] = name
    state["validation"] = report


def _modeling_is_available(df: pd.DataFrame | None) -> bool:
    report = st.session_state.get("validation")
    return df is not None and bool(report and getattr(report, "is_valid", False))


def _clear_if_configuration_changed(
    state: MutableMapping[str, object], config: ExperimentConfig, origin: str
) -> bool:
    """Clear a prior result only when its originating workflow was edited."""
    if state.get("latest_experiment") is None or state.get("latest_origin") != origin:
        return False
    if state.get("latest_config") == config:
        return False
    _clear_latest_result(state)
    return True


def _set_latest_result(
    result: ExperimentResult, selected_metric: str, dataframe: pd.DataFrame, origin: str
) -> None:
    st.session_state.latest_experiment = result
    st.session_state.latest_metric = selected_metric
    st.session_state.latest_config = result.config
    st.session_state.latest_origin = origin
    st.session_state.latest_dataset_fingerprint = dataframe_fingerprint(dataframe)


def _uploaded_dataset_signature(upload: object) -> str:
    """Identify uploaded CSV content without reloading it on every rerun."""
    try:
        content = upload.getvalue()  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError("The uploaded dataset must provide its file content.") from exc
    if not isinstance(content, bytes):
        raise TypeError("The uploaded dataset content must be bytes.")
    return sha256(content).hexdigest()


def _should_reload_uploaded_dataset(state: MutableMapping[str, object], upload: object) -> bool:
    """Return true only when an uploaded CSV's content differs from the loaded one."""
    signature = _uploaded_dataset_signature(upload)
    if state.get("loaded_upload_signature") == signature:
        return False
    state["loaded_upload_signature"] = signature
    return True


def _experiment_storage() -> ExperimentStorage:
    return ExperimentStorage(Path(__file__).parent / "data" / "experiments.db")


def _load_dataset_controls() -> None:
    st.sidebar.subheader("Dataset")
    source = st.sidebar.radio("Load from", ("Upload CSV", "Local CSV path"))

    if source == "Upload CSV":
        upload = st.sidebar.file_uploader("AI4I 2020 CSV", type=["csv"])
        if upload is not None and _should_reload_uploaded_dataset(st.session_state, upload):
            _store_dataset(upload, upload.name)
    else:
        st.session_state.loaded_upload_signature = None
        path = st.sidebar.text_input("CSV path", placeholder="C:/data/ai4i2020.csv")
        if st.sidebar.button("Load local CSV", use_container_width=True):
            if not path.strip():
                st.sidebar.error("Enter a path to a CSV file.")
            else:
                _store_dataset(path.strip(), path.strip())

    st.sidebar.subheader("Feature Engineering")
    st.sidebar.checkbox("Use engineered features", key="use_engineered_features")


def _store_dataset(source: object, name: str) -> None:
    result = load_csv(source)
    if result.errors:
        st.sidebar.error("Could not read the CSV. " + " ".join(result.errors))
        return

    report = validate_ai4i_schema(result.dataframe)
    _replace_dataset_state(st.session_state, result.dataframe, name, report)
    if report.is_valid:
        st.sidebar.success("Dataset loaded and validated.")
    else:
        st.sidebar.warning("Dataset loaded, but validation needs attention.")


def _dataset_status(df: pd.DataFrame | None) -> None:
    if df is None:
        st.info("Load the official AI4I 2020 CSV from the sidebar to begin.")
        return

    report = st.session_state.validation
    st.caption(f"Loaded: {st.session_state.dataset_name}")
    cols = st.columns(4)
    cols[0].metric("Rows", f"{len(df):,}")
    cols[1].metric("Columns", len(df.columns))
    cols[2].metric("Missing values", int(df.isna().sum().sum()))
    failure_count = pd.to_numeric(df["Machine failure"], errors="coerce").sum() if "Machine failure" in df else None
    cols[3].metric("Machine failures", f"{failure_count:,}" if failure_count is not None else "Unavailable")

    if report and not report.is_valid:
        st.error("This file is not ready for modelling. Resolve the validation issues below.")
        for issue in report.errors:
            st.error(issue)
    elif report:
        for warning in report.warnings:
            st.warning(warning)


def overview_page(df: pd.DataFrame | None) -> None:
    st.title("MaintenAI Lab")
    st.write("An interactive, leakage-aware experimentation workspace for the AI4I 2020 Predictive Maintenance dataset.")
    _dataset_status(df)
    if df is not None:
        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)


def eda_page(df: pd.DataFrame | None) -> None:
    st.title("Exploratory Data Analysis")
    _dataset_status(df)
    if df is None:
        return

    features = distribution_feature_columns(df)
    feature = st.selectbox("Distribution feature", features, index=0 if features else None)
    if feature:
        st.plotly_chart(distribution_figure(df, feature), use_container_width=True)

    left, right = st.columns(2)
    with left:
        class_statistics = failure_class_statistics(df)
        figure = failure_count_figure(df)
        if figure and class_statistics is not None:
            st.subheader("Class imbalance")
            st.dataframe(class_statistics, hide_index=True, use_container_width=True)
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("Failure counts require a 'Machine failure' column.")
    with right:
        figure = failure_type_figure(df)
        if figure:
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("Failure-type analysis appears when AI4I failure-mode columns are present.")

    figure = correlation_figure(df)
    if figure:
        st.subheader("Numeric correlations")
        st.warning(
            "Columns marked as target or failure subtype labels are outcome-derived. "
            "Their correlations are descriptive only and must not be treated as leakage-free predictive evidence."
        )
        st.plotly_chart(figure, use_container_width=True)


def _algorithm_parameters(task_type: str, algorithm: str) -> dict[str, object]:
    """Render only estimator parameters exposed by the current Model Lab choice."""
    if algorithm in {"logistic_regression", "ridge", "svm", "svr"}:
        return {"C" if algorithm in {"logistic_regression", "svm", "svr"} else "alpha": st.number_input(
            "Regularization strength" if algorithm != "ridge" else "Alpha", min_value=0.0001, value=1.0, key="model_lab_regularization"
        )}
    if algorithm in {"decision_tree", "decision_tree_regressor"}:
        depth = st.number_input("Maximum depth (0 = unlimited)", min_value=0, value=0, step=1, key="model_lab_max_depth")
        return {} if depth == 0 else {"max_depth": int(depth)}
    if algorithm in {"random_forest", "random_forest_regressor"}:
        trees = st.slider("Number of trees", 50, 500, 200, 25, key="model_lab_trees")
        depth = st.number_input("Maximum depth (0 = unlimited)", min_value=0, value=0, step=1, key="model_lab_forest_depth")
        params: dict[str, object] = {"n_estimators": trees}
        if depth:
            params["max_depth"] = int(depth)
        return params
    if algorithm == "knn":
        return {"n_neighbors": st.slider("Neighbors", 1, 50, 5, key="model_lab_neighbors")}
    if algorithm == "naive_bayes":
        return {"var_smoothing": st.number_input("Variance smoothing", min_value=1e-12, value=1e-9, format="%.1e", key="model_lab_smoothing")}
    if algorithm == "kmeans":
        return {"n_clusters": st.slider("Clusters", 2, 12, 3, key="model_lab_clusters")}
    if algorithm == "dbscan":
        return {
            "eps": st.number_input("Epsilon", min_value=0.01, value=0.5, step=0.05, key="model_lab_eps"),
            "min_samples": st.slider("Minimum samples", 2, 30, 5, key="model_lab_min_samples"),
        }
    if algorithm == "agglomerative":
        return {"n_clusters": st.slider("Clusters", 2, 12, 3, key="model_lab_agg_clusters")}
    return {}


def _calculate_live_metrics(result: ExperimentResult, preds: np.ndarray | None) -> dict[str, float]:
    metrics = dict(result.metrics)
    if result.truth is not None and preds is not None:
        metrics["accuracy"] = float(accuracy_score(result.truth, preds))
        metrics["precision_macro"] = float(precision_score(result.truth, preds, average="macro", zero_division=0))
        metrics["recall_macro"] = float(recall_score(result.truth, preds, average="macro", zero_division=0))
        metrics["f1_macro"] = float(f1_score(result.truth, preds, average="macro", zero_division=0))
        if 1 in result.truth:
            metrics["precision_failure"] = float(precision_score(result.truth, preds, pos_label=1, zero_division=0))
            metrics["recall_failure"] = float(recall_score(result.truth, preds, pos_label=1, zero_division=0))
            metrics["f1_failure"] = float(f1_score(result.truth, preds, pos_label=1, zero_division=0))
    return metrics


def _render_experiment_result(result: ExperimentResult, selected_metric: str | None = None, predictions: np.ndarray | None = None) -> None:
    """Render only data calculated by a completed engine run."""
    display_preds = predictions if predictions is not None else result.predictions
    display_metrics = _calculate_live_metrics(result, display_preds) if (result.config.task_type == "classification" and display_preds is not None) else result.metrics

    if selected_metric and selected_metric in display_metrics:
        st.metric(f"Selected metric: {selected_metric}", f"{display_metrics[selected_metric]:.4f}")
    elif selected_metric:
        st.warning(f"{selected_metric.replace('_', ' ').title()} is undefined for this completed run.")

    if result.config.task_type == "classification":
        cols = st.columns(4)
        if "f1_failure" in display_metrics:
            cols[0].metric("Failure F1", f"{display_metrics['f1_failure']:.4f}")
        if "precision_failure" in display_metrics:
            cols[1].metric("Failure Precision", f"{display_metrics['precision_failure']:.4f}")
        if "recall_failure" in display_metrics:
            cols[2].metric("Failure Recall", f"{display_metrics['recall_failure']:.4f}")
        if "pr_auc" in display_metrics:
            cols[3].metric("PR-AUC", f"{display_metrics['pr_auc']:.4f}")

    metrics = pd.DataFrame({"Metric": list(display_metrics), "Value": list(display_metrics.values())})
    st.dataframe(metrics, hide_index=True, use_container_width=True)
    if result.validation_scores:
        st.caption("Validation scores (each fitted on a training fold)")
        st.dataframe(pd.DataFrame({"Fold": range(1, len(result.validation_scores) + 1), "Score": result.validation_scores}), hide_index=True)
    if display_preds is not None and result.truth is not None:
        st.subheader("Held-out predictions")
        pred_dict = {"Actual": result.truth}
        if result.prediction_scores is not None:
            pred_dict["Probability"] = np.round(result.prediction_scores, 4)
        pred_dict["Predicted"] = display_preds
        st.dataframe(pd.DataFrame(pred_dict), hide_index=True, use_container_width=True)
    if result.labels is not None:
        st.subheader("Cluster assignments")
        st.dataframe(pd.Series(result.labels, name="Cluster").value_counts().rename_axis("Cluster").reset_index(name="Count"), hide_index=True)
    if result.explained_variance_ratio is not None:
        st.subheader("PCA explained variance")
        st.dataframe(pd.DataFrame({"Component": range(1, len(result.explained_variance_ratio) + 1), "Explained variance": result.explained_variance_ratio}), hide_index=True)
    for warning in result.warnings:
        st.warning(warning)


def _save_latest_experiment(df: pd.DataFrame | None) -> None:
    result = st.session_state.latest_experiment
    if result is None or df is None:
        st.warning("Run an experiment before saving it.")
        return
    if st.session_state.get("latest_dataset_fingerprint") != dataframe_fingerprint(df):
        _clear_latest_result()
        st.warning("The loaded dataset changed after this run. Run a new experiment before saving.")
        return
    try:
        record = _experiment_storage().save_result(result, st.session_state.dataset_name or "Unnamed dataset", df)
    except StorageError as exc:
        st.error(str(exc))
    else:
        st.success(f"Saved completed experiment {record.experiment_id} to SQLite.")


def model_lab_page(df: pd.DataFrame | None) -> None:
    st.title("Model Lab")
    _dataset_status(df)
    if not _modeling_is_available(df):
        if df is not None:
            st.error("Model Lab requires a dataset that passes AI4I schema validation.")
        return
    if st.button("Reset Parameters", key="model_lab_reset"):
        for key in [key for key in st.session_state if key.startswith("model_lab_") and key != "model_lab_reset"]:
            del st.session_state[key]
        st.rerun()

    task_type = st.selectbox("Experiment type", ["classification", "clustering", "pca"], format_func=str.title, key="model_lab_task")
    algorithm = st.selectbox("Algorithm", algorithms_for(task_type), format_func=lambda value: value.replace("_", " ").title(), key="model_lab_algorithm")
    supervised = task_type in {"classification", "regression"}
    target: str | None = None
    if supervised:
        targets = target_options(df, task_type)
        if not targets:
            st.error("No suitable target column is available for this task type.")
            return
        default_target = "Machine failure" if task_type == "classification" and "Machine failure" in targets else targets[0]
        target = st.selectbox("Target variable", targets, index=targets.index(default_target), key=f"model_lab_target_{task_type}")

    defaults = default_features_for(df, target)
    is_eng = bool(st.session_state.get("use_engineered_features", False))
    feature_key = f"model_lab_features_{task_type}_{target or 'unsupervised'}_{is_eng}"
    features = st.multiselect("Input features", list(df.columns), default=defaults, key=feature_key)
    if target == "Machine failure":
        st.caption("Machine failure safeguards: UDI, Product ID, and failure subtype labels cannot be used as predictors.")

    if supervised:
        split_label = st.selectbox("Train / test split", list(TRAIN_TEST_SPLITS), index=1, key="model_lab_split")
        split = split_test_fraction(split_label)
        validation_options = ["holdout", "kfold"] + (["stratified_kfold"] if task_type == "classification" else [])
        validation = st.selectbox("Validation", validation_options, format_func=lambda value: value.replace("_", " ").title(), key="model_lab_validation")
        folds = st.slider("Validation folds", 2, 10, 5, key="model_lab_folds") if validation != "holdout" else 5
    else:
        split, validation, folds = 0.2, "none", 5

    st.subheader("Preprocessing")
    pre_left, pre_right = st.columns(2)
    with pre_left:
        numeric_imputation = st.selectbox("Numeric missing values", ["median", "mean"], key="model_lab_numeric_imputation")
        scale_numeric = st.checkbox("Scale numeric features", value=True, key="model_lab_scale")
    with pre_right:
        categorical_imputation = st.selectbox("Categorical missing values", ["most_frequent", "constant"], key="model_lab_categorical_imputation")
        encode_categorical = st.checkbox("One-hot encode categorical features", value=True, key="model_lab_encode")

    class_weight, resampling = "none", "none"
    if task_type == "classification":
        st.subheader("Class imbalance")
        imbalance = st.radio("Handling", ["none", "balanced", "random_over"], horizontal=True, key="model_lab_imbalance")
        class_weight = "balanced" if imbalance == "balanced" else "none"
        resampling = "random_over" if imbalance == "random_over" else "none"
        if algorithm in {"knn", "naive_bayes"} and imbalance == "balanced":
            st.info("This estimator does not expose class weights; choose random oversampling or a weight-aware algorithm.")

    st.subheader("Algorithm parameters")
    hyperparameters = _algorithm_parameters(task_type, algorithm)
    pca_components = None
    if task_type == "pca":
        pca_components = st.slider("PCA components", 1, max(1, min(len(features), len(df))), min(2, max(1, min(len(features), len(df)))), key="model_lab_pca_components")
    selected_metric = st.selectbox("Primary evaluation metric", metric_options(task_type), format_func=lambda value: value.replace("_", " ").title(), key="model_lab_metric")
    random_state = st.number_input("Random seed", min_value=0, value=42, step=1, key="model_lab_seed")

    current_config = make_config(
        task_type=task_type, algorithm=algorithm, target=target, features=features, test_size=split,
        validation=validation, cv_folds=folds, random_state=int(random_state), class_weight=class_weight,
        resampling=resampling, numeric_imputation=numeric_imputation, categorical_imputation=categorical_imputation,
        scale_numeric=scale_numeric, encode_categorical=encode_categorical, hyperparameters=hyperparameters,
        pca_components=pca_components,
    )
    if _clear_if_configuration_changed(st.session_state, current_config, "model_lab"):
        st.info("The previous result was cleared because the experiment configuration changed.")

    run, save, compare = st.columns(3)
    with run:
        run_clicked = st.button("Run Experiment", type="primary", use_container_width=True)
    with save:
        if st.button("Save Experiment", use_container_width=True):
            _save_latest_experiment(df)
    with compare:
        compare_clicked = st.button("Compare Runs", use_container_width=True)
    if compare_clicked:
        st.info("Use Experiment History to compare durable saved experiments.")

    if run_clicked:
        _clear_latest_result()
        try:
            config = current_config
        except ConfigurationError as exc:
            st.error(str(exc))
        else:
            try:
                result = execute_experiment(df, config)
            except ConfigurationError as exc:
                st.error(str(exc))
                try:
                    record = _experiment_storage().save_failure(config, st.session_state.dataset_name or "Unnamed dataset", df, str(exc))
                except StorageError as storage_exc:
                    st.warning(f"Failed-run metadata could not be stored: {storage_exc}")
                else:
                    st.info(f"Saved failed experiment metadata {record.experiment_id} to SQLite.")
            else:
                _set_latest_result(result, selected_metric, df, "model_lab")
                st.success("Experiment completed using the loaded dataset.")
                _render_experiment_result(result, selected_metric)


def results_page(df: pd.DataFrame | None) -> None:
    st.title("Results")
    _dataset_status(df)
    result = st.session_state.latest_experiment
    if result is None:
        st.info("Run an experiment in Model Lab to see its real outputs here.")
        return
    st.caption(f"Latest run: {result.config.task_type.title()} · {result.config.algorithm.replace('_', ' ').title()}")

    preds = result.predictions
    if result.config.task_type == "classification":
        st.info("Accuracy can be misleading because the positive failure class represents only about 3.4% of observations.")
        if result.prediction_scores is not None and hasattr(st, "slider"):
            threshold = st.slider(
                "Decision threshold",
                min_value=0.05,
                max_value=0.95,
                value=0.50,
                step=0.01,
                key=f"results_threshold_slider_{result.config.algorithm}",
                help="Adjust classification decision threshold for probability-supported models.",
            )
            preds = (result.prediction_scores >= threshold).astype(int)

    _render_experiment_result(result, st.session_state.latest_metric, predictions=preds)
    st.subheader("Experiment configuration")
    st.dataframe(experiment_summary(result), hide_index=True, use_container_width=True)
    if result.config.task_type == "classification":
        for figure in classification_figures(result, predictions=preds).values():
            st.plotly_chart(figure, use_container_width=True)
        st.caption("Precision, recall, F1, ROC and precision–recall curves describe this held-out split only; rare classes can make them unstable.")
    elif result.config.task_type == "regression":
        for figure in regression_figures(result).values():
            st.plotly_chart(figure, use_container_width=True)
        st.caption("Residual patterns are descriptive diagnostics for the held-out rows, not evidence of causal relationships.")
    elif result.config.task_type == "clustering":
        figure = cluster_figure(result)
        if figure:
            st.plotly_chart(figure, use_container_width=True)
        st.caption("Cluster quality metrics depend on selected features and preprocessing; labels are not ground truth.")
    elif result.config.task_type == "pca":
        figure = pca_variance_figure(result)
        if figure:
            st.plotly_chart(figure, use_container_width=True)
        projection = pca_projection_figure(result)
        if projection:
            st.plotly_chart(projection, use_container_width=True)
        st.caption("Explained variance measures retained variation, not predictive quality.")


def explainability_page(df: pd.DataFrame | None) -> None:
    st.title("Explainability")
    _dataset_status(df)
    result = st.session_state.latest_experiment
    if result is None:
        st.info("Run a supervised experiment before requesting explanations.")
        return
    importance, status = global_feature_importance(result)
    if importance is None:
        st.info(status.reason)
    else:
        st.caption(status.reason)
        st.bar_chart(importance.set_index("Feature"))
        st.caption("Importance is model-specific and reflects association within this fitted model; it does not establish causation.")
    shap = shap_status(result)
    if shap.available:
        shap_importance, shap_result = shap_global_importance(result)
        if shap_importance is not None:
            st.subheader("SHAP global importance")
            st.bar_chart(shap_importance.set_index("Feature"))
        else:
            st.info(shap_result.reason)
    else:
        st.info(shap.reason)


def clustering_pca_page(df: pd.DataFrame | None) -> None:
    st.title("Clustering & PCA")
    _dataset_status(df)
    if not _modeling_is_available(df):
        if df is not None:
            st.error("Clustering & PCA requires a dataset that passes AI4I schema validation.")
        return
    task = st.radio("Unsupervised task", ["clustering", "pca"], horizontal=True, key="unsupervised_task")
    algorithm = st.selectbox("Algorithm", algorithms_for(task), format_func=lambda value: value.replace("_", " ").title(), key="unsupervised_algorithm")
    features = st.multiselect("Unsupervised features", list(df.columns), default=default_features_for(df, None), key="unsupervised_features")
    st.caption("AI4I outcome labels and identifiers are rejected from unsupervised fitting to avoid target influence.")
    hyperparameters = _algorithm_parameters(task, algorithm)
    components = st.slider("PCA components", 1, max(1, min(len(features), len(df))), min(2, max(1, min(len(features), len(df)))), key="unsupervised_components") if task == "pca" else None
    seed = st.number_input("Random seed", min_value=0, value=42, key="unsupervised_seed")
    current_config = make_config(
        task_type=task, algorithm=algorithm, target=None, features=features, random_state=int(seed),
        hyperparameters=hyperparameters, pca_components=components,
    )
    if _clear_if_configuration_changed(st.session_state, current_config, "unsupervised"):
        st.info("The previous result was cleared because the experiment configuration changed.")
    if st.button("Run unsupervised experiment", type="primary"):
        _clear_latest_result()
        try:
            config = current_config
            result = execute_experiment(df, config)
        except ConfigurationError as exc:
            st.error(str(exc))
        else:
            _set_latest_result(
                result, "silhouette" if task == "clustering" else "explained_variance_total", df, "unsupervised"
            )
            _render_experiment_result(result, st.session_state.latest_metric)
            figure = cluster_figure(result) if task == "clustering" else pca_variance_figure(result)
            if figure:
                st.plotly_chart(figure, use_container_width=True)
            if task == "pca":
                projection = pca_projection_figure(result)
                if projection:
                    st.plotly_chart(projection, use_container_width=True)


def experiment_history_page(df: pd.DataFrame | None) -> None:
    st.title("Experiment History")
    st.caption("SQLite-backed metadata only. Fitted model objects and predictions are never stored here.")
    try:
        storage = _experiment_storage()
        records = storage.list_records()
    except StorageError as exc:
        st.error(str(exc))
        return
    if not records:
        st.info("No saved experiments yet. Complete a run in Model Lab and choose Save Experiment.")
        return
    st.dataframe(pd.DataFrame([record.summary() for record in records]), hide_index=True, use_container_width=True)
    export_left, export_right = st.columns(2)
    with export_left:
        st.download_button("Export history as CSV", storage.export_csv(records), "maintenai-experiments.csv", "text/csv")
    with export_right:
        st.download_button("Export history as JSON", storage.export_json(records), "maintenai-experiments.json", "application/json")

    options = {f"{record.created_at} · {record.algorithm} · {record.experiment_id}": record.experiment_id for record in records}
    selected_labels = st.multiselect("Experiments to compare", list(options), key="history_compare")
    if st.button("Compare selected experiments"):
        comparison = storage.compare([options[label] for label in selected_labels])
        if comparison.records:
            st.dataframe(pd.DataFrame([record.summary() for record in comparison.records]), hide_index=True, use_container_width=True)
        if comparison.compatible:
            st.success("Selected runs have compatible task, dataset, target, validation, split, and metric setup.")
        for warning in comparison.warnings:
            st.warning(warning)

    st.subheader("Delete an experiment")
    delete_label = st.selectbox("Saved experiment", list(options), key="history_delete")
    confirmed = st.checkbox("I understand this permanently deletes the saved metadata record.", key="history_delete_confirm")
    if st.button("Delete selected experiment", disabled=not confirmed):
        try:
            deleted = storage.delete(options[delete_label])
        except StorageError as exc:
            st.error(str(exc))
        else:
            if deleted:
                st.success("Experiment metadata deleted.")
                st.rerun()
            else:
                st.warning("Experiment record was not found; refresh history.")


def pending_page(title: str, message: str, df: pd.DataFrame | None) -> None:
    st.title(title)
    _dataset_status(df)
    st.info(message)


def main() -> None:
    _init_state()
    _load_dataset_controls()
    page = st.sidebar.radio(
        "Workspace",
        ["Overview", "EDA", "Model Lab", "Results", "Explainability", "Clustering & PCA", "Experiment History"],
    )
    df = st.session_state.dataset
    if df is not None and st.session_state.get("use_engineered_features", False):
        modeling_df = add_engineered_features(df)
    else:
        modeling_df = df

    if page == "Overview":
        overview_page(df)
    elif page == "EDA":
        eda_page(df)
    elif page == "Model Lab":
        model_lab_page(modeling_df)
    elif page == "Results":
        results_page(modeling_df)
    elif page == "Explainability":
        explainability_page(modeling_df)
    elif page == "Clustering & PCA":
        clustering_pca_page(modeling_df)
    elif page == "Experiment History":
        experiment_history_page(df)


if __name__ == "__main__":
    main()
