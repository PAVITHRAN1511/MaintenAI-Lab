"""UI-facing configuration helpers that delegate all fitting to ``ml_engine``."""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.ml_engine import (
    ConfigurationError,
    ExperimentConfig,
    ExperimentResult,
    default_feature_columns,
    run_clustering_experiment,
    run_pca_experiment,
    run_supervised_experiment,
)

ALGORITHMS = {
    "classification": ("logistic_regression", "decision_tree", "random_forest", "knn", "svm", "naive_bayes"),
    "regression": ("linear_regression", "ridge", "decision_tree_regressor", "random_forest_regressor", "svr"),
    "clustering": ("kmeans", "dbscan", "agglomerative"),
    "pca": ("pca",),
}

METRICS = {
    "classification": ("accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc", "f1_failure", "precision_failure", "recall_failure", "pr_auc"),
    "regression": ("r2", "rmse", "mae"),
    "clustering": ("silhouette", "calinski_harabasz", "clusters", "noise_points"),
    "pca": ("explained_variance_total", "components"),
}

TRAIN_TEST_SPLITS = {"70:30": 0.30, "80:20": 0.20, "90:10": 0.10}


def algorithms_for(task_type: str) -> tuple[str, ...]:
    return ALGORITHMS.get(task_type, ())


def metric_options(task_type: str) -> tuple[str, ...]:
    return METRICS.get(task_type, ())


def split_test_fraction(label: str) -> float:
    """Return the held-out test fraction for a Model Lab split label."""
    try:
        return TRAIN_TEST_SPLITS[label]
    except KeyError as exc:
        raise ConfigurationError(f"Unsupported train/test split '{label}'.") from exc


def target_options(df: pd.DataFrame, task_type: str) -> list[str]:
    """Return target choices appropriate to the requested supervised task."""
    candidates = [column for column in df.columns if column not in {"UDI", "Product ID"}]
    if task_type == "classification":
        return [column for column in candidates if 2 <= df[column].dropna().nunique() <= max(2, len(df) // 2)]
    if task_type == "regression":
        return [column for column in candidates if pd.api.types.is_numeric_dtype(df[column]) and df[column].dropna().nunique() > 2]
    return []


def default_features_for(df: pd.DataFrame, target: str | None) -> list[str]:
    return default_feature_columns(df, target)


def make_config(
    *, task_type: str, algorithm: str, target: str | None, features: list[str], test_size: float = 0.2,
    validation: str = "holdout", cv_folds: int = 5, random_state: int = 42,
    class_weight: str = "none", resampling: str = "none", numeric_imputation: str = "median",
    categorical_imputation: str = "most_frequent", scale_numeric: bool = True,
    encode_categorical: bool = True, hyperparameters: dict[str, Any] | None = None,
    pca_components: int | float | None = None,
) -> ExperimentConfig:
    if algorithm not in algorithms_for(task_type):
        raise ConfigurationError(f"'{algorithm}' is not available for {task_type}.")
    if task_type in {"clustering", "pca"}:
        target = None
        validation = "none"
    if validation == "stratified_kfold" and task_type != "classification":
        raise ConfigurationError("Stratified K-fold is available only for classification.")
    if resampling != "none" and task_type != "classification":
        raise ConfigurationError("Resampling is available only for classification.")
    return ExperimentConfig(
        task_type=task_type, algorithm=algorithm, target=target, features=tuple(features), test_size=test_size,
        validation=validation, cv_folds=cv_folds, random_state=random_state, class_weight=class_weight,
        resampling=resampling, numeric_imputation=numeric_imputation, categorical_imputation=categorical_imputation,
        scale_numeric=scale_numeric, encode_categorical=encode_categorical,
        hyperparameters=hyperparameters or {}, pca_components=pca_components,
    )


def execute_experiment(df: pd.DataFrame, config: ExperimentConfig) -> ExperimentResult:
    """Run one real experiment through the ML engine selected by its task type."""
    if config.task_type in {"classification", "regression"}:
        return run_supervised_experiment(df, config)
    if config.task_type == "clustering":
        return run_clustering_experiment(df, config)
    return run_pca_experiment(df, config)
