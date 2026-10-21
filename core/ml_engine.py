"""Leakage-aware, reusable scikit-learn experimentation engine.

This module is UI-agnostic.  It returns fitted objects and observed metrics only;
the Streamlit Model Lab can consume it in a later milestone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    calinski_harabasz_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from core.data_loader import FAILURE_TYPE_COLUMNS

TaskType = Literal["classification", "regression", "clustering", "pca"]
ValidationType = Literal["holdout", "kfold", "stratified_kfold", "none"]

IDENTIFIER_COLUMNS = frozenset({"UDI", "Product ID"})
OUTCOME_LABEL_COLUMNS = frozenset({"Machine failure", *FAILURE_TYPE_COLUMNS})


class ConfigurationError(ValueError):
    """Raised when an experiment would be invalid or leak data."""


@dataclass(frozen=True)
class ExperimentConfig:
    task_type: TaskType
    algorithm: str
    target: str | None = None
    features: tuple[str, ...] | None = None
    random_state: int = 42
    test_size: float = 0.2
    validation: ValidationType = "holdout"
    cv_folds: int = 5
    class_weight: Literal["none", "balanced"] = "none"
    resampling: Literal["none", "random_over"] = "none"
    numeric_imputation: Literal["median", "mean"] = "median"
    categorical_imputation: Literal["most_frequent", "constant"] = "most_frequent"
    scale_numeric: bool = True
    encode_categorical: bool = True
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    pca_components: int | float | None = None


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    model: Any
    metrics: dict[str, float]
    validation_scores: list[float] = field(default_factory=list)
    predictions: np.ndarray | None = None
    truth: np.ndarray | None = None
    transformed_data: np.ndarray | None = None
    labels: np.ndarray | None = None
    explained_variance_ratio: np.ndarray | None = None
    feature_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    prediction_scores: np.ndarray | None = None
    test_features: pd.DataFrame | None = None
    runtime_seconds: float | None = None


def default_feature_columns(df: pd.DataFrame, target: str | None = None) -> list[str]:
    """Return safe defaults, excluding identifiers and AI4I outcome-derived labels."""
    # Failure outcomes are post-outcome labels in AI4I.  Excluding them for
    # every workflow is safer than relying on callers to choose a particular
    # target type (for example, a regression experiment must not use them as
    # convenient but scientifically invalid predictors).
    excluded = set(IDENTIFIER_COLUMNS).union(OUTCOME_LABEL_COLUMNS)
    if target:
        excluded.add(target)
    return [column for column in df.columns if column not in excluded]


def build_preprocessor(features: pd.DataFrame, config: ExperimentConfig | None = None) -> ColumnTransformer:
    """Build preprocessing that learns statistics only when a pipeline is fitted."""
    config = config or ExperimentConfig(task_type="pca", algorithm="pca")
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy=config.numeric_imputation))]
        if config.scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        if not config.encode_categorical:
            raise ConfigurationError("Categorical features require one-hot encoding. Enable categorical encoding or remove them.")
        categorical_imputer = SimpleImputer(
            strategy=config.categorical_imputation,
            fill_value="Missing" if config.categorical_imputation == "constant" else None,
        )
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", categorical_imputer),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ConfigurationError("Choose at least one numeric or categorical feature.")
    return ColumnTransformer(transformers, remainder="drop")


def _estimator(config: ExperimentConfig) -> Any:
    params = dict(config.hyperparameters)
    seed = config.random_state
    weighted = "balanced" if config.class_weight == "balanced" else None
    if config.task_type == "classification":
        registry = {
            "logistic_regression": lambda: LogisticRegression(max_iter=1_000, random_state=seed, class_weight=weighted, **params),
            "decision_tree": lambda: DecisionTreeClassifier(random_state=seed, class_weight=weighted, **params),
            "random_forest": lambda: RandomForestClassifier(random_state=seed, class_weight=weighted, **{"n_estimators": 200, **params}),
            "knn": lambda: KNeighborsClassifier(**params),
            "svm": lambda: SVC(random_state=seed, class_weight=weighted, probability=True, **params),
            "naive_bayes": lambda: GaussianNB(**params),
        }
    elif config.task_type == "regression":
        registry = {
            "linear_regression": lambda: LinearRegression(**params),
            "ridge": lambda: Ridge(random_state=seed, **params),
            "decision_tree_regressor": lambda: DecisionTreeRegressor(random_state=seed, **params),
            "random_forest_regressor": lambda: RandomForestRegressor(random_state=seed, **{"n_estimators": 200, **params}),
            "svr": lambda: SVR(**params),
        }
    elif config.task_type == "clustering":
        registry = {
            "kmeans": lambda: KMeans(random_state=seed, n_init="auto", **params),
            "dbscan": lambda: DBSCAN(**params),
            "agglomerative": lambda: AgglomerativeClustering(**params),
        }
    else:
        raise ConfigurationError(f"No estimator is available for task '{config.task_type}'.")
    try:
        return registry[config.algorithm]()
    except KeyError as exc:
        raise ConfigurationError(f"'{config.algorithm}' is not supported for {config.task_type}.") from exc


def _validated_features(df: pd.DataFrame, config: ExperimentConfig, supervised: bool) -> list[str]:
    if supervised and not config.target:
        raise ConfigurationError("A target is required for supervised experiments.")
    if config.target and config.target not in df.columns:
        raise ConfigurationError(f"Target '{config.target}' is not in the loaded dataset.")
    features = list(config.features) if config.features is not None else default_feature_columns(df, config.target)
    if not features:
        raise ConfigurationError("Select at least one input feature.")
    unknown = sorted(set(features).difference(df.columns))
    if unknown:
        raise ConfigurationError("Unknown feature(s): " + ", ".join(unknown) + ".")
    forbidden = set(features).intersection(IDENTIFIER_COLUMNS)
    forbidden.update(set(features).intersection(OUTCOME_LABEL_COLUMNS))
    if config.target in features:
        forbidden.add(config.target)
    if forbidden:
        raise ConfigurationError("Data leakage or identifier feature detected: " + ", ".join(sorted(forbidden)) + ".")
    return features


def _oversample(X: pd.DataFrame, y: pd.Series, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    """Deterministically oversample minority classes; call only with a training fold."""
    counts = y.value_counts()
    if len(counts) < 2:
        return X, y
    max_count = int(counts.max())
    rng = np.random.RandomState(random_state)
    selections = [np.flatnonzero(y.to_numpy() == value) for value in counts.index]
    indices = np.concatenate([rng.choice(group, size=max_count, replace=len(group) < max_count) for group in selections])
    rng.shuffle(indices)
    return X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray, pipeline: Pipeline, X_test: pd.DataFrame) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_true.nunique() == 2 and 1 in y_true.values:
        metrics["precision_failure"] = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        metrics["recall_failure"] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        metrics["f1_failure"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    if y_true.nunique() == 2 and hasattr(pipeline, "predict_proba"):
        try:
            scores = pipeline.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
            metrics["pr_auc"] = float(average_precision_score(y_true, scores))
        except (ValueError, AttributeError):
            pass
    return metrics


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }
    if len(y_true) >= 2:
        metrics["r2"] = float(r2_score(y_true, y_pred))
    return metrics


def _validation_score(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, config: ExperimentConfig) -> list[float]:
    if config.validation == "none" or config.validation == "holdout":
        return []
    if config.cv_folds < 2:
        raise ConfigurationError("Cross-validation requires at least two folds.")
    if config.validation == "stratified_kfold":
        if config.task_type != "classification":
            raise ConfigurationError("Stratified K-fold is available only for classification.")
        if y.value_counts().min() < config.cv_folds:
            raise ConfigurationError("Each class needs at least as many rows as the requested Stratified K-fold count.")
        splitter = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        splits = splitter.split(X, y)
    else:
        if len(X) < config.cv_folds:
            raise ConfigurationError("K-fold count cannot exceed the number of training rows.")
        splitter = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        splits = splitter.split(X)
    scores: list[float] = []
    for fold, (train_idx, validation_idx) in enumerate(splits):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        if config.resampling == "random_over" and config.task_type == "classification":
            X_train, y_train = _oversample(X_train, y_train, config.random_state + fold)
        fitted = clone(pipeline).fit(X_train, y_train)
        prediction = fitted.predict(X.iloc[validation_idx])
        if config.task_type == "classification":
            if 1 in y.values:
                score = f1_score(y.iloc[validation_idx], prediction, pos_label=1, zero_division=0)
            else:
                score = f1_score(y.iloc[validation_idx], prediction, average="macro", zero_division=0)
        else:
            score = r2_score(y.iloc[validation_idx], prediction)
        scores.append(float(score))
    return scores


def run_supervised_experiment(df: pd.DataFrame, config: ExperimentConfig) -> ExperimentResult:
    """Fit a supervised experiment with a held-out test set and optional CV on training data."""
    started = perf_counter()
    if config.task_type not in {"classification", "regression"}:
        raise ConfigurationError("Use this function only for classification or regression.")
    if not 0 < config.test_size < 1:
        raise ConfigurationError("test_size must be between 0 and 1.")
    features = _validated_features(df, config, supervised=True)
    X, y = df[features].copy(), df[config.target].copy()  # type: ignore[index]
    valid_target = y.notna()
    X, y = X.loc[valid_target], y.loc[valid_target]
    if len(X) < 4:
        raise ConfigurationError("At least four rows with a target value are required.")
    if config.task_type == "classification" and y.nunique() < 2:
        raise ConfigurationError("Classification requires at least two target classes.")
    warnings: list[str] = []
    if config.class_weight == "balanced" and config.algorithm in {"knn", "naive_bayes"}:
        warnings.append("This estimator does not support class weights; choose random oversampling or a weight-aware estimator.")
    stratify = None
    if config.task_type == "classification" and y.value_counts().min() >= 2:
        stratify = y
    elif config.task_type == "classification":
        warnings.append("A rare class prevented stratified holdout splitting; the holdout split is unstratified.")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state, stratify=stratify
        )
    except ValueError as exc:
        raise ConfigurationError(f"Unable to create the requested holdout split: {exc}") from exc
    pipeline = Pipeline([("preprocessor", build_preprocessor(X, config)), ("model", _estimator(config))])
    validation_scores = _validation_score(pipeline, X_train, y_train, config)
    if config.resampling == "random_over" and config.task_type == "classification":
        X_train, y_train = _oversample(X_train, y_train, config.random_state)
    try:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"Model fitting failed: {exc}") from exc
    metrics = _classification_metrics(y_test, predictions, pipeline, X_test) if config.task_type == "classification" else _regression_metrics(y_test, predictions)
    scores = None
    if config.task_type == "classification" and y_test.nunique() == 2 and hasattr(pipeline, "predict_proba"):
        try:
            scores = pipeline.predict_proba(X_test)[:, 1]
        except ValueError:
            pass
    return ExperimentResult(
        config, pipeline, metrics, validation_scores, predictions, y_test.to_numpy(), feature_columns=features,
        warnings=warnings, prediction_scores=scores, test_features=X_test.copy(), runtime_seconds=perf_counter() - started,
    )


def run_clustering_experiment(df: pd.DataFrame, config: ExperimentConfig) -> ExperimentResult:
    """Fit a clustering estimator on preprocessed selected features."""
    started = perf_counter()
    if config.task_type != "clustering":
        raise ConfigurationError("Use this function only for clustering.")
    features = _validated_features(df, config, supervised=False)
    X = df[features].copy()
    preprocessing = Pipeline([("preprocessor", build_preprocessor(X, config))])
    try:
        transformed = preprocessing.fit_transform(X)
        model = _estimator(config)
        labels = model.fit_predict(transformed)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"Clustering failed: {exc}") from exc
    non_noise = labels != -1
    distinct = np.unique(labels[non_noise])
    metrics: dict[str, float] = {"clusters": float(len(distinct)), "noise_points": float(np.sum(labels == -1))}
    if len(distinct) >= 2 and np.sum(non_noise) > len(distinct):
        metrics["silhouette"] = float(silhouette_score(transformed[non_noise], labels[non_noise]))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(transformed[non_noise], labels[non_noise]))
    return ExperimentResult(config, Pipeline([("preprocessor", preprocessing), ("model", model)]), metrics, transformed_data=transformed, labels=labels, feature_columns=features, runtime_seconds=perf_counter() - started)


def run_pca_experiment(df: pd.DataFrame, config: ExperimentConfig) -> ExperimentResult:
    """Fit PCA after leakage-safe preprocessing and return real variance ratios."""
    started = perf_counter()
    if config.task_type != "pca":
        raise ConfigurationError("Use this function only for PCA.")
    features = _validated_features(df, config, supervised=False)
    X = df[features].copy()
    preprocessing = build_preprocessor(X, config)
    try:
        transformed = preprocessing.fit_transform(X)
        pca = PCA(n_components=config.pca_components, random_state=config.random_state, **config.hyperparameters)
        components = pca.fit_transform(transformed)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"PCA failed: {exc}") from exc
    ratios = pca.explained_variance_ratio_
    metrics = {"components": float(pca.n_components_), "explained_variance_total": float(ratios.sum())}
    return ExperimentResult(config, Pipeline([("preprocessor", preprocessing), ("pca", pca)]), metrics, transformed_data=components, explained_variance_ratio=ratios, feature_columns=features, runtime_seconds=perf_counter() - started)
