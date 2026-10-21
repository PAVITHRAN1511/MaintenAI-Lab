"""Diagnostic data and Plotly charts built exclusively from real experiment outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from core.ml_engine import ExperimentResult


def experiment_summary(result: ExperimentResult) -> pd.DataFrame:
    config = result.config
    return pd.DataFrame({"Setting": ["Task", "Algorithm", "Target", "Features", "Validation", "Runtime (seconds)"], "Value": [config.task_type, config.algorithm, config.target or "—", ", ".join(result.feature_columns), config.validation.replace("_", " "), f"{result.runtime_seconds:.3f}" if result.runtime_seconds is not None else "Unavailable"]})


def classification_figures(
    result: ExperimentResult,
    predictions: np.ndarray | None = None,
    baseline_accuracy: float = 0.9661,
) -> dict[str, go.Figure]:
    if result.truth is None:
        return {}
    preds = predictions if predictions is not None else result.predictions
    if preds is None:
        return {}
    labels = np.unique(np.concatenate([result.truth, preds]))
    matrix = confusion_matrix(result.truth, preds, labels=labels)
    figures: dict[str, go.Figure] = {
        "confusion_matrix": px.imshow(
            matrix,
            x=[str(item) for item in labels],
            y=[str(item) for item in labels],
            text_auto=True,
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            title="Confusion matrix",
        )
    }
    if result.prediction_scores is not None and len(np.unique(result.truth)) == 2:
        pos_label = 1 if 1 in result.truth else labels[-1]
        fpr, tpr, _ = roc_curve(result.truth, result.prediction_scores, pos_label=pos_label)
        precision, recall, _ = precision_recall_curve(result.truth, result.prediction_scores, pos_label=pos_label)
        roc_fig = px.line(x=fpr, y=tpr, labels={"x": "False positive rate", "y": "True positive rate"}, title="ROC curve")
        roc_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "gray"})
        figures["roc"] = roc_fig

        pr_fig = px.line(x=recall, y=precision, labels={"x": "Recall", "y": "Precision"}, title="Precision–recall curve")
        figures["precision_recall"] = pr_fig

    # Evaluation metrics chart with trivial baseline reference line
    metric_keys = [
        "accuracy", "f1_macro", "f1_failure", "precision_failure", "recall_failure", "roc_auc", "pr_auc"
    ]
    available_metrics = {k: result.metrics[k] for k in metric_keys if k in result.metrics}
    if available_metrics:
        names = [k.replace("_", " ").title() for k in available_metrics]
        vals = list(available_metrics.values())
        chart = go.Figure()
        chart.add_bar(x=names, y=vals, text=[f"{v:.4f}" for v in vals], textposition="auto", name="Model Metric")
        chart.add_hline(
            y=baseline_accuracy,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Trivial Baseline ({baseline_accuracy:.4f})",
            annotation_position="top right",
        )
        chart.update_layout(
            title="Evaluation Metrics with Baseline Reference Line",
            yaxis={"range": [0, 1.05], "title": "Score"},
        )
        figures["metrics_chart"] = chart

    return figures


def regression_figures(result: ExperimentResult) -> dict[str, go.Figure]:
    if result.truth is None or result.predictions is None:
        return {}
    residuals = result.truth - result.predictions
    comparison = pd.DataFrame({"Actual": result.truth, "Predicted": result.predictions, "Residual": residuals})
    low, high = float(min(comparison["Actual"].min(), comparison["Predicted"].min())), float(max(comparison["Actual"].max(), comparison["Predicted"].max()))
    actual = px.scatter(comparison, x="Actual", y="Predicted", title="Actual versus predicted")
    actual.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line={"dash": "dash"})
    residual = px.scatter(comparison, x="Predicted", y="Residual", title="Residuals versus predicted")
    residual.add_hline(y=0, line_dash="dash")
    return {"actual_vs_predicted": actual, "residuals": residual}


def cluster_projection(result: ExperimentResult) -> pd.DataFrame | None:
    if result.transformed_data is None or result.labels is None or len(result.transformed_data) < 2:
        return None
    data = result.transformed_data
    coordinates = data[:, :2] if data.shape[1] == 2 else PCA(n_components=2, random_state=result.config.random_state).fit_transform(data)
    return pd.DataFrame({"Component 1": coordinates[:, 0], "Component 2": coordinates[:, 1], "Cluster": result.labels.astype(str)})


def cluster_figure(result: ExperimentResult) -> go.Figure | None:
    projection = cluster_projection(result)
    return px.scatter(projection, x="Component 1", y="Component 2", color="Cluster", title="Cluster projection") if projection is not None else None


def pca_variance_figure(result: ExperimentResult) -> go.Figure | None:
    if result.explained_variance_ratio is None:
        return None
    variance = pd.DataFrame({"Component": np.arange(1, len(result.explained_variance_ratio) + 1), "Explained variance": result.explained_variance_ratio, "Cumulative variance": np.cumsum(result.explained_variance_ratio)})
    figure = go.Figure()
    figure.add_bar(x=variance["Component"], y=variance["Explained variance"], name="Explained")
    figure.add_scatter(x=variance["Component"], y=variance["Cumulative variance"], name="Cumulative", yaxis="y2")
    figure.update_layout(title="PCA explained variance", yaxis={"title": "Per-component variance"}, yaxis2={"title": "Cumulative variance", "overlaying": "y", "side": "right", "range": [0, 1]})
    return figure


def pca_projection_figure(result: ExperimentResult) -> go.Figure | None:
    """Plot the actual first two fitted PCA component coordinates when available."""
    if result.transformed_data is None or result.transformed_data.shape[1] < 2:
        return None
    projection = pd.DataFrame({"PC1": result.transformed_data[:, 0], "PC2": result.transformed_data[:, 1]})
    return px.scatter(projection, x="PC1", y="PC2", title="PCA component projection")
