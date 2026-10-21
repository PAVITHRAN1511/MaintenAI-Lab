import pytest

from core.ml_engine import ConfigurationError
from core.model_lab import algorithms_for, default_features_for, execute_experiment, make_config, metric_options, split_test_fraction, target_options
from tests.test_ml_engine import supervised_frame


def test_model_lab_options_are_task_appropriate() -> None:
    frame = supervised_frame()
    assert "random_forest" in algorithms_for("classification")
    assert "ridge" in algorithms_for("regression")
    assert "silhouette" in metric_options("clustering")
    assert "Machine failure" in target_options(frame, "classification")
    assert "continuous_target" in target_options(frame, "regression")


def test_model_lab_configuration_executes_the_existing_engine() -> None:
    frame = supervised_frame()
    features = default_features_for(frame, "Machine failure")
    config = make_config(
        task_type="classification", algorithm="decision_tree", target="Machine failure", features=features,
        test_size=0.3, validation="stratified_kfold", cv_folds=3, random_state=99,
        numeric_imputation="mean", categorical_imputation="most_frequent", scale_numeric=True,
    )
    result = execute_experiment(frame, config)
    assert "accuracy" in result.metrics
    assert len(result.predictions) == len(result.truth)
    assert result.config.numeric_imputation == "mean"


def test_model_lab_rejects_incompatible_configuration() -> None:
    with pytest.raises(ConfigurationError, match="Stratified"):
        make_config(task_type="regression", algorithm="ridge", target="y", features=["x"], validation="stratified_kfold")


def test_train_test_split_labels_map_to_expected_test_fractions() -> None:
    assert split_test_fraction("70:30") == 0.30
    assert split_test_fraction("80:20") == 0.20
    assert split_test_fraction("90:10") == 0.10
    with pytest.raises(ConfigurationError, match="Unsupported"):
        split_test_fraction("80 / 20")
