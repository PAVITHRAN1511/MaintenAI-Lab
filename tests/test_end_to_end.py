from pathlib import Path

from core.data_loader import load_csv, validate_ai4i_schema
from core.experiment_storage import ExperimentStorage
from core.model_lab import default_features_for, execute_experiment, make_config


def test_ai4i_load_train_persist_export_and_compare(tmp_path) -> None:
    """Exercise the shipped AI4I data through the real user-facing core workflow."""
    dataset_path = Path(__file__).parents[1] / "data" / "ai4i2020.csv"
    loaded = load_csv(dataset_path)
    assert not loaded.errors and loaded.dataframe is not None
    assert validate_ai4i_schema(loaded.dataframe).is_valid

    frame = loaded.dataframe
    config = make_config(
        task_type="classification",
        algorithm="logistic_regression",
        target="Machine failure",
        features=default_features_for(frame, "Machine failure"),
        test_size=0.2,
        validation="holdout",
        random_state=42,
    )
    result = execute_experiment(frame, config)
    assert {"accuracy", "precision_macro", "recall_macro", "f1_macro"}.issubset(result.metrics)
    assert len(result.predictions) == len(result.truth) > 0

    database = tmp_path / "experiments.db"
    storage = ExperimentStorage(database)
    first = storage.save_result(result, dataset_path.name, frame, experiment_id="actual-ai4i-1")
    second = storage.save_result(result, dataset_path.name, frame, experiment_id="actual-ai4i-2")
    reopened = ExperimentStorage(database)
    assert reopened.get(first.experiment_id) is not None
    assert "actual-ai4i-1" in reopened.export_csv()
    assert "actual-ai4i-2" in reopened.export_json()
    assert reopened.compare([first.experiment_id, second.experiment_id]).compatible
