from dataclasses import dataclass

import pandas as pd
import pytest

import app as application
from app import (
    _clear_latest_result,
    _clear_if_configuration_changed,
    _replace_dataset_state,
    _should_reload_uploaded_dataset,
    _uploaded_dataset_signature,
)
from core.data_loader import ValidationReport
from core.ml_engine import ExperimentConfig, run_supervised_experiment
from tests.test_ml_engine import supervised_frame


def _state() -> dict[str, object]:
    return {
        "latest_experiment": object(),
        "latest_metric": "accuracy",
        "latest_config": ExperimentConfig(
            task_type="classification", algorithm="logistic_regression", target="Machine failure",
            features=("Torque [Nm]",),
        ),
        "latest_origin": "model_lab",
        "latest_dataset_fingerprint": "old-fingerprint",
    }


def test_replacing_a_dataset_clears_all_latest_result_state() -> None:
    state = _state()
    dataframe = pd.DataFrame({"Machine failure": [0, 1]})
    _replace_dataset_state(state, dataframe, "replacement.csv", ValidationReport())
    assert state["dataset"] is dataframe
    assert state["dataset_name"] == "replacement.csv"
    assert state["latest_experiment"] is None
    assert state["latest_metric"] is None
    assert state["latest_config"] is None
    assert state["latest_dataset_fingerprint"] is None


def test_changing_the_originating_configuration_clears_a_stale_result() -> None:
    state = _state()
    changed = ExperimentConfig(
        task_type="classification", algorithm="decision_tree", target="Machine failure",
        features=("Torque [Nm]",),
    )
    assert _clear_if_configuration_changed(state, changed, "model_lab")
    assert state["latest_experiment"] is None


def test_visiting_another_workflow_does_not_discard_a_result() -> None:
    state = _state()
    unsupervised = ExperimentConfig(task_type="clustering", algorithm="kmeans", features=("Torque [Nm]",))
    assert not _clear_if_configuration_changed(state, unsupervised, "unsupervised")
    assert state["latest_experiment"] is not None


@dataclass
class _UploadedFile:
    content: bytes

    def getvalue(self) -> bytes:
        return self.content


def test_same_uploaded_csv_is_not_reloaded_on_streamlit_reruns() -> None:
    state: dict[str, object] = {}
    upload = _UploadedFile(b"first,csv\n1,2\n")
    assert _should_reload_uploaded_dataset(state, upload)
    assert not _should_reload_uploaded_dataset(state, upload)
    assert _should_reload_uploaded_dataset(state, _UploadedFile(b"replacement,csv\n3,4\n"))


def test_uploaded_dataset_signature_requires_bytes_content() -> None:
    with pytest.raises(TypeError, match="bytes"):
        _uploaded_dataset_signature(_UploadedFile("not-bytes"))  # type: ignore[arg-type]


class _SessionState(dict[str, object]):
    def __getattr__(self, key: str) -> object:
        return self[key]

    def __setattr__(self, key: str, value: object) -> None:
        self[key] = value


class _RenderedPages:
    """Small Streamlit double used to exercise page functions with real results."""

    def __init__(self, state: _SessionState) -> None:
        self.session_state = state
        self.figures: list[object] = []
        self.bar_charts: list[object] = []
        self.infos: list[str] = []

    def title(self, *_: object, **__: object) -> None:
        pass

    def caption(self, *_: object, **__: object) -> None:
        pass

    def columns(self, count: int) -> list["_RenderedPages"]:
        return [self] * count

    def metric(self, *_: object, **__: object) -> None:
        pass

    def warning(self, *_: object, **__: object) -> None:
        pass

    def error(self, *_: object, **__: object) -> None:
        pass

    def subheader(self, *_: object, **__: object) -> None:
        pass

    def dataframe(self, *_: object, **__: object) -> None:
        pass

    def plotly_chart(self, figure: object, **_: object) -> None:
        self.figures.append(figure)

    def bar_chart(self, chart: object, **_: object) -> None:
        self.bar_charts.append(chart)

    def info(self, message: str, **_: object) -> None:
        self.infos.append(message)

    def slider(self, *_: object, **__: object) -> float:
        return 0.50


def test_results_and_explainability_render_a_real_session_result(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = supervised_frame()
    result = run_supervised_experiment(
        frame,
        ExperimentConfig(
            task_type="classification", algorithm="logistic_regression", target="Machine failure",
            features=("Torque [Nm]", "Rotational speed [rpm]", "Type"),
        ),
    )
    state = _SessionState(
        dataset_name="synthetic-ai4i.csv", validation=ValidationReport(), latest_experiment=result,
        latest_metric="accuracy",
    )
    rendered = _RenderedPages(state)
    monkeypatch.setattr(application, "st", rendered)

    application.results_page(frame)
    application.explainability_page(frame)

    assert len(rendered.figures) >= 3  # confusion matrix, ROC, and precision-recall
    assert len(rendered.bar_charts) == 1
    assert state.latest_experiment is result


def test_result_pages_explain_missing_or_cleared_experiment_state(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _SessionState(dataset_name="synthetic-ai4i.csv", validation=ValidationReport(), latest_experiment=object())
    _clear_latest_result(state)
    rendered = _RenderedPages(state)
    monkeypatch.setattr(application, "st", rendered)
    frame = supervised_frame()

    application.results_page(frame)
    application.explainability_page(frame)

    assert any("Run an experiment" in message for message in rendered.infos)
