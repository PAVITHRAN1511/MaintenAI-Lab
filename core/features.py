"""Domain-specific feature engineering for AI4I 2020 predictive maintenance dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

ENGINEERED_FEATURE_NAMES = (
    "Temp difference [K]",
    "Power [W]",
    "Overstrain [minNm]",
    "Temp ratio",
)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute physics-based features inspired by AI4I 2020 failure mechanics.

    Features engineered:
    - Temp difference [K]: Process temperature minus Air temperature (drives Heat Dissipation Failure).
    - Power [W]: Rotational power in Watts calculated as Torque * Speed * (2 * pi / 60) (drives Power Failure).
    - Overstrain [minNm]: Tool wear multiplied by Torque (drives Overstrain Failure).
    - Temp ratio: Ratio of Process temperature to Air temperature.

    All features are derived strictly from raw sensor inputs, preventing any label or identifier leakage.
    """
    data = df.copy()

    has_air_temp = "Air temperature [K]" in data.columns
    has_proc_temp = "Process temperature [K]" in data.columns
    has_speed = "Rotational speed [rpm]" in data.columns
    has_torque = "Torque [Nm]" in data.columns
    has_wear = "Tool wear [min]" in data.columns

    if has_proc_temp and has_air_temp:
        data["Temp difference [K]"] = data["Process temperature [K]"] - data["Air temperature [K]"]
        data["Temp ratio"] = data["Process temperature [K]"] / data["Air temperature [K]"]

    if has_torque and has_speed:
        data["Power [W]"] = data["Torque [Nm]"] * data["Rotational speed [rpm]"] * (2 * np.pi / 60)

    if has_wear and has_torque:
        data["Overstrain [minNm]"] = data["Tool wear [min]"] * data["Torque [Nm]"]

    return data
