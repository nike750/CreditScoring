"""Feature engineering helpers for the Give Me Some Credit dataset."""
from __future__ import annotations

import pandas as pd

TARGET = "SeriousDlqin2yrs"


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a small set of transparent derived features.

    The original columns are retained. Ratios are protected against division by zero.
    """
    df = data.copy()

    if {"MonthlyIncome", "DebtRatio"}.issubset(df.columns):
        df["EstimatedMonthlyDebt"] = df["MonthlyIncome"] * df["DebtRatio"]

    delinquency_cols = [
        c for c in [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
        ] if c in df.columns
    ]
    if delinquency_cols:
        df["TotalPastDueEvents"] = df[delinquency_cols].fillna(0).sum(axis=1)

    # Avoid retaining an accidental duplicate of the unsecured-utilization feature.
    if (
        "RevolvingUtilizationRatio" in df.columns
        and "RevolvingUtilizationOfUnsecuredLines" in df.columns
        and df["RevolvingUtilizationRatio"].equals(
            df["RevolvingUtilizationOfUnsecuredLines"]
        )
    ):
        df = df.drop(columns=["RevolvingUtilizationRatio"])

    return df


def split_features_target(data: pd.DataFrame, target: str = TARGET):
    """Validate and split a dataframe into X and y."""
    if target not in data.columns:
        raise ValueError(f"Required target column {target!r} was not found.")
    X = data.drop(columns=[target])
    y = data[target].astype(int)
    return X, y
