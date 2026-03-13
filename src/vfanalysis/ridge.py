"""Ridge-line and ridge-regression helpers for power/clock analysis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from vfanalysis._shared import add_interval_midpoint, residual_diagnostics, sorted_core_ids


@dataclass(frozen=True)
class RidgeRegressionResult:
    """Diagnostic summary for an sklearn ridge regression fit.

    Attributes:
        predictors: Predictor column names used in the model.
        target: Modeled target column.
        alpha: Ridge regularization weight.
        coefficients: Per-predictor coefficients.
        intercept: Fitted intercept.
        r2: Coefficient of determination on the training data.
        residual_std: Sample standard deviation of residuals.
        residual_var: Sample variance of residuals.
        n_samples: Number of fitted rows.
    """

    predictors: tuple[str, ...]
    target: str
    alpha: float
    coefficients: dict[str, float]
    intercept: float
    r2: float
    residual_std: float
    residual_var: float
    n_samples: int


def compute_power_clock_ridge(
    df: pd.DataFrame,
    quantile: float = 0.95,
    bins: int = 30,
    power_min: float | None = None,
    clock_min: float | None = None,
) -> pd.DataFrame:
    """Compute per-core boost ridge lines over power bins.

    Args:
        df: Telemetry dataframe containing ``core``, ``power``, and ``clock``.
        quantile: Clock quantile extracted per power bin.
        bins: Number of equally spaced power edges to generate per core.
        power_min: Optional strict lower bound applied before binning.
        clock_min: Optional strict lower bound applied before binning.

    Returns:
        A dataframe with ``core``, ``power_mid``, and ``clock_quantile``.

    Assumptions:
        Each included core spans a non-zero power range and has at least
        ``bins`` rows available after filtering.
    """

    data = df.copy()
    if power_min is not None:
        data = data[data["power"] > power_min]
    if clock_min is not None:
        data = data[data["clock"] > clock_min]

    ridges: list[pd.DataFrame] = []

    for core in sorted_core_ids(data):
        core_df = data[data["core"] == core]
        if len(core_df) < bins:
            continue

        power_edges = np.linspace(
            float(core_df["power"].min()),
            float(core_df["power"].max()),
            bins,
        )
        core_df = core_df.copy()
        core_df["power_bin"] = pd.cut(core_df["power"], power_edges)

        ridge = (
            core_df.groupby("power_bin", observed=False)["clock"]
            .quantile(quantile)
            .reset_index(name="clock_quantile")
        )
        ridge = add_interval_midpoint(
            ridge,
            interval_col="power_bin",
            midpoint_col="power_mid",
        )
        ridge["core"] = core
        ridges.append(ridge[["core", "power_mid", "clock_quantile"]].dropna())

    if not ridges:
        return pd.DataFrame(columns=["core", "power_mid", "clock_quantile"])

    return pd.concat(ridges, ignore_index=True)


def fit_ridge_regression(
    df: pd.DataFrame,
    predictors: Sequence[str],
    target: str,
    alpha: float = 1.0,
) -> RidgeRegressionResult:
    """Fit a ridge regression model and return diagnostics.

    Args:
        df: Input dataframe containing predictors and target.
        predictors: Predictor column names.
        target: Target column name.
        alpha: Ridge regularization weight.

    Returns:
        Diagnostic summary for the fitted model.

    Assumptions:
        The requested columns are numeric after dropping missing values.
    """

    frame = df[list(predictors) + [target]].dropna()
    if frame.empty:
        raise ValueError("No rows available after dropping NaNs for ridge regression")

    x = frame[list(predictors)]
    y = frame[target]

    model = Ridge(alpha=alpha)
    model.fit(x, y)
    y_hat = model.predict(x)
    diagnostics = residual_diagnostics(y, y_hat)

    coefficients = {name: float(value) for name, value in zip(predictors, model.coef_, strict=True)}

    return RidgeRegressionResult(
        predictors=tuple(predictors),
        target=target,
        alpha=alpha,
        coefficients=coefficients,
        intercept=float(model.intercept_),
        r2=diagnostics.r2,
        residual_std=diagnostics.residual_std,
        residual_var=diagnostics.residual_var,
        n_samples=len(frame),
    )
