"""Ridge-line and ridge-regression helpers for power/clock analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class RidgeRegressionResult:
    """Diagnostic summary for an sklearn ridge regression fit."""

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
    """Compute per-core boost ridge lines over power bins."""

    data = df.copy()
    if power_min is not None:
        data = data[data["power"] > power_min]
    if clock_min is not None:
        data = data[data["clock"] > clock_min]

    ridges: list[pd.DataFrame] = []

    for core in sorted(data["core"].dropna().unique()):
        core_df = data[data["core"] == core]
        if len(core_df) < bins:
            continue

        power_edges = np.linspace(core_df["power"].min(), core_df["power"].max(), bins)
        power_cats = pd.cut(core_df["power"], power_edges)

        ridge = core_df.groupby(power_cats, observed=False)["clock"].quantile(quantile)
        out = ridge.reset_index(name="clock_quantile")
        out["power_mid"] = out["power"].apply(lambda interval: interval.mid)
        out["core"] = core
        ridges.append(out[["core", "power_mid", "clock_quantile"]].dropna())

    if not ridges:
        return pd.DataFrame(columns=["core", "power_mid", "clock_quantile"])

    return pd.concat(ridges, ignore_index=True)


def fit_ridge_regression(
    df: pd.DataFrame,
    predictors: Sequence[str],
    target: str,
    alpha: float = 1.0,
) -> RidgeRegressionResult:
    """Fit a ridge regression model and return diagnostics."""

    frame = df[list(predictors) + [target]].dropna()
    if frame.empty:
        raise ValueError("No rows available after dropping NaNs for ridge regression")

    x = frame[list(predictors)]
    y = frame[target]

    model = Ridge(alpha=alpha)
    model.fit(x, y)
    y_hat = model.predict(x)
    residuals = y - y_hat

    return RidgeRegressionResult(
        predictors=tuple(predictors),
        target=target,
        alpha=alpha,
        coefficients={name: float(value) for name, value in zip(predictors, model.coef_, strict=True)},
        intercept=float(model.intercept_),
        r2=float(model.score(x, y)),
        residual_std=float(residuals.std(ddof=1)),
        residual_var=float(residuals.var(ddof=1)),
        n_samples=len(frame),
    )
