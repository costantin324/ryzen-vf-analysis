"""Regression utilities extracted from exploratory notebook analyses."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from vfanalysis.vfcurve import extract_vf_curve, vf_power_law


@dataclass(frozen=True)
class RegressionResult:
    """Normalized diagnostic payload for regression fits."""

    model_name: str
    target: str
    predictors: tuple[str, ...]
    coefficients: dict[str, float]
    intercept: float
    r2: float
    residual_std: float
    residual_var: float
    n_samples: int
    core: int | None = None


def _coerce_frame(df: pd.DataFrame, predictors: Sequence[str], target: str) -> pd.DataFrame:
    frame = df[list(predictors) + [target]].dropna()
    if frame.empty:
        raise ValueError("No rows available for regression after dropping NaNs")
    return frame


def _regression_result_from_predictions(
    *,
    model_name: str,
    target: str,
    predictors: Sequence[str],
    coefficients: dict[str, float],
    intercept: float,
    y: pd.Series,
    y_hat: np.ndarray,
    core: int | None,
) -> RegressionResult:
    residuals = y - y_hat
    ss_res = float(np.sum((residuals.to_numpy(dtype=float)) ** 2))
    ss_tot = float(np.sum((y.to_numpy(dtype=float) - y.mean()) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1 - (ss_res / ss_tot)

    return RegressionResult(
        model_name=model_name,
        target=target,
        predictors=tuple(predictors),
        coefficients=coefficients,
        intercept=float(intercept),
        r2=r2,
        residual_std=float(np.std(residuals, ddof=1)),
        residual_var=float(np.var(residuals, ddof=1)),
        n_samples=len(y),
        core=core,
    )


def fit_linear_regression(
    df: pd.DataFrame,
    predictor: str = "vid",
    target: str = "clock",
    core: int | None = None,
) -> RegressionResult:
    """Fit linear regression matching notebook ``clock ~ vid`` analysis."""

    subset = df if core is None else df[df["core"] == core]
    frame = _coerce_frame(subset, [predictor], target)

    x = frame[[predictor]]
    y = frame[target]

    model = LinearRegression()
    model.fit(x, y)
    y_hat = model.predict(x)

    return _regression_result_from_predictions(
        model_name="linear",
        target=target,
        predictors=[predictor],
        coefficients={predictor: float(model.coef_[0])},
        intercept=float(model.intercept_),
        y=y,
        y_hat=y_hat,
        core=core,
    )


def fit_multivariate_regression(
    df: pd.DataFrame,
    predictors: Sequence[str] = ("vid", "power", "temp"),
    target: str = "clock",
    core: int | None = None,
) -> RegressionResult:
    """Fit multivariate linear regression for exploratory clock modeling."""

    subset = df if core is None else df[df["core"] == core]
    frame = _coerce_frame(subset, predictors, target)

    x = frame[list(predictors)]
    y = frame[target]

    model = LinearRegression()
    model.fit(x, y)
    y_hat = model.predict(x)

    coefficients = {
        name: float(value)
        for name, value in zip(predictors, model.coef_, strict=True)
    }

    return _regression_result_from_predictions(
        model_name="multivariate_linear",
        target=target,
        predictors=list(predictors),
        coefficients=coefficients,
        intercept=float(model.intercept_),
        y=y,
        y_hat=y_hat,
        core=core,
    )


def fit_power_law_regression(
    df: pd.DataFrame,
    target: str = "clock",
    core: int | None = None,
    bins: int = 40,
    quantile: float = 0.99,
) -> RegressionResult:
    """Fit notebook-style power-law V/F curve regression."""

    if core is None:
        raise ValueError("fit_power_law_regression requires a specific core")

    vf = extract_vf_curve(df, core=core, bins=bins, quantile=quantile, freq_col=target)
    fit_df = vf.dropna()

    voltage = fit_df["vid_mid"].to_numpy(dtype=float)
    freq = fit_df[target].to_numpy(dtype=float)

    params, _ = curve_fit(vf_power_law, voltage, freq, p0=[5000.0, 0.8, 0.5], maxfev=10_000)
    y_hat = vf_power_law(voltage, *params)

    return _regression_result_from_predictions(
        model_name="power_law",
        target=target,
        predictors=["vid_mid"],
        coefficients={"a": float(params[0]), "vt": float(params[1]), "b": float(params[2])},
        intercept=float("nan"),
        y=fit_df[target],
        y_hat=y_hat,
        core=core,
    )


def build_thermal_features(
    df: pd.DataFrame,
    smooth_window: int = 20,
    integral_window: int = 200,
    lags: Sequence[int] = (5, 20, 100),
) -> pd.DataFrame:
    """Build notebook-style thermal regression helper columns."""

    out = df.copy()
    out["temp_lag"] = out["cpu_temp"].shift(-5)
    out["ppt_integral"] = out["ppt"].rolling(integral_window).mean()
    out["ppt_smooth"] = out["ppt"].rolling(smooth_window).mean()
    out["temp_smooth"] = out["cpu_temp"].rolling(smooth_window).mean()

    for lag in lags:
        out[f"ppt_lag{lag}"] = out["ppt"].shift(lag)

    return out


def fit_ols_regression(
    df: pd.DataFrame,
    predictors: Sequence[str],
    target: str,
    core: int | None = None,
) -> RegressionResult:
    """Fit statsmodels OLS with notebook-style diagnostics."""

    subset = df if core is None else df[df["core"] == core]
    frame = _coerce_frame(subset, predictors, target)

    x = sm.add_constant(frame[list(predictors)], has_constant="add")
    y = frame[target]
    model = sm.OLS(y, x).fit()
    y_hat = model.predict(x)

    coefficients = {
        predictor: float(model.params[predictor])
        for predictor in predictors
        if predictor in model.params.index
    }

    return _regression_result_from_predictions(
        model_name="ols",
        target=target,
        predictors=list(predictors),
        coefficients=coefficients,
        intercept=float(model.params.get("const", np.nan)),
        y=y,
        y_hat=y_hat.to_numpy(dtype=float),
        core=core,
    )


def fit_sklearn_exploratory_regressions(
    df: pd.DataFrame,
    core: int | None = None,
) -> pd.DataFrame:
    """Run the notebook's sklearn exploratory regressions and return diagnostics."""

    results = [fit_linear_regression(df, predictor="vid", target="clock", core=core)]

    with suppress(ValueError):
        results.append(
            fit_multivariate_regression(
                df,
                predictors=("vid", "power", "temp"),
                target="clock",
                core=core,
            )
        )

    return pd.DataFrame([asdict(result) for result in results])


def per_core_fit_summary(
    df: pd.DataFrame,
    fit_fn: Callable[..., RegressionResult],
    **fit_kwargs: object,
) -> pd.DataFrame:
    """Apply a regression fit function per core and collect diagnostics."""

    rows: list[dict[str, object]] = []
    for core in sorted(df["core"].dropna().unique()):
        try:
            result = fit_fn(df=df, core=int(core), **fit_kwargs)
        except (RuntimeError, ValueError):
            continue
        rows.append(asdict(result))

    return pd.DataFrame(rows).sort_values("core").reset_index(drop=True)


def thermal_exploratory_regressions(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate notebook thermal regressions over smoothed/lagged PPT features."""

    features_df = build_thermal_features(df)

    specs = [
        ("cpu_temp", ("ppt",)),
        ("temp_lag", ("ppt",)),
        ("temp_smooth", ("ppt_smooth",)),
        ("temp_smooth", ("ppt_integral",)),
        ("temp_smooth", ("ppt_lag5",)),
        ("temp_smooth", ("ppt_lag20",)),
        ("temp_smooth", ("ppt_lag100",)),
    ]

    rows: list[dict[str, object]] = []
    for target, predictors in specs:
        try:
            result = fit_ols_regression(features_df, predictors=predictors, target=target)
        except ValueError:
            continue

        row = asdict(result)
        row["spec"] = f"{target} ~ {' + '.join(predictors)}"
        rows.append(row)

    return pd.DataFrame(rows)
