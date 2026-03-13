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

from vfanalysis._shared import residual_diagnostics, sorted_core_ids, subset_core
from vfanalysis.vfcurve import extract_vf_curve, vf_power_law


@dataclass(frozen=True)
class RegressionResult:
    """Normalized diagnostic payload for regression fits.

    Attributes:
        model_name: Short model identifier.
        target: Modeled target column.
        predictors: Predictor column names.
        coefficients: Per-predictor or named model coefficients.
        intercept: Fitted intercept, when defined.
        r2: Coefficient of determination on the training data.
        residual_std: Sample standard deviation of residuals.
        residual_var: Sample variance of residuals.
        n_samples: Number of fitted rows.
        core: Optional core identifier associated with the fit.
    """

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
    """Drop rows missing any requested predictor or target columns.

    Args:
        df: Input dataframe.
        predictors: Predictor column names.
        target: Target column name.

    Returns:
        A dataframe restricted to complete rows.

    Assumptions:
        The requested columns exist in ``df``.
    """

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
    """Build a normalized regression-result object from predictions.

    Args:
        model_name: Short model identifier.
        target: Modeled target column.
        predictors: Predictor column names.
        coefficients: Named coefficients to store in the result.
        intercept: Fitted intercept.
        y: Observed target values.
        y_hat: Predicted target values.
        core: Optional core identifier.

    Returns:
        A populated regression result.

    Assumptions:
        ``y`` and ``y_hat`` are aligned and numeric.
    """

    diagnostics = residual_diagnostics(y, y_hat)
    return RegressionResult(
        model_name=model_name,
        target=target,
        predictors=tuple(predictors),
        coefficients=coefficients,
        intercept=float(intercept),
        r2=diagnostics.r2,
        residual_std=diagnostics.residual_std,
        residual_var=diagnostics.residual_var,
        n_samples=len(y),
        core=core,
    )


def fit_linear_regression(
    df: pd.DataFrame,
    predictor: str = "vid",
    target: str = "clock",
    core: int | None = None,
) -> RegressionResult:
    """Fit the notebook's single-predictor linear regression.

    Args:
        df: Input telemetry dataframe.
        predictor: Predictor column name.
        target: Target column name.
        core: Optional core identifier to isolate.

    Returns:
        Diagnostic summary for the fitted model.

    Assumptions:
        ``predictor`` and ``target`` are numeric after dropping missing values.
    """

    subset = subset_core(df, core)
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
    """Fit the notebook's multivariate linear regression.

    Args:
        df: Input telemetry dataframe.
        predictors: Predictor column names.
        target: Target column name.
        core: Optional core identifier to isolate.

    Returns:
        Diagnostic summary for the fitted model.

    Assumptions:
        The requested columns are numeric after dropping missing values.
    """

    subset = subset_core(df, core)
    frame = _coerce_frame(subset, predictors, target)

    x = frame[list(predictors)]
    y = frame[target]

    model = LinearRegression()
    model.fit(x, y)
    y_hat = model.predict(x)

    coefficients = {name: float(value) for name, value in zip(predictors, model.coef_, strict=True)}

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
    """Fit the notebook's power-law V/F regression for one core.

    Args:
        df: Input telemetry dataframe.
        target: Frequency column to model.
        core: Core identifier to fit.
        bins: Number of voltage bins used to extract the curve.
        quantile: Frequency quantile extracted per voltage bin.

    Returns:
        Diagnostic summary for the fitted model.

    Assumptions:
        ``core`` is provided and the extracted V/F curve has enough points for
        a stable nonlinear fit.
    """

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
    """Build notebook-style thermal regression helper columns.

    Args:
        df: Input dataframe containing ``cpu_temp`` and ``ppt``.
        smooth_window: Rolling window used for smoothed series.
        integral_window: Rolling window used for the PPT integral proxy.
        lags: Lag offsets applied to ``ppt``.

    Returns:
        A dataframe copy with lagged and smoothed thermal features added.

    Assumptions:
        Rows are already ordered in time.
    """

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
    """Fit a statsmodels OLS regression with notebook-style diagnostics.

    Args:
        df: Input dataframe.
        predictors: Predictor column names.
        target: Target column name.
        core: Optional core identifier to isolate.

    Returns:
        Diagnostic summary for the fitted model.

    Assumptions:
        The requested columns are numeric after dropping missing values.
    """

    subset = subset_core(df, core)
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
    """Run the notebook's sklearn exploratory regressions.

    Args:
        df: Input telemetry dataframe.
        core: Optional core identifier to isolate.

    Returns:
        A dataframe of regression diagnostics for the requested scope.

    Assumptions:
        The dataframe contains the columns needed by the linear and optional
        multivariate fits.
    """

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


def collect_sklearn_regression_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Collect exploratory sklearn regressions for every core.

    Args:
        df: Input telemetry dataframe containing a ``core`` column.

    Returns:
        A concatenated diagnostics dataframe across all cores.

    Assumptions:
        Every core should be processed independently using the same fit specs.
    """

    rows = [fit_sklearn_exploratory_regressions(df, core=core) for core in sorted_core_ids(df)]
    return pd.concat(rows, ignore_index=True)


def per_core_fit_summary(
    df: pd.DataFrame,
    fit_fn: Callable[..., RegressionResult],
    **fit_kwargs: object,
) -> pd.DataFrame:
    """Apply a regression fit function per core and collect diagnostics.

    Args:
        df: Input telemetry dataframe containing a ``core`` column.
        fit_fn: Regression helper returning :class:`RegressionResult`.
        **fit_kwargs: Additional keyword arguments forwarded to ``fit_fn``.

    Returns:
        One diagnostics row per successfully fitted core.

    Assumptions:
        ``fit_fn`` accepts ``df`` and ``core`` keyword arguments.
    """

    rows: list[dict[str, object]] = []
    for core in sorted_core_ids(df):
        try:
            result = fit_fn(df=df, core=int(core), **fit_kwargs)
        except RuntimeError, ValueError:
            continue
        rows.append(asdict(result))

    return pd.DataFrame(rows).sort_values("core").reset_index(drop=True)


def thermal_exploratory_regressions(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate notebook thermal regressions over lagged PPT features.

    Args:
        df: Input dataframe containing the thermal columns used in the notebook.

    Returns:
        Diagnostics for each thermal regression specification that could be fit.

    Assumptions:
        Rows are already ordered in time and represent one continuous trace.
    """

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
