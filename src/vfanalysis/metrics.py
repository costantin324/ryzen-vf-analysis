"""Aggregate and statistical metrics for Ryzen V/F telemetry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vfanalysis._shared import sorted_core_ids, subset_core
from vfanalysis.ridge import compute_power_clock_ridge

_MIN_SLOPE_POINTS = 10


def _median_ratio(df: pd.DataFrame, numerator: str, denominator: str) -> float:
    """Compute the median of a ratio after zero-safe division.

    Args:
        df: Input dataframe.
        numerator: Numerator column name.
        denominator: Denominator column name.

    Returns:
        The median ratio as a float.

    Assumptions:
        Both columns are numeric and aligned row-wise.
    """

    values = df[numerator] / df[denominator].replace(0, np.nan)
    return float(values.median())


def dv_df_slope(df: pd.DataFrame, core: int | None = None) -> float:
    """Estimate the linear ``dV/dF`` slope for one core or the full frame.

    Args:
        df: Telemetry dataframe containing ``clock`` and ``vid``.
        core: Optional core identifier to isolate before fitting.

    Returns:
        The slope from ``vid = a + b * clock`` or ``NaN`` when there are too
        few valid points.

    Assumptions:
        ``clock`` and ``vid`` are numeric and approximately linear over the
        selected operating region.
    """

    subset = subset_core(df, core).dropna(subset=["clock", "vid"])
    if len(subset) < _MIN_SLOPE_POINTS:
        return float("nan")
    return float(np.polyfit(subset["clock"], subset["vid"], deg=1)[0])


def clock_per_power(df: pd.DataFrame, core: int | None = None) -> float:
    """Compute the median clock-per-power efficiency metric.

    Args:
        df: Telemetry dataframe containing ``clock`` and ``power``.
        core: Optional core identifier to isolate before aggregation.

    Returns:
        The median ``clock / power`` ratio.

    Assumptions:
        ``power`` is positive for rows of interest.
    """

    subset = subset_core(df, core).dropna(subset=["clock", "power"])
    return _median_ratio(subset, "clock", "power")


def silicon_efficiency(df: pd.DataFrame, core: int | None = None) -> float:
    """Compute the notebook's ``clock / vid^2`` silicon-efficiency proxy.

    Args:
        df: Telemetry dataframe containing ``clock`` and ``vid``.
        core: Optional core identifier to isolate before aggregation.

    Returns:
        The median silicon-efficiency proxy.

    Assumptions:
        ``vid`` is non-zero for the rows contributing to the metric.
    """

    subset = subset_core(df, core).dropna(subset=["clock", "vid"])
    values = subset["clock"] / (subset["vid"] ** 2)
    return float(values.median())


def effective_voltage_per_mhz(df: pd.DataFrame, core: int | None = None) -> float:
    """Compute the median voltage required per effective MHz.

    Args:
        df: Telemetry dataframe containing ``vid`` and ``eff_clock``.
        core: Optional core identifier to isolate before aggregation.

    Returns:
        The median ``vid / eff_clock`` ratio.

    Assumptions:
        ``eff_clock`` is non-zero for the rows contributing to the metric.
    """

    subset = subset_core(df, core).dropna(subset=["vid", "eff_clock"])
    return _median_ratio(subset, "vid", "eff_clock")


def voltage_per_mhz_squared(df: pd.DataFrame, core: int | None = None) -> float:
    """Compute the median ``vid / clock^2`` curvature proxy.

    Args:
        df: Telemetry dataframe containing ``vid`` and ``clock``.
        core: Optional core identifier to isolate before aggregation.

    Returns:
        The median curvature proxy.

    Assumptions:
        ``clock`` is non-zero for the rows contributing to the metric.
    """

    subset = subset_core(df, core).dropna(subset=["vid", "clock"])
    values = subset["vid"] / (subset["clock"] ** 2).replace(0, np.nan)
    return float(values.median())


def boost_ridge(
    df: pd.DataFrame,
    quantile: float = 0.95,
    bins: int = 30,
    power_min: float = 10.0,
) -> pd.DataFrame:
    """Return per-core boost ridge lines over power bins.

    Args:
        df: Telemetry dataframe containing ``core``, ``power``, and ``clock``.
        quantile: Clock quantile extracted in each power bin.
        bins: Number of equally spaced power edges per core.
        power_min: Optional minimum power threshold before binning.

    Returns:
        A dataframe with ``core``, ``power_mid``, and ``clock_quantile``.

    Assumptions:
        The selected rows provide enough samples per core to support binning.
    """

    return compute_power_clock_ridge(
        df,
        quantile=quantile,
        bins=bins,
        power_min=power_min,
    )


def core_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the standard per-core summary metrics used in the notebook.

    Args:
        df: Telemetry dataframe containing per-core V/F measurements.

    Returns:
        A dataframe with one summary row per core.

    Assumptions:
        ``df`` contains the columns referenced by the summary metrics, and each
        core has enough valid rows for meaningful medians.
    """

    rows: list[dict[str, float | int]] = []

    for core in sorted_core_ids(df):
        core_df = df[df["core"] == core]
        median_temp = float("nan")
        if "temp" in core_df.columns:
            median_temp = float(core_df["temp"].median())

        rows.append(
            {
                "core": int(core),
                "rows": int(len(core_df)),
                "dv_df_slope": dv_df_slope(core_df),
                "clock_per_power": clock_per_power(core_df),
                "silicon_efficiency": silicon_efficiency(core_df),
                "effective_voltage_per_mhz": effective_voltage_per_mhz(core_df),
                "voltage_per_mhz_squared": voltage_per_mhz_squared(core_df),
                "max_clock": float(core_df["clock"].max()),
                "max_eff_clock": float(core_df["eff_clock"].max()),
                "median_power": float(core_df["power"].median()),
                "median_temp": median_temp,
            }
        )

    return pd.DataFrame(rows).sort_values("core").reset_index(drop=True)
