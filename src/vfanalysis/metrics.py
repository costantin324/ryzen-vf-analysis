"""Aggregate/statistical metrics for Ryzen V/F telemetry analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vfanalysis.ridge import compute_power_clock_ridge


_MIN_SLOPE_POINTS = 10


def _subset_core(df: pd.DataFrame, core: int | None) -> pd.DataFrame:
    return df if core is None else df[df["core"] == core]


def _median_ratio(df: pd.DataFrame, numerator: str, denominator: str) -> float:
    values = df[numerator] / df[denominator].replace(0, np.nan)
    return float(values.median())


def dv_df_slope(df: pd.DataFrame, core: int | None = None) -> float:
    """Estimate dV/dF slope via linear fit ``vid = a + b * clock``."""

    subset = _subset_core(df, core).dropna(subset=["clock", "vid"])
    if len(subset) < _MIN_SLOPE_POINTS:
        return float("nan")
    return float(np.polyfit(subset["clock"], subset["vid"], deg=1)[0])


def clock_per_power(df: pd.DataFrame, core: int | None = None) -> float:
    """Median frequency-per-watt efficiency metric."""

    subset = _subset_core(df, core).dropna(subset=["clock", "power"])
    return _median_ratio(subset, "clock", "power")


def silicon_efficiency(df: pd.DataFrame, core: int | None = None) -> float:
    """Median ``clock / vid^2`` efficiency proxy used in the exploratory notebook."""

    subset = _subset_core(df, core).dropna(subset=["clock", "vid"])
    values = subset["clock"] / (subset["vid"] ** 2)
    return float(values.median())


def effective_voltage_per_mhz(df: pd.DataFrame, core: int | None = None) -> float:
    """Median voltage required per effective MHz."""

    subset = _subset_core(df, core).dropna(subset=["vid", "eff_clock"])
    return _median_ratio(subset, "vid", "eff_clock")


def voltage_per_mhz_squared(df: pd.DataFrame, core: int | None = None) -> float:
    """Median ``vid / clock^2`` metric for high-frequency voltage curvature checks."""

    subset = _subset_core(df, core).dropna(subset=["vid", "clock"])
    values = subset["vid"] / (subset["clock"] ** 2).replace(0, np.nan)
    return float(values.median())


def boost_ridge(
    df: pd.DataFrame,
    quantile: float = 0.95,
    bins: int = 30,
    power_min: float = 10.0,
) -> pd.DataFrame:
    """Return per-core boost ridge lines (clock quantile over power bins)."""

    return compute_power_clock_ridge(
        df,
        quantile=quantile,
        bins=bins,
        power_min=power_min,
    )


def core_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-core summary metrics used across analysis and reporting."""

    rows: list[dict[str, float | int]] = []

    for core in sorted(df["core"].dropna().unique()):
        core_df = df[df["core"] == core]

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
                "median_temp": float(core_df["temp"].median()) if "temp" in core_df else float("nan"),
            }
        )

    return pd.DataFrame(rows).sort_values("core").reset_index(drop=True)
