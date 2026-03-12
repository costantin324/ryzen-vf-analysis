"""
Feature engineering utilities for CPU telemetry analysis.

This module adds deterministic derived metrics to telemetry dataframes.
Features are computed per-row and do not perform filtering or aggregation.

Design principles
-----------------
- Pure transformations
- No filtering or plotting
- No global state
- Safe for reuse in pipelines and notebooks
- Compatible with AI-assisted refactoring

Typical pipeline
----------------
df = load_data(...)
df = add_all_features(df)
df = filter_df(df, filter_config)
"""

from __future__ import annotations

import pandas as pd


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """
    Ensure required columns exist in dataframe.

    Raises
    ------
    ValueError
        If required columns are missing.
    """

    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ------------------------------------------------------------
# Frequency derived metrics
# ------------------------------------------------------------


def add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add frequency-related derived metrics.

    Features added
    --------------
    eff_ratio
        Ratio of effective clock to nominal clock.

    Useful for detecting:
    - boost ramp transitions
    - scheduler artifacts
    - parked cores
    """

    _require_columns(df, ["clock", "eff_clock"])

    df = df.copy()

    df["eff_ratio"] = df["eff_clock"] / df["clock"]

    return df


# ------------------------------------------------------------
# Voltage derived metrics
# ------------------------------------------------------------


def add_voltage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add voltage efficiency metrics.

    Features added
    --------------
    voltage_per_freq
        Voltage required per MHz.

    Useful for:
    - silicon quality comparison
    - per-core efficiency analysis
    """

    _require_columns(df, ["vid", "clock"])

    df = df.copy()

    df["voltage_per_freq"] = df["vid"] / df["clock"]

    return df


# ------------------------------------------------------------
# Power derived metrics
# ------------------------------------------------------------


def add_power_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add power efficiency metrics.

    Features added
    --------------
    freq_per_w
        Clock frequency achieved per watt.

    Useful for:
    - efficiency ranking of cores
    - boost envelope analysis
    """

    _require_columns(df, ["clock", "power"])

    df = df.copy()

    df["freq_per_w"] = df["clock"] / df["power"]

    return df


# ------------------------------------------------------------
# Extended efficiency metrics
# ------------------------------------------------------------


def add_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add extended efficiency metrics combining voltage, power, and frequency.

    Features added
    --------------
    effective_voltage_per_mhz
        Voltage required per effective MHz.

    power_per_mhz
        Power consumed per MHz.

    These metrics are useful for:
    - high-load silicon efficiency analysis
    - cooling efficiency investigation
    """

    _require_columns(df, ["vid", "eff_clock", "power", "clock"])

    df = df.copy()

    df["effective_voltage_per_mhz"] = df["vid"] / df["eff_clock"]
    df["power_per_mhz"] = df["power"] / df["clock"]

    return df


# ------------------------------------------------------------
# Master feature pipeline
# ------------------------------------------------------------


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all supported feature transformations.

    This is the standard entry point used by most analysis pipelines.

    Features added
    --------------
    eff_ratio
    voltage_per_freq
    freq_per_w
    effective_voltage_per_mhz
    power_per_mhz

    Returns
    -------
    DataFrame
        Copy of dataframe with additional feature columns.
    """

    df = add_frequency_features(df)
    df = add_voltage_features(df)
    df = add_power_features(df)
    df = add_efficiency_features(df)

    return df
