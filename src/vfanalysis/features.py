"""Deterministic row-wise feature engineering for V/F telemetry."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Validate that all required columns are present.

    Args:
        df: Input dataframe.
        columns: Column names required by the transformation.

    Returns:
        None.

    Assumptions:
        ``df`` is a pandas dataframe with string column labels.
    """

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series while treating zero denominators as missing.

    Args:
        numerator: Numeric numerator series.
        denominator: Numeric denominator series.

    Returns:
        A float series with zero denominators replaced by ``NaN``.

    Assumptions:
        ``numerator`` and ``denominator`` are aligned pandas series.
    """

    valid = denominator.replace(0, np.nan)
    return numerator / valid


def add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add frequency-derived row features.

    Args:
        df: Telemetry dataframe containing ``clock`` and ``eff_clock``.

    Returns:
        A copy of ``df`` with ``eff_ratio`` added.

    Assumptions:
        ``clock`` is expressed in MHz and ``eff_clock`` is aligned to the same
        sampling interval.
    """

    _require_columns(df, ["clock", "eff_clock"])

    out = df.copy()
    out["eff_ratio"] = _safe_divide(out["eff_clock"], out["clock"])
    return out


def add_voltage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add voltage/frequency row features.

    Args:
        df: Telemetry dataframe containing ``vid`` and ``clock``.

    Returns:
        A copy of ``df`` with voltage-per-frequency features added.

    Assumptions:
        ``vid`` is measured in volts and ``clock`` is measured in MHz.
    """

    _require_columns(df, ["vid", "clock"])

    out = df.copy()
    out["voltage_per_freq"] = _safe_divide(out["vid"], out["clock"])
    out["voltage_per_mhz_squared"] = _safe_divide(out["vid"], out["clock"] ** 2)
    return out


def add_power_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add power efficiency row features.

    Args:
        df: Telemetry dataframe containing ``clock`` and ``power``.

    Returns:
        A copy of ``df`` with clock-per-power helper columns added.

    Assumptions:
        ``power`` is expressed in watts and sampled per row.
    """

    _require_columns(df, ["clock", "power"])

    out = df.copy()
    out["freq_per_w"] = _safe_divide(out["clock"], out["power"])
    out["power_per_mhz"] = _safe_divide(out["power"], out["clock"])
    return out


def add_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add combined voltage/frequency efficiency features.

    Args:
        df: Telemetry dataframe containing ``vid`` and ``eff_clock``.

    Returns:
        A copy of ``df`` with effective-voltage-per-MHz added.

    Assumptions:
        ``eff_clock`` represents the effective frequency for the same sample as
        ``vid``.
    """

    _require_columns(df, ["vid", "eff_clock"])

    out = df.copy()
    out["effective_voltage_per_mhz"] = _safe_divide(out["vid"], out["eff_clock"])
    return out


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all deterministic row-wise feature transformations.

    Args:
        df: Input telemetry dataframe.

    Returns:
        A copy of ``df`` with all standard derived features added.

    Assumptions:
        ``df`` satisfies the column requirements of every feature stage.
    """

    out = add_frequency_features(df)
    out = add_voltage_features(out)
    out = add_power_features(out)
    out = add_efficiency_features(out)
    return out
