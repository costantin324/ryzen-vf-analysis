"""Deterministic row-wise feature engineering for V/F telemetry."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid = denominator.replace(0, np.nan)
    return numerator / valid


def add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add frequency-derived row features."""

    _require_columns(df, ["clock", "eff_clock"])

    out = df.copy()
    out["eff_ratio"] = _safe_divide(out["eff_clock"], out["clock"])
    return out


def add_voltage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add voltage/frequency row features."""

    _require_columns(df, ["vid", "clock"])

    out = df.copy()
    out["voltage_per_freq"] = _safe_divide(out["vid"], out["clock"])
    out["voltage_per_mhz_squared"] = _safe_divide(out["vid"], out["clock"] ** 2)
    return out


def add_power_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add power efficiency row features."""

    _require_columns(df, ["clock", "power"])

    out = df.copy()
    out["freq_per_w"] = _safe_divide(out["clock"], out["power"])
    out["power_per_mhz"] = _safe_divide(out["power"], out["clock"])
    return out


def add_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add combined voltage/frequency/power row features."""

    _require_columns(df, ["vid", "eff_clock"])

    out = df.copy()
    out["effective_voltage_per_mhz"] = _safe_divide(out["vid"], out["eff_clock"])
    return out


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all deterministic row-wise feature transformations."""

    out = add_frequency_features(df)
    out = add_voltage_features(out)
    out = add_power_features(out)
    out = add_efficiency_features(out)
    return out
