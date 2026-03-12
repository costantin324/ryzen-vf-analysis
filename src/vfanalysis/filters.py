"""Row-level filters for telemetry dataframes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FilterConfig:
    """Threshold configuration for row filtering."""

    eff_clock_min: float | None = None
    eff_clock_max: float | None = None
    clock_min: float | None = None
    clock_max: float | None = None
    eff_ratio_min: float | None = None
    eff_ratio_max: float | None = None
    power_min: float | None = None
    power_max: float | None = None
    vid_min: float | None = None
    vid_max: float | None = None
    temp_max: float | None = None


def filter_df(df: pd.DataFrame, cfg: FilterConfig) -> pd.DataFrame:
    """Return a filtered dataframe according to ``cfg`` thresholds."""

    mask = pd.Series(True, index=df.index)

    if cfg.eff_clock_min is not None:
        mask &= df["eff_clock"] > cfg.eff_clock_min
    if cfg.eff_clock_max is not None:
        mask &= df["eff_clock"] < cfg.eff_clock_max

    if cfg.clock_min is not None:
        mask &= df["clock"] > cfg.clock_min
    if cfg.clock_max is not None:
        mask &= df["clock"] < cfg.clock_max

    if cfg.eff_ratio_min is not None:
        mask &= df["eff_clock"] > cfg.eff_ratio_min * df["clock"]
    if cfg.eff_ratio_max is not None:
        mask &= df["eff_clock"] < cfg.eff_ratio_max * df["clock"]

    if cfg.power_min is not None:
        mask &= df["power"] > cfg.power_min
    if cfg.power_max is not None:
        mask &= df["power"] < cfg.power_max

    if cfg.vid_min is not None:
        mask &= df["vid"] > cfg.vid_min
    if cfg.vid_max is not None:
        mask &= df["vid"] < cfg.vid_max

    if cfg.temp_max is not None and "temp" in df.columns:
        mask &= df["temp"] < cfg.temp_max

    return df.loc[mask].copy()


def base_filter_config() -> FilterConfig:
    """Light cleanup filter matching prior exploratory defaults."""

    return FilterConfig(vid_min=0.8, power_min=1.0)


def vf_filter_config() -> FilterConfig:
    """V/F-focused filter configuration from exploratory notebook logic."""

    return FilterConfig(eff_clock_min=3000, eff_ratio_min=0.6, power_min=4.5, vid_min=1.0)


def boost_filter_config() -> FilterConfig:
    """Boost-focused filter configuration from exploratory notebook logic."""

    return FilterConfig(eff_clock_min=4000, power_min=8.0)
