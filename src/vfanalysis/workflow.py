"""High-level workflow helpers for notebooks and scripts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from vfanalysis.features import add_all_features
from vfanalysis.filters import FilterConfig, base_filter_config, filter_df, vf_filter_config
from vfanalysis.io import load_processed_dataset


@dataclass(frozen=True)
class PreparedTelemetryFrames:
    """Standard dataframe stages used throughout the analysis notebook.

    Attributes:
        raw: Dataset loaded from parquet files without derived features.
        features: ``raw`` with deterministic row-wise features added.
        base: Lightly filtered telemetry frame.
        vf: More selective V/F analysis frame.
    """

    raw: pd.DataFrame
    features: pd.DataFrame
    base: pd.DataFrame
    vf: pd.DataFrame


@dataclass(frozen=True)
class WorkloadDatasets:
    """Named workload subsets used in the notebook's modeling sections.

    Attributes:
        scalar: Low-to-mid power scalar workload subset.
        loaded: Higher-power scalar workload subset.
        avx: Heavy AVX-like workload subset.
    """

    scalar: pd.DataFrame
    loaded: pd.DataFrame
    avx: pd.DataFrame


def prepare_analysis_frames(
    df_raw: pd.DataFrame,
    *,
    base_cfg: FilterConfig | None = None,
    vf_cfg: FilterConfig | None = None,
) -> PreparedTelemetryFrames:
    """Apply the standard feature and filtering pipeline.

    Args:
        df_raw: Long-format per-core telemetry dataframe.
        base_cfg: Optional base cleanup thresholds. Defaults to
            :func:`vfanalysis.filters.base_filter_config`.
        vf_cfg: Optional V/F-focused thresholds. Defaults to
            :func:`vfanalysis.filters.vf_filter_config`.

    Returns:
        A dataclass containing the raw, feature-engineered, base-filtered, and
        V/F-filtered frames.

    Assumptions:
        ``df_raw`` contains the columns required by the feature and filter
        stages used in the default pipeline.
    """

    base_config = base_cfg if base_cfg is not None else base_filter_config()
    vf_config = vf_cfg if vf_cfg is not None else vf_filter_config()

    df_features = add_all_features(df_raw)
    df_base = filter_df(df_features, base_config)
    df_vf = filter_df(df_base, vf_config)

    return PreparedTelemetryFrames(
        raw=df_raw.copy(),
        features=df_features,
        base=df_base,
        vf=df_vf,
    )


def load_analysis_frames(
    dataset_dir: Path | None = None,
    *,
    pattern: str = "*.parquet",
    base_cfg: FilterConfig | None = None,
    vf_cfg: FilterConfig | None = None,
) -> PreparedTelemetryFrames:
    """Load parquet telemetry data and run the standard notebook pipeline.

    Args:
        dataset_dir: Optional parquet directory override.
        pattern: Glob pattern selecting parquet files to load.
        base_cfg: Optional base cleanup thresholds.
        vf_cfg: Optional V/F-focused thresholds.

    Returns:
        Prepared telemetry frames for notebook and script use.

    Assumptions:
        The selected dataset directory contains processed parquet files with the
        schema produced by :func:`vfanalysis.io.build_parquet_dataset`.
    """

    df_raw = load_processed_dataset(dataset_dir=dataset_dir, pattern=pattern)
    return prepare_analysis_frames(df_raw, base_cfg=base_cfg, vf_cfg=vf_cfg)


def select_workload_dataset(
    df: pd.DataFrame,
    *,
    power_min: float | None = None,
    power_max: float | None = None,
    clock_min: float | None = None,
    clock_max: float | None = None,
    eff_ratio_min: float | None = None,
    eff_ratio_max: float | None = None,
    vid_min: float | None = None,
    vid_max: float | None = None,
    temp_max: float | None = None,
) -> pd.DataFrame:
    """Apply inclusive notebook-style workload bounds.

    Args:
        df: Input telemetry dataframe.
        power_min: Minimum inclusive power threshold in watts.
        power_max: Maximum inclusive power threshold in watts.
        clock_min: Minimum inclusive clock threshold in MHz.
        clock_max: Maximum inclusive clock threshold in MHz.
        eff_ratio_min: Minimum inclusive effective-clock ratio threshold.
        eff_ratio_max: Maximum inclusive effective-clock ratio threshold.
        vid_min: Minimum inclusive VID threshold in volts.
        vid_max: Maximum inclusive VID threshold in volts.
        temp_max: Maximum inclusive temperature threshold in degrees C.

    Returns:
        A filtered dataframe copy.

    Assumptions:
        The relevant columns exist whenever their matching thresholds are
        provided.
    """

    mask = pd.Series(True, index=df.index)

    if power_min is not None:
        mask &= df["power"] >= power_min
    if power_max is not None:
        mask &= df["power"] <= power_max
    if clock_min is not None:
        mask &= df["clock"] >= clock_min
    if clock_max is not None:
        mask &= df["clock"] <= clock_max
    if eff_ratio_min is not None:
        mask &= df["eff_ratio"] >= eff_ratio_min
    if eff_ratio_max is not None:
        mask &= df["eff_ratio"] <= eff_ratio_max
    if vid_min is not None:
        mask &= df["vid"] >= vid_min
    if vid_max is not None:
        mask &= df["vid"] <= vid_max
    if temp_max is not None and "temp" in df.columns:
        mask &= df["temp"] <= temp_max

    return df.loc[mask].copy()


def build_workload_datasets(df: pd.DataFrame) -> WorkloadDatasets:
    """Construct the standard scalar, loaded, and AVX notebook subsets.

    Args:
        df: Base-filtered telemetry dataframe.

    Returns:
        Named workload subsets matching the notebook's previous inline filters.

    Assumptions:
        ``df`` contains ``power``, ``clock``, and ``eff_ratio`` columns.
    """

    scalar = select_workload_dataset(
        df,
        power_min=2.0,
        power_max=12.0,
        clock_min=4200.0,
        eff_ratio_min=0.7,
    )
    loaded = select_workload_dataset(
        df,
        power_min=10.0,
        power_max=20.0,
        clock_min=4200.0,
        eff_ratio_min=0.7,
    )
    avx = select_workload_dataset(
        df,
        power_min=12.0,
        clock_min=2800.0,
        clock_max=4600.0,
        eff_ratio_min=0.7,
    )
    return WorkloadDatasets(scalar=scalar, loaded=loaded, avx=avx)


def build_core_observations(df_summary: pd.DataFrame) -> dict[str, dict[int, float] | str]:
    """Create a compact summary dictionary for notebook reporting.

    Args:
        df_summary: Per-core summary dataframe returned by
            :func:`vfanalysis.metrics.core_summary`.

    Returns:
        A dictionary containing high-level per-core observations.

    Assumptions:
        ``df_summary`` contains ``core``, ``max_eff_clock``, and
        ``clock_per_power`` columns.
    """

    max_eff_clock = df_summary.set_index("core")["max_eff_clock"].round(1).to_dict()
    median_clock_per_power = df_summary.set_index("core")["clock_per_power"].round(2).to_dict()
    return {
        "max_eff_clock_by_core_mhz": max_eff_clock,
        "median_clock_per_power_by_core": median_clock_per_power,
        "note": (
            "Per-core CO implications are exploratory estimates only. Use "
            "dedicated stability testing before treating inferred ranking as "
            "actionable."
        ),
    }


def smooth_co_estimate_curve(
    group: pd.DataFrame,
    *,
    freq_col: str = "freq_bin",
    estimate_col: str = "co_estimate",
    max_step_drop: float = 3.0,
) -> pd.DataFrame:
    """Apply the notebook's monotonic smoothing pass to CO estimates.

    Args:
        group: One core/temperature slice of the CO estimate dataset.
        freq_col: Column defining the increasing frequency order.
        estimate_col: Column holding the CO estimate to smooth.
        max_step_drop: Maximum allowed decrease between adjacent bins.

    Returns:
        A sorted dataframe copy with the smoothed estimate column.

    Assumptions:
        ``group`` contains the supplied columns and represents one logical
        curve that should be smoothed independently.
    """

    out = group.sort_values(freq_col).copy()
    values = out[estimate_col].to_numpy(dtype=float).copy()

    for index in range(1, len(values)):
        values[index] = max(values[index], values[index - 1] - max_step_drop)

    out[estimate_col] = values
    return out


def summarize_numeric_spans(
    df: pd.DataFrame,
    *,
    exclude_columns: Sequence[str] = ("core",),
) -> pd.DataFrame:
    """Summarize per-column ranges for a numeric dataframe.

    Args:
        df: Dataframe containing per-core numeric metrics.
        exclude_columns: Columns to exclude from the numeric span summary.

    Returns:
        A dataframe with min, max, span, mean, and median for each numeric column.

    Assumptions:
        Numeric columns contain comparable values across rows.
    """

    rows: list[dict[str, float | str]] = []
    excluded = set(exclude_columns)

    for column in df.columns:
        if column in excluded or not is_numeric_dtype(df[column]):
            continue

        series = df[column].dropna().astype(float)
        if series.empty:
            continue

        rows.append(
            {
                "metric": column,
                "min": float(series.min()),
                "max": float(series.max()),
                "span": float(series.max() - series.min()),
                "mean": float(series.mean()),
                "median": float(series.median()),
            }
        )

    return pd.DataFrame(rows)


def format_numeric_value(value: object) -> object:
    """Format numeric values for notebook display.

    Args:
        value: Scalar value to format.

    Returns:
        A formatted string for numeric values, or the original object for
        non-numeric inputs.

    Assumptions:
        Scientific notation is more readable for very small values.
    """

    if isinstance(value, bool) or not isinstance(value, int | float):
        return value

    numeric = float(value)
    if not math.isfinite(numeric):
        return str(numeric)
    if numeric == 0.0:
        return "0"

    magnitude = abs(numeric)
    if magnitude < 1e-3 or magnitude >= 1e4:
        return f"{numeric:.3e}"
    if magnitude < 1.0:
        return f"{numeric:.6f}"
    return f"{numeric:.4f}"


def format_numeric_table(
    df: pd.DataFrame,
    *,
    exclude_columns: Sequence[str] = ("core", "metric"),
) -> pd.DataFrame:
    """Format numeric dataframe columns for easier notebook comparison.

    Args:
        df: Input dataframe.
        exclude_columns: Columns that should remain unformatted.

    Returns:
        A copy of ``df`` with numeric columns converted to readable strings.

    Assumptions:
        The formatted dataframe is intended for display rather than further numeric
        computation.
    """

    out = df.copy()
    excluded = set(exclude_columns)

    for column in out.columns:
        if column in excluded or not is_numeric_dtype(out[column]):
            continue
        out[column] = out[column].map(format_numeric_value)

    return out
