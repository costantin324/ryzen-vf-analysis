"""V/F curve extraction and fitting helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from vfanalysis._shared import grouped_quantile_curve, residual_diagnostics, sorted_core_ids


@dataclass(frozen=True)
class PowerLawFitResult:
    """Diagnostics for ``f = a * (v - vt)^b`` V/F model fits.

    Attributes:
        core: Core identifier, or ``None`` for aggregate fits.
        a: Multiplicative power-law coefficient.
        vt: Threshold-voltage estimate.
        b: Curvature exponent.
        r2: Coefficient of determination on fitted points.
        residual_std: Sample standard deviation of residuals.
        residual_var: Sample variance of residuals.
        n_samples: Number of fitted V/F points.
    """

    core: int | None
    a: float
    vt: float
    b: float
    r2: float
    residual_std: float
    residual_var: float
    n_samples: int


def vf_power_law(voltage: np.ndarray, a: float, vt: float, b: float) -> np.ndarray:
    """Evaluate the notebook's power-law V/F model.

    Args:
        voltage: Input voltage values.
        a: Multiplicative coefficient.
        vt: Threshold-voltage shift.
        b: Curvature exponent.

    Returns:
        Predicted frequency values.

    Assumptions:
        ``voltage`` is expressed in volts and may include values near ``vt``.
    """

    shifted = np.clip(voltage - vt, a_min=1e-12, a_max=None)
    return a * np.power(shifted, b)


def extract_vf_curve(
    df: pd.DataFrame,
    core: int,
    bins: int = 40,
    quantile: float = 0.99,
    freq_col: str = "clock",
) -> pd.DataFrame:
    """Extract a quantile V/F curve for one core by voltage bins.

    Args:
        df: Telemetry dataframe containing ``core``, ``vid``, and ``freq_col``.
        core: Core identifier to isolate.
        bins: Number of voltage bins.
        quantile: Frequency quantile extracted per voltage bin.
        freq_col: Frequency column to summarize.

    Returns:
        A dataframe with ``vid_mid`` and the selected frequency column.

    Assumptions:
        The selected core contains enough rows to populate the requested bins.
    """

    core_df = df[df["core"] == core].copy()
    vf = grouped_quantile_curve(
        core_df,
        bin_column="vid",
        value_column=freq_col,
        bins=bins,
        quantile=quantile,
        bin_label="vid_bin",
        value_label=freq_col,
        midpoint_label="vid_mid",
    )
    return vf[["vid_mid", freq_col]].dropna().reset_index(drop=True)


def extract_aggregate_vf_curve(
    df: pd.DataFrame,
    bins: int = 40,
    quantile: float = 0.99,
    freq_col: str = "clock",
) -> pd.DataFrame:
    """Extract an aggregate V/F curve across all cores.

    Args:
        df: Telemetry dataframe containing ``vid`` and ``freq_col``.
        bins: Number of voltage bins.
        quantile: Frequency quantile extracted per voltage bin.
        freq_col: Frequency column to summarize.

    Returns:
        A dataframe with aggregate ``vid_mid`` and the selected frequency.

    Assumptions:
        Combining all cores is acceptable for the exploratory aggregate fit.
    """

    vf = grouped_quantile_curve(
        df,
        bin_column="vid",
        value_column=freq_col,
        bins=bins,
        quantile=quantile,
        bin_label="vid_bin",
        value_label=freq_col,
        midpoint_label="vid_mid",
    )
    return vf[["vid_mid", freq_col]].dropna().sort_values("vid_mid").reset_index(drop=True)


def fit_power_law_curve(
    vf_curve: pd.DataFrame,
    core: int | None = None,
    voltage_col: str = "vid_mid",
    freq_col: str = "clock",
    min_voltage: float | None = None,
) -> PowerLawFitResult:
    """Fit a notebook-style power law to an extracted V/F curve.

    Args:
        vf_curve: Extracted V/F curve dataframe.
        core: Optional core identifier for diagnostics.
        voltage_col: Voltage column name.
        freq_col: Frequency column name.
        min_voltage: Optional minimum voltage included in the fit.

    Returns:
        Power-law fit diagnostics.

    Assumptions:
        The filtered curve contains at least three points after any optional voltage
        cutoff is applied.
    """

    fit_df = vf_curve[[voltage_col, freq_col]].dropna()
    if min_voltage is not None:
        fit_df = fit_df[fit_df[voltage_col] > min_voltage]

    if len(fit_df) < 3:
        raise ValueError("Need at least three V/F points for power-law fit")

    voltage = fit_df[voltage_col].to_numpy(dtype=float)
    freq = fit_df[freq_col].to_numpy(dtype=float)
    params, _ = curve_fit(
        vf_power_law,
        voltage,
        freq,
        p0=[5000.0, 0.8, 0.5],
        maxfev=10_000,
    )

    fitted = vf_power_law(voltage, *params)
    diagnostics = residual_diagnostics(freq, fitted)

    return PowerLawFitResult(
        core=core,
        a=float(params[0]),
        vt=float(params[1]),
        b=float(params[2]),
        r2=diagnostics.r2,
        residual_std=diagnostics.residual_std,
        residual_var=diagnostics.residual_var,
        n_samples=len(fit_df),
    )


def fit_power_law_per_core(
    df: pd.DataFrame,
    bins: int = 40,
    quantile: float = 0.99,
    freq_col: str = "clock",
    min_voltage: float | None = None,
) -> pd.DataFrame:
    """Fit power-law V/F curves per core and return diagnostics.

    Args:
        df: Telemetry dataframe containing per-core V/F points.
        bins: Number of voltage bins per core.
        quantile: Frequency quantile extracted per voltage bin.
        freq_col: Frequency column to model.
        min_voltage: Optional minimum voltage included in each fit.

    Returns:
        One diagnostics row per successfully fitted core.

    Assumptions:
        Each included core has enough valid V/F bins for a stable fit.
    """

    rows: list[dict[str, float | int]] = []

    for core in sorted_core_ids(df):
        vf = extract_vf_curve(df, core=core, bins=bins, quantile=quantile, freq_col=freq_col)
        try:
            result = fit_power_law_curve(
                vf,
                core=int(core),
                freq_col=freq_col,
                min_voltage=min_voltage,
            )
        except ValueError:
            continue

        rows.append(
            {
                "core": int(core),
                "a": result.a,
                "vt": result.vt,
                "b": result.b,
                "r2": result.r2,
                "residual_std": result.residual_std,
                "residual_var": result.residual_var,
                "n_samples": result.n_samples,
            }
        )

    return pd.DataFrame(rows).sort_values("core").reset_index(drop=True)


def extract_min_voltage_curve(
    df: pd.DataFrame,
    core: int | None = None,
    bins: int = 40,
    quantile: float = 0.05,
) -> pd.DataFrame:
    """Extract the lower-voltage envelope over clock bins.

    Args:
        df: Telemetry dataframe containing ``clock`` and ``vid``.
        core: Optional core identifier to isolate before binning.
        bins: Number of clock bins.
        quantile: VID quantile extracted per clock bin.

    Returns:
        A dataframe with ``clock_mid`` and ``vid_min``.

    Assumptions:
        The selected rows contain enough clock spread for the requested binning.
    """

    curve_df = df.copy()
    if core is not None:
        curve_df = curve_df[curve_df["core"] == core].copy()

    curve = grouped_quantile_curve(
        curve_df,
        bin_column="clock",
        value_column="vid",
        bins=bins,
        quantile=quantile,
        bin_label="clock_bin",
        value_label="vid_min",
        midpoint_label="clock_mid",
    )
    return curve[["clock_mid", "vid_min"]].dropna().sort_values("clock_mid").reset_index(drop=True)


def extract_voltage_frontier(
    df: pd.DataFrame,
    core: int,
    bins: int = 35,
) -> pd.DataFrame:
    """Extract notebook-style voltage frontier statistics over clock bins.

    Args:
        df: Telemetry dataframe containing ``core``, ``clock``, ``vid``, and
            ``power``.
        core: Core identifier to isolate.
        bins: Number of clock bins.

    Returns:
        A dataframe containing median clock, voltage quantiles, median power,
        and sample counts per bin.

    Assumptions:
        The selected core has enough rows to support the requested binning.
    """

    frontier_df = df[df["core"] == core].copy()
    frontier_df["clock_bin"] = pd.cut(frontier_df["clock"], bins=bins)

    out = frontier_df.groupby("clock_bin", observed=False).agg(
        clock_mid=("clock", "median"),
        vid_q05=("vid", lambda series: series.quantile(0.05)),
        vid_q50=("vid", lambda series: series.quantile(0.50)),
        power_q50=("power", "median"),
        n=("vid", "size"),
    )
    return out.reset_index(drop=True).dropna()


def bootstrap_power_law_fit(
    df: pd.DataFrame,
    core: int,
    n_boot: int = 200,
    bins: int = 40,
    quantile: float = 0.99,
    freq_col: str = "clock",
    seed: int | None = None,
    min_voltage: float | None = None,
) -> pd.DataFrame:
    """Bootstrap notebook-style power-law fits for one core.

    Args:
        df: Telemetry dataframe containing the target core.
        core: Core identifier to resample.
        n_boot: Number of bootstrap resamples.
        bins: Number of voltage bins used per resample.
        quantile: Frequency quantile extracted per voltage bin.
        freq_col: Frequency column to model.
        seed: Optional deterministic seed.
        min_voltage: Optional minimum voltage included in each fit.

    Returns:
        A dataframe with one row per successful bootstrap fit.

    Assumptions:
        Resampling the selected core with replacement is a reasonable proxy for
        fit uncertainty.
    """

    results: list[dict[str, float | int]] = []
    core_df = df[df["core"] == core]

    for boot in range(n_boot):
        random_state = None if seed is None else seed + boot
        sample = core_df.sample(len(core_df), replace=True, random_state=random_state)
        vf = extract_vf_curve(sample, core=core, bins=bins, quantile=quantile, freq_col=freq_col)
        try:
            fit = fit_power_law_curve(
                vf,
                core=core,
                freq_col=freq_col,
                min_voltage=min_voltage,
            )
        except ValueError:
            continue

        results.append(
            {
                "core": core,
                "boot": boot,
                "a": fit.a,
                "vt": fit.vt,
                "b": fit.b,
                "r2": fit.r2,
            }
        )

    return pd.DataFrame(results)


def avx_transfer_check(df_avx: pd.DataFrame, fit_table: pd.DataFrame) -> pd.DataFrame:
    """Compare scalar-fit power-law models against AVX-like telemetry.

    Args:
        df_avx: AVX-like telemetry dataframe containing ``core``, ``vid``, and
            ``clock``.
        fit_table: Per-core power-law fit table with ``a``, ``vt``, and ``b``.

    Returns:
        A dataframe summarizing mean and quantile prediction deltas per core.

    Assumptions:
        ``fit_table`` rows correspond to the same core numbering used in
        ``df_avx``.
    """

    rows: list[dict[str, float | int]] = []

    for _, row in fit_table.iterrows():
        core = int(row["core"])
        avx = df_avx[df_avx["core"] == core]
        if avx.empty:
            continue

        pred = vf_power_law(
            avx["vid"].to_numpy(dtype=float),
            float(row["a"]),
            float(row["vt"]),
            float(row["b"]),
        )
        delta = avx["clock"].to_numpy(dtype=float) - pred
        rows.append(
            {
                "core": core,
                "avx_mean_delta_mhz": float(np.mean(delta)),
                "avx_p10_delta_mhz": float(np.quantile(delta, 0.10)),
                "avx_p90_delta_mhz": float(np.quantile(delta, 0.90)),
                "n_avx": int(len(avx)),
            }
        )

    return pd.DataFrame(rows)
