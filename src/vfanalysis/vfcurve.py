"""V/F curve extraction and fitting helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class PowerLawFitResult:
    """Diagnostics for ``f = a * (v - vt)^b`` V/F model fits."""

    core: int | None
    a: float
    vt: float
    b: float
    r2: float
    residual_std: float
    residual_var: float
    n_samples: int


def vf_power_law(voltage: np.ndarray, a: float, vt: float, b: float) -> np.ndarray:
    """Power-law model used in exploratory notebook cells."""

    shifted = np.clip(voltage - vt, a_min=1e-12, a_max=None)
    return a * np.power(shifted, b)


def extract_vf_curve(
    df: pd.DataFrame,
    core: int,
    bins: int = 40,
    quantile: float = 0.99,
    freq_col: str = "clock",
) -> pd.DataFrame:
    """Extract quantile V/F curve for one core by voltage bins."""

    core_df = df[df["core"] == core].copy()
    core_df["vid_bin"] = pd.cut(core_df["vid"], bins=bins)

    vf = (
        core_df.groupby("vid_bin", observed=False)[freq_col]
        .quantile(quantile)
        .reset_index(name=freq_col)
    )
    vf["vid_mid"] = vf["vid_bin"].apply(lambda interval: interval.mid)

    return vf[["vid_mid", freq_col]].dropna().reset_index(drop=True)


def fit_power_law_curve(
    vf_curve: pd.DataFrame,
    core: int | None = None,
    voltage_col: str = "vid_mid",
    freq_col: str = "clock",
    min_voltage: float = 0.95,
) -> PowerLawFitResult:
    """Fit a notebook-style power law to an extracted V/F curve."""

    fit_df = vf_curve[[voltage_col, freq_col]].dropna()
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
    residuals = freq - fitted

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((freq - np.mean(freq)) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1 - (ss_res / ss_tot)

    return PowerLawFitResult(
        core=core,
        a=float(params[0]),
        vt=float(params[1]),
        b=float(params[2]),
        r2=r2,
        residual_std=float(np.std(residuals, ddof=1)),
        residual_var=float(np.var(residuals, ddof=1)),
        n_samples=len(fit_df),
    )


def fit_power_law_per_core(
    df: pd.DataFrame,
    bins: int = 40,
    quantile: float = 0.99,
    freq_col: str = "clock",
) -> pd.DataFrame:
    """Fit power-law V/F curves per core and return a diagnostics dataframe."""

    rows: list[dict[str, float | int]] = []

    for core in sorted(df["core"].dropna().unique()):
        vf = extract_vf_curve(df, core=core, bins=bins, quantile=quantile, freq_col=freq_col)
        try:
            result = fit_power_law_curve(vf, core=int(core), freq_col=freq_col)
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
