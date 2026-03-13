"""Visualization helpers for V/F analysis notebooks and scripts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from vfanalysis._shared import sorted_core_ids
from vfanalysis.metrics import boost_ridge
from vfanalysis.vfcurve import extract_vf_curve, fit_power_law_curve, vf_power_law
from vfanalysis.workflow import select_workload_dataset


@dataclass(frozen=True)
class CoreTelemetryDiagnostics:
    """Summary diagnostics returned by :func:`plot_core_telemetry_3d`.

    Attributes:
        filtered: Filtered dataframe used for plotting.
        slope_by_core: Per-core ``dV/dF`` slopes estimated on the plotted sample.
        efficiency_by_core: Per-core median ``clock / power`` values estimated on the
            plotted sample.
    """

    filtered: pd.DataFrame
    slope_by_core: pd.DataFrame
    efficiency_by_core: pd.DataFrame


def _core_list(df: pd.DataFrame, cores: Iterable[int] | None = None) -> list[int]:
    """Return sorted core identifiers for plotting."""

    core_values = None if cores is None else list(cores)
    return sorted_core_ids(df, core_values)


def _apply_figure_spacing(
    fig: Figure,
    *,
    left: float = 0.08,
    right: float = 0.98,
    bottom: float = 0.10,
    top: float = 0.93,
    wspace: float = 0.28,
    hspace: float = 0.30,
) -> None:
    """Apply fixed subplot spacing without using ``tight_layout``.

    Args:
        fig: Figure to adjust.
        left: Left margin.
        right: Right margin.
        bottom: Bottom margin.
        top: Top margin.
        wspace: Width spacing between subplots.
        hspace: Height spacing between subplots.

    Returns:
        None.

    Assumptions:
        A fixed layout is preferable to Matplotlib's auto-layout heuristics for these
        notebook figures.
    """

    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=wspace,
        hspace=hspace,
    )


def sample_df(df: pd.DataFrame, sample_size: int, seed: int = 42) -> pd.DataFrame:
    """Return up to ``sample_size`` rows from a dataframe."""

    return df.sample(min(len(df), sample_size), random_state=seed)


def plot_vf_hexbin_per_core(
    df: pd.DataFrame,
    freq_col: str = "eff_clock",
    sample_size: int = 200_000,
    gridsize: int = 60,
    cmap: str = "viridis",
) -> tuple[Figure, np.ndarray]:
    """Plot per-core VID vs. frequency hexbins."""

    plot_df = sample_df(df, sample_size=sample_size)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    for core in _core_list(plot_df):
        ax = axes.flat[core]
        core_df = plot_df[plot_df["core"] == core]
        ax.hexbin(core_df["vid"], core_df[freq_col], gridsize=gridsize, cmap=cmap)
        ax.set_title(f"Core {core}")
        ax.set_xlabel("VID (V)")
        ax.set_ylabel(f"{freq_col} (MHz)")

    _apply_figure_spacing(fig, bottom=0.12, hspace=0.36)
    return fig, axes


def plot_power_clock_hexbin(
    df: pd.DataFrame,
    gridsize: int = 60,
    cmap: str = "viridis",
) -> tuple[Figure, Axes]:
    """Plot global power-vs-clock density."""

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hexbin(df["power"], df["clock"], gridsize=gridsize, cmap=cmap)
    ax.set_xlabel("Core Power (W)")
    ax.set_ylabel("Clock (MHz)")
    _apply_figure_spacing(fig, left=0.12, right=0.96, bottom=0.12, top=0.95)
    return fig, ax


def plot_boost_ridge(
    df: pd.DataFrame,
    power_min: float = 10.0,
    quantile: float = 0.95,
    bins: int = 30,
) -> tuple[Figure, Axes]:
    """Plot per-core boost ridge lines."""

    ridge_df = boost_ridge(df, power_min=power_min, quantile=quantile, bins=bins)

    fig, ax = plt.subplots(figsize=(8, 6))
    for core in _core_list(ridge_df):
        core_df = ridge_df[ridge_df["core"] == core]
        ax.plot(core_df["power_mid"], core_df["clock_quantile"], label=f"core {core}")

    ax.set_xlabel("Power (W)")
    ax.set_ylabel("Boost clock (MHz)")
    ax.legend()
    _apply_figure_spacing(fig, left=0.12, right=0.96, bottom=0.12, top=0.95)
    return fig, ax


def plot_vf_curves(
    df: pd.DataFrame,
    bins: int = 40,
    quantile: float = 0.99,
    freq_col: str = "clock",
) -> tuple[Figure, Axes]:
    """Plot extracted V/F quantile curves for each core."""

    fig, ax = plt.subplots(figsize=(10, 6))

    for core in _core_list(df):
        curve_df = extract_vf_curve(
            df,
            core=core,
            bins=bins,
            quantile=quantile,
            freq_col=freq_col,
        )
        if curve_df.empty:
            continue
        ax.plot(curve_df["vid_mid"], curve_df[freq_col], label=f"core {core}")

    ax.set_xlabel("VID (V)")
    ax.set_ylabel(f"{freq_col} (MHz)")
    ax.legend(ncols=2)
    _apply_figure_spacing(fig, left=0.10, right=0.98, bottom=0.12, top=0.95)
    return fig, ax


def plot_clock_variance_by_power(
    df: pd.DataFrame,
    power_min: float = 1.0,
    clock_min: float = 3000.0,
    power_bin_count: int = 30,
) -> tuple[Figure, Axes]:
    """Plot per-core clock standard deviation across power bins."""

    bins = np.linspace(power_min, float(df["power"].max()), power_bin_count)

    fig, ax = plt.subplots(figsize=(8, 6))

    for core in _core_list(df):
        core_df = df[df["core"] == core].copy()
        core_df = core_df[(core_df["power"] > power_min) & (core_df["clock"] > clock_min)]
        if core_df.empty:
            continue

        core_df["power_bin"] = pd.cut(core_df["power"], bins)
        stats = core_df.groupby("power_bin", observed=False).agg(
            mean_power=("power", "mean"),
            std_clock=("clock", "std"),
        )

        ax.plot(stats["mean_power"], stats["std_clock"], label=f"core {core}")

    ax.set_xlabel("Power (W)")
    ax.set_ylabel("Clock Std Dev (MHz)")
    ax.legend(ncols=2)
    _apply_figure_spacing(fig, left=0.12, right=0.96, bottom=0.12, top=0.95)
    return fig, ax


def plot_power_law_fits_per_core(
    df: pd.DataFrame,
    *,
    bins: int = 40,
    quantile: float = 0.99,
    freq_col: str = "clock",
    min_voltage: float | None = None,
) -> tuple[Figure, np.ndarray, pd.DataFrame]:
    """Plot all per-core power-law fits in a single multi-panel figure.

    Args:
        df: Input telemetry dataframe.
        bins: Number of voltage bins per core.
        quantile: Frequency quantile extracted per voltage bin.
        freq_col: Frequency column to model.
        min_voltage: Optional minimum voltage included in each fit.

    Returns:
        The figure, axes array, and per-core fit summary table.

    Assumptions:
        Core identifiers map cleanly onto a 2x4 subplot layout.
    """

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    rows: list[dict[str, float | int]] = []

    for core in _core_list(df):
        ax = axes.flat[core]
        vf = extract_vf_curve(df, core=core, bins=bins, quantile=quantile, freq_col=freq_col)
        if vf.empty:
            ax.set_visible(False)
            continue

        fit = fit_power_law_curve(
            vf,
            core=core,
            freq_col=freq_col,
            min_voltage=min_voltage,
        )
        voltage = vf["vid_mid"].to_numpy(dtype=float)
        freq = vf[freq_col].to_numpy(dtype=float)
        grid = np.linspace(float(voltage.min()), float(voltage.max()), 200)
        fitted = vf_power_law(grid, fit.a, fit.vt, fit.b)

        ax.scatter(voltage, freq, s=18, alpha=0.9, label="ridge points")
        ax.plot(grid, fitted, linewidth=2.5, label="power-law fit")
        ax.set_title(f"Core {core}")
        ax.set_xlabel("VID (V)")
        ax.set_ylabel(f"{freq_col} (MHz)")
        ax.text(
            0.03,
            0.97,
            "\n".join(
                [
                    f"a = {fit.a:.0f}",
                    f"Vt = {fit.vt:.3f} V",
                    f"b = {fit.b:.3f}",
                    f"R2 = {fit.r2:.4f}",
                ]
            ),
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "alpha": 0.7},
        )
        ax.legend(loc="lower right", fontsize=9)

        rows.append(
            {
                "core": core,
                "a": fit.a,
                "vt": fit.vt,
                "b": fit.b,
                "r2": fit.r2,
                "residual_std": fit.residual_std,
                "residual_var": fit.residual_var,
                "n_samples": fit.n_samples,
            }
        )

    _apply_figure_spacing(fig, left=0.06, right=0.98, bottom=0.08, top=0.93)
    return fig, axes, pd.DataFrame(rows).sort_values("core").reset_index(drop=True)


def plot_power_law_fit_span(
    df: pd.DataFrame,
    *,
    bins: int = 40,
    quantile: float = 0.99,
    freq_col: str = "clock",
    min_voltage: float | None = None,
    grid_points: int = 250,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Plot the cross-core spread of power-law fits over a shared voltage grid.

    Args:
        df: Input telemetry dataframe.
        bins: Number of voltage bins per core.
        quantile: Frequency quantile extracted per voltage bin.
        freq_col: Frequency column to model.
        min_voltage: Optional minimum voltage included in each fit.
        grid_points: Number of shared voltage grid points.

    Returns:
        The figure, axes, and a dataframe containing min/max/span by voltage.

    Assumptions:
        All cores share a meaningful overlapping voltage range for comparison.
    """

    curves: list[pd.DataFrame] = []
    fits: list[tuple[int, float, float, float]] = []

    for core in _core_list(df):
        curve = extract_vf_curve(df, core=core, bins=bins, quantile=quantile, freq_col=freq_col)
        if curve.empty:
            continue
        fit = fit_power_law_curve(
            curve,
            core=core,
            freq_col=freq_col,
            min_voltage=min_voltage,
        )
        curves.append(curve)
        fits.append((core, fit.a, fit.vt, fit.b))

    if not curves:
        raise ValueError("No per-core curves available for span plotting")

    grid_min = max(float(curve["vid_mid"].min()) for curve in curves)
    grid_max = min(float(curve["vid_mid"].max()) for curve in curves)
    if grid_min >= grid_max:
        grid_min = min(float(curve["vid_mid"].min()) for curve in curves)
        grid_max = max(float(curve["vid_mid"].max()) for curve in curves)

    voltage_grid = np.linspace(grid_min, grid_max, grid_points)
    predictions = []
    for _core, a, vt, b in fits:
        predicted = vf_power_law(voltage_grid, a, vt, b)
        predictions.append(predicted)

    prediction_matrix = np.vstack(predictions)
    span_df = pd.DataFrame(
        {
            "vid": voltage_grid,
            "clock_min": prediction_matrix.min(axis=0),
            "clock_max": prediction_matrix.max(axis=0),
            "clock_mean": prediction_matrix.mean(axis=0),
        }
    )
    span_df["clock_span"] = span_df["clock_max"] - span_df["clock_min"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for row, predicted in zip(fits, predictions, strict=True):
        ax.plot(voltage_grid, predicted, linewidth=1.4, alpha=0.5, label=f"core {row[0]}")

    ax.fill_between(
        span_df["vid"],
        span_df["clock_min"],
        span_df["clock_max"],
        alpha=0.22,
        color="tab:blue",
        label="per-core span",
    )
    ax.plot(span_df["vid"], span_df["clock_mean"], color="black", linewidth=2, label="mean fit")
    ax.set_xlabel("VID (V)")
    ax.set_ylabel(f"{freq_col} (MHz)")
    ax.set_title("Per-core fit spread across the shared voltage range")
    ax.legend(ncols=3, fontsize=9)
    _apply_figure_spacing(fig, left=0.10, right=0.98, bottom=0.12, top=0.93)
    return fig, ax, span_df


def plot_core_telemetry_3d(
    df: pd.DataFrame,
    *,
    power_min: float | None = None,
    power_max: float | None = None,
    clock_min: float | None = None,
    vid_min: float | None = None,
    vid_max: float | None = None,
    temp_max: float | None = None,
    sample_size: int = 10_000,
    zoom_high_freq: bool = False,
    seed: int = 42,
) -> tuple[Figure, CoreTelemetryDiagnostics]:
    """Plot notebook-style 3D per-core telemetry scatterplots."""

    filtered = select_workload_dataset(
        df,
        power_min=power_min,
        power_max=power_max,
        clock_min=clock_min,
        vid_min=vid_min,
        vid_max=vid_max,
        temp_max=temp_max,
    )

    fig = plt.figure(figsize=(24, 18))
    scatter_artist = None
    slope_rows: list[dict[str, float | int]] = []
    efficiency_rows: list[dict[str, float | int]] = []

    for core in _core_list(filtered):
        ax = fig.add_subplot(2, 4, core + 1, projection="3d")
        core_df = filtered[filtered["core"] == core]
        if core_df.empty:
            continue

        sample = sample_df(core_df, sample_size=sample_size, seed=seed)
        scatter_artist = ax.scatter(
            sample["vid"],
            sample["clock"],
            sample["power"],
            c=sample["temp"],
            cmap="plasma",
            s=2,
        )

        ax.set_title(f"Core {core}")
        ax.set_xlabel("VID")
        ax.set_ylabel("Clock")
        ax.set_zlabel("Power")

        slope = float("nan")
        if len(sample) >= 2:
            try:
                slope = float(np.polyfit(sample["clock"], sample["vid"], 1)[0])
            except (TypeError, ValueError):
                slope = float("nan")
        slope_rows.append({"core": core, "dv_df_slope": slope})

        efficiency = float((sample["clock"] / sample["power"]).median())
        efficiency_rows.append({"core": core, "clock_per_power": efficiency})

        if zoom_high_freq:
            threshold = float(sample["clock"].quantile(0.95))
            zoom = sample[sample["clock"] >= threshold]
            ax.scatter(
                zoom["vid"],
                zoom["clock"],
                zoom["power"],
                color="cyan",
                s=5,
                alpha=0.7,
            )

    if scatter_artist is not None:
        fig.subplots_adjust(right=0.88)
        colorbar_ax = fig.add_axes((0.90, 0.15, 0.02, 0.7))
        fig.colorbar(scatter_artist, cax=colorbar_ax, label="CPU Temp (deg C)")

    params = "\n".join(
        [
            "filters:",
            f"power_min={power_min}",
            f"power_max={power_max}",
            f"clock_min={clock_min}",
            f"vid_min={vid_min}",
            f"vid_max={vid_max}",
            f"temp_max={temp_max}",
        ]
    )
    fig.text(0.01, 0.01, params, fontsize=11, family="monospace")
    _apply_figure_spacing(fig, left=0.05, right=0.88, bottom=0.07, top=0.95, hspace=0.18)

    diagnostics = CoreTelemetryDiagnostics(
        filtered=filtered,
        slope_by_core=pd.DataFrame(slope_rows).sort_values("core").reset_index(drop=True),
        efficiency_by_core=(
            pd.DataFrame(efficiency_rows).sort_values("core").reset_index(drop=True)
        ),
    )
    return fig, diagnostics


def residual_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    x_bins: int = 35,
    y_bins: int = 35,
) -> pd.DataFrame:
    """Summarize a z-value surface over quantile-binned x/y coordinates."""

    heatmap_df = df.copy()
    heatmap_df["x_bin"] = pd.qcut(heatmap_df[x_col], x_bins, duplicates="drop")
    heatmap_df["y_bin"] = pd.qcut(heatmap_df[y_col], y_bins, duplicates="drop")

    pivot = heatmap_df.groupby(["y_bin", "x_bin"], observed=False)[z_col].median().unstack()
    pivot.index = pivot.index.map(lambda interval: float(interval.mid))
    pivot.columns = pivot.columns.map(lambda interval: float(interval.mid))
    return pivot


