"""Visualization helpers for V/F analysis notebooks and scripts."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vfanalysis.metrics import boost_ridge
from vfanalysis.vfcurve import extract_vf_curve


def _core_list(df: pd.DataFrame, cores: Iterable[int] | None = None) -> list[int]:
    if cores is not None:
        return sorted(int(core) for core in cores)
    return sorted(int(core) for core in df["core"].dropna().unique())


def sample_df(df: pd.DataFrame, sample_size: int, seed: int = 42) -> pd.DataFrame:
    """Return up-to ``sample_size`` rows from dataframe."""

    return df.sample(min(len(df), sample_size), random_state=seed)


def plot_vf_hexbin_per_core(
    df: pd.DataFrame,
    freq_col: str = "eff_clock",
    sample_size: int = 200_000,
    gridsize: int = 60,
    cmap: str = "viridis",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot per-core VID vs frequency hexbins."""

    plot_df = sample_df(df, sample_size=sample_size)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    for core in _core_list(plot_df):
        ax = axes.flat[core]
        core_df = plot_df[plot_df["core"] == core]
        ax.hexbin(core_df["vid"], core_df[freq_col], gridsize=gridsize, cmap=cmap)
        ax.set_title(f"Core {core}")
        ax.set_xlabel("VID (V)")
        ax.set_ylabel(f"{freq_col} (MHz)")

    fig.tight_layout()
    return fig, axes


def plot_power_clock_hexbin(
    df: pd.DataFrame,
    gridsize: int = 60,
    cmap: str = "viridis",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot global power-vs-clock density."""

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hexbin(df["power"], df["clock"], gridsize=gridsize, cmap=cmap)
    ax.set_xlabel("Core Power (W)")
    ax.set_ylabel("Clock (MHz)")
    fig.tight_layout()
    return fig, ax


def plot_boost_ridge(
    df: pd.DataFrame,
    power_min: float = 10.0,
    quantile: float = 0.95,
    bins: int = 30,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot per-core boost ridge lines."""

    ridge_df = boost_ridge(df, power_min=power_min, quantile=quantile, bins=bins)

    fig, ax = plt.subplots(figsize=(8, 6))
    for core in _core_list(ridge_df):
        core_df = ridge_df[ridge_df["core"] == core]
        ax.plot(core_df["power_mid"], core_df["clock_quantile"], label=f"core {core}")

    ax.set_xlabel("Power (W)")
    ax.set_ylabel("Boost clock (MHz)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_vf_curves(
    df: pd.DataFrame,
    bins: int = 40,
    quantile: float = 0.99,
    freq_col: str = "clock",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot extracted V/F quantile curves for each core."""

    fig, ax = plt.subplots(figsize=(10, 6))

    for core in _core_list(df):
        curve_df = extract_vf_curve(df, core=core, bins=bins, quantile=quantile, freq_col=freq_col)
        if curve_df.empty:
            continue
        ax.plot(curve_df["vid_mid"], curve_df[freq_col], label=f"core {core}")

    ax.set_xlabel("VID (V)")
    ax.set_ylabel(f"{freq_col} (MHz)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_clock_variance_by_power(
    df: pd.DataFrame,
    power_min: float = 1.0,
    clock_min: float = 3000.0,
    power_bin_count: int = 30,
) -> tuple[plt.Figure, plt.Axes]:
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
    ax.legend()
    fig.tight_layout()
    return fig, ax
