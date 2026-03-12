import numpy as np
import pandas as pd


def dv_df_slope(df: pd.DataFrame, core: int | None = None) -> float:
    """
    Estimate voltage-frequency slope (dV/dF).

    Uses linear regression:
        vid = a + b * clock

    Returns slope b.
    """

    d = df if core is None else df[df["core"] == core]

    if len(d) < 10:
        return float("nan")

    slope = np.polyfit(d["clock"], d["vid"], 1)[0]

    return float(slope)


def clock_per_power(df: pd.DataFrame) -> float:
    """
    Compute median clock per watt efficiency.
    """

    return float((df["clock"] / df["power"]).median())


def silicon_efficiency(df: pd.DataFrame) -> float:
    """
    Approximate silicon efficiency metric.

    Higher values indicate better cores.
    """

    return float((df["clock"] / (df["vid"] ** 2)).median())


def core_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce per-core summary statistics.
    """

    rows = []

    for core in sorted(df["core"].unique()):
        d = df[df["core"] == core]

        rows.append(
            {
                "core": core,
                "dv_df": dv_df_slope(d),
                "clock_per_power": clock_per_power(d),
                "silicon_efficiency": silicon_efficiency(d),
                "max_clock": d["clock"].max(),
                "median_power": d["power"].median(),
            }
        )

    return pd.DataFrame(rows)
