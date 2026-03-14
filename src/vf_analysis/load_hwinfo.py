"""Compatibility wrapper around :mod:`vfanalysis.io`."""

from __future__ import annotations

import pandas as pd

from vfanalysis.io import (
    aggregate_parquet_dataset,
    hwinfo_log_dirs,
    iter_hwinfo_log_files,
    iter_hwinfo_logs,
    load_all_hwinfo_logs,
    load_hwinfo_csv,
)

__all__ = [
    "aggregate_parquet_dataset",
    "hwinfo_log_dirs",
    "iter_hwinfo_log_files",
    "iter_hwinfo_logs",
    "load_all_hwinfo_logs",
    "load_all_logs",
    "load_hwinfo_csv",
]


def load_all_logs() -> pd.DataFrame:
    """Return all HWInfo logs via the legacy function name.

    Args:
        None.

    Returns:
        Concatenated HWInfo telemetry data.

    Assumptions:
        The configured log directories contain at least one readable HWInfo CSV.
    """

    return load_all_hwinfo_logs()

