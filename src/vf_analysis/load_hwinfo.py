"""Compatibility wrapper around :mod:`vfanalysis.io`."""

from vfanalysis.io import iter_hwinfo_logs, load_all_hwinfo_logs, load_hwinfo_csv

__all__ = ["iter_hwinfo_logs", "load_all_hwinfo_logs", "load_all_logs", "load_hwinfo_csv"]


def load_all_logs():
    """Legacy alias for ``load_all_hwinfo_logs``."""

    return load_all_hwinfo_logs()
