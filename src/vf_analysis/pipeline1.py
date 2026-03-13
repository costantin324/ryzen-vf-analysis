"""Legacy dataset builder entrypoint; prefer ``vfanalysis.io.build_parquet_dataset``."""

from __future__ import annotations

from vfanalysis.io import build_parquet_dataset


def build_dataset() -> None:
    """Build the parquet dataset from raw HWInfo logs.

    Args:
        None.

    Returns:
        None.

    Assumptions:
        The environment variables used by :mod:`vfanalysis.io` are configured.
    """

    outputs = build_parquet_dataset()
    print(f"Prepared {len(outputs)} parquet files")


if __name__ == "__main__":
    build_dataset()
