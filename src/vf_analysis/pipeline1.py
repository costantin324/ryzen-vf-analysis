"""Legacy dataset builder entrypoint; prefer ``vfanalysis.io`` helpers."""

from __future__ import annotations

from vfanalysis.io import aggregate_parquet_dataset, build_parquet_dataset


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


def aggregate_dataset() -> None:
    """Aggregate processed parquet shards into one parquet file.

    Args:
        None.

    Returns:
        None.

    Assumptions:
        The processed parquet dataset exists and shares a compatible schema.
    """

    aggregate_file = aggregate_parquet_dataset()
    print(f"Wrote aggregate parquet to {aggregate_file}")


if __name__ == "__main__":
    build_dataset()

