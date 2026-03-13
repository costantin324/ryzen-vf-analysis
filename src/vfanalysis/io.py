"""I/O utilities for HWInfo ingestion and processed parquet datasets."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_CORE_VID_RE = re.compile(r"Core (\d+) VID")
_CORE_POWER_RE = re.compile(r"Core (\d+) Power")
_CORE_CLOCK_RE = re.compile(r"Core (\d+) Clock")
_CORE_EFFECTIVE_RE = re.compile(r"Core (\d+) T\d Effective Clock")
_CORE_EFFECTIVE_ALT_RE = re.compile(r"Core (\d+) Effective Clock")
_CORE_TEMP_RE = re.compile(r"Core(\d+) \(CCD")


class CoreSensors(TypedDict, total=False):
    """Per-core sensor column mapping detected from a wide HWInfo frame."""

    vid: str
    power: str
    clock: str
    eff_clocks: list[str]
    temp: str


def _read_env_path(var_name: str) -> Path:
    """Read and validate a filesystem path from the environment.

    Args:
        var_name: Environment variable name containing a path.

    Returns:
        A validated existing path.

    Assumptions:
        The environment variable is expected to exist and point at a readable
        file or directory.
    """

    value = os.getenv(var_name)
    if value is None:
        raise RuntimeError(f"{var_name} not defined in environment")

    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    return path


def hwinfo_log_dir() -> Path:
    """Return the configured raw HWInfo log directory.

    Args:
        None.

    Returns:
        The path stored in ``HWINFO_LOG_DIR``.

    Assumptions:
        ``HWINFO_LOG_DIR`` is defined and points to an existing directory.
    """

    return _read_env_path("HWINFO_LOG_DIR")


def vf_dataset_dir() -> Path:
    """Return the configured processed dataset directory.

    Args:
        None.

    Returns:
        The path stored in ``VF_DATASET_DIR``.

    Assumptions:
        ``VF_DATASET_DIR`` is defined and points to an existing directory.
    """

    return _read_env_path("VF_DATASET_DIR")


def load_hwinfo_csv(file: Path) -> pd.DataFrame:
    """Load one HWInfo CSV file and normalize timestamps and numeric columns.

    Args:
        file: HWInfo CSV file encoded with CP1252-style Windows defaults.

    Returns:
        A dataframe with normalized timestamps, ``t`` in seconds, and numeric
        sensor columns coerced to floats where possible.

    Assumptions:
        The CSV contains ``Date`` and ``Time`` columns in the format exported by
        HWInfo.
    """

    df = pd.read_csv(
        file,
        encoding="cp1252",
        low_memory=False,
        on_bad_lines="skip",
        engine="c",
    )

    df["source_file"] = file.name
    df["run_id"] = file.stem
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )

    df = df.dropna(subset=["timestamp"])
    if df.empty:
        raise RuntimeError(f"{file} produced empty dataframe after timestamp parsing")

    df["t"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

    exclude = {"Date", "Time", "timestamp", "t", "run_id", "source_file"}
    numeric_cols = [column for column in df.columns if column not in exclude]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df.copy()


def iter_hwinfo_logs(log_dir: Path | None = None) -> Iterator[pd.DataFrame]:
    """Yield parsed HWInfo logs one by one.

    Args:
        log_dir: Optional directory override. When omitted,
            :func:`hwinfo_log_dir` is used.

    Returns:
        An iterator of parsed dataframes, one per CSV file.

    Assumptions:
        The directory contains at least one ``.csv`` file exported by HWInfo.
    """

    input_dir = log_dir if log_dir is not None else hwinfo_log_dir()
    files = sorted(input_dir.glob("*.csv"))

    if not files:
        raise RuntimeError(f"No CSV files found in {input_dir}")

    for file in files:
        yield load_hwinfo_csv(file)


def load_all_hwinfo_logs(log_dir: Path | None = None) -> pd.DataFrame:
    """Load and concatenate all HWInfo logs into a single dataframe.

    Args:
        log_dir: Optional directory override.

    Returns:
        Concatenated HWInfo telemetry data.

    Assumptions:
        All CSV files in the selected directory share a compatible schema.
    """

    return pd.concat(iter_hwinfo_logs(log_dir=log_dir), ignore_index=True)


def detect_core_columns(df: pd.DataFrame) -> dict[int, CoreSensors]:
    """Detect per-core sensor columns from a wide HWInfo dataframe.

    Args:
        df: Wide-form HWInfo dataframe with one column per sensor.

    Returns:
        A mapping from integer core ID to sensor column names.

    Assumptions:
        Sensor columns follow the naming patterns emitted by HWInfo for Ryzen
        cores.
    """

    core_map: dict[int, CoreSensors] = {}

    for column in df.columns:
        vid_match = _CORE_VID_RE.match(column)
        if vid_match:
            core = int(vid_match.group(1))
            core_map.setdefault(core, {})["vid"] = column
            continue

        power_match = _CORE_POWER_RE.match(column)
        if power_match:
            core = int(power_match.group(1))
            core_map.setdefault(core, {})["power"] = column
            continue

        clock_match = _CORE_CLOCK_RE.match(column)
        if clock_match:
            core = int(clock_match.group(1))
            core_map.setdefault(core, {})["clock"] = column
            continue

        effective_match = _CORE_EFFECTIVE_RE.match(column)
        if effective_match:
            core = int(effective_match.group(1))
            core_map.setdefault(core, {}).setdefault("eff_clocks", []).append(column)
            continue

        effective_alt_match = _CORE_EFFECTIVE_ALT_RE.match(column)
        if effective_alt_match:
            core = int(effective_alt_match.group(1))
            core_map.setdefault(core, {}).setdefault("eff_clocks", []).append(column)
            continue

        temp_match = _CORE_TEMP_RE.match(column)
        if temp_match:
            core = int(temp_match.group(1))
            core_map.setdefault(core, {})["temp"] = column

    return core_map


def _get_first_present(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    """Return the first column present from a candidate list.

    Args:
        df: Input dataframe.
        candidates: Candidate column names ordered by preference.

    Returns:
        The first matching series, or ``None`` when no candidate exists.

    Assumptions:
        Candidate names are valid string column labels.
    """

    for candidate in candidates:
        if candidate in df.columns:
            return df[candidate]
    return None


def _get_first_matching(df: pd.DataFrame, needle: str) -> pd.Series | None:
    """Return the first column whose name contains a substring.

    Args:
        df: Input dataframe.
        needle: Substring to search for in column names.

    Returns:
        The first matching series, or ``None`` when no column matches.

    Assumptions:
        Substring matching is sufficient to identify the desired HWInfo sensor.
    """

    for column in df.columns:
        if needle in column:
            return df[column]
    return None


def build_core_dataframe(
    df: pd.DataFrame,
    core_map: dict[int, CoreSensors] | None = None,
) -> pd.DataFrame:
    """Convert a wide HWInfo dataframe into long per-core telemetry rows.

    Args:
        df: Wide-form HWInfo dataframe.
        core_map: Optional precomputed sensor-column mapping.

    Returns:
        Long-form per-core telemetry data sorted by timestamp and core.

    Assumptions:
        The dataframe contains timestamp metadata columns and at least one core
        with VID, clock, and effective-clock sensors.
    """

    map_to_use = core_map if core_map is not None else detect_core_columns(df)
    ppt = _get_first_present(df, ["CPU PPT [W]", "CPU PPT"])
    cpu_temp = _get_first_matching(df, "CPU (Tctl/Tdie)")

    core_frames: list[pd.DataFrame] = []

    for core, sensors in sorted(map_to_use.items()):
        if "vid" not in sensors or "clock" not in sensors:
            continue

        eff_cols = sensors.get("eff_clocks", [])
        if not eff_cols:
            continue

        numeric_eff = df[eff_cols].apply(pd.to_numeric, errors="coerce")
        eff_clock = numeric_eff.mean(axis=1)

        frame = pd.DataFrame(
            {
                "timestamp": df["timestamp"],
                "t": df["t"],
                "run_id": df["run_id"],
                "source_file": df["source_file"],
                "core": core,
                "vid": df[sensors["vid"]],
                "clock": df[sensors["clock"]],
                "eff_clock": eff_clock,
                "temp": df[sensors["temp"]] if "temp" in sensors else None,
                "power": df[sensors["power"]] if "power" in sensors else None,
                "ppt": ppt,
                "cpu_temp": cpu_temp,
            }
        )
        frame["eff_ratio"] = frame["eff_clock"] / frame["clock"]
        core_frames.append(frame)

    if not core_frames:
        raise RuntimeError("No core telemetry columns were detected in dataframe")

    result = pd.concat(core_frames, ignore_index=True)
    return result.sort_values(["timestamp", "core"]).reset_index(drop=True)


def build_parquet_dataset(
    log_dir: Path | None = None,
    dataset_dir: Path | None = None,
    overwrite: bool = False,
    compression: str = "zstd",
) -> list[Path]:
    """Build long-format parquet files from HWInfo CSV logs.

    Args:
        log_dir: Optional raw CSV directory override.
        dataset_dir: Optional parquet output directory override.
        overwrite: Whether existing parquet files should be regenerated.
        compression: Parquet compression codec.

    Returns:
        Paths to the generated or reused parquet files.

    Assumptions:
        Raw logs share a compatible schema and can be converted independently.
    """

    out_dir = dataset_dir if dataset_dir is not None else vf_dataset_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for log_df in iter_hwinfo_logs(log_dir=log_dir):
        run_id = str(log_df["run_id"].iloc[0])
        outfile = out_dir / f"{run_id}.parquet"

        if outfile.exists() and not overwrite:
            outputs.append(outfile)
            continue

        core_map = detect_core_columns(log_df)
        core_df = build_core_dataframe(log_df, core_map)
        core_df.to_parquet(outfile, compression=compression)
        outputs.append(outfile)

    return outputs


def load_processed_dataset(
    dataset_dir: Path | None = None,
    pattern: str = "*.parquet",
) -> pd.DataFrame:
    """Load processed parquet files into one dataframe.

    Args:
        dataset_dir: Optional parquet directory override.
        pattern: Glob pattern selecting parquet files.

    Returns:
        Concatenated processed telemetry dataframe.

    Assumptions:
        The selected parquet files share a compatible long-format schema.
    """

    source_dir = dataset_dir if dataset_dir is not None else vf_dataset_dir()
    files = sorted(source_dir.glob(pattern))

    if not files:
        raise RuntimeError(f"No parquet files found in {source_dir}")

    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
