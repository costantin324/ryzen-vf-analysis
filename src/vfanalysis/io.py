"""I/O utilities for HWInfo ingestion and processed parquet datasets."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_CORE_VID_RE = re.compile(r"Core (\d+) VID")
_CORE_POWER_RE = re.compile(r"Core (\d+) Power")
_CORE_CLOCK_RE = re.compile(r"Core (\d+) Clock")
_CORE_EFFECTIVE_RE = re.compile(r"Core (\d+) T\d Effective Clock")
_CORE_EFFECTIVE_ALT_RE = re.compile(r"Core (\d+) Effective Clock")
_CORE_TEMP_RE = re.compile(r"Core(\d+) \(CCD")


def _read_env_path(var_name: str) -> Path:
    value = os.getenv(var_name)
    if value is None:
        raise RuntimeError(f"{var_name} not defined in environment")

    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    return path


def hwinfo_log_dir() -> Path:
    """Return the configured raw HWInfo log directory from ``HWINFO_LOG_DIR``."""

    return _read_env_path("HWINFO_LOG_DIR")


def vf_dataset_dir() -> Path:
    """Return the configured processed dataset directory from ``VF_DATASET_DIR``."""

    return _read_env_path("VF_DATASET_DIR")


def load_hwinfo_csv(file: Path) -> pd.DataFrame:
    """Load one HWInfo CSV file and normalize timestamps + numeric columns."""

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


def iter_hwinfo_logs(log_dir: Path | None = None) -> Iterable[pd.DataFrame]:
    """Yield parsed HWInfo logs one-by-one from ``log_dir`` (or ``HWINFO_LOG_DIR``)."""

    input_dir = log_dir if log_dir is not None else hwinfo_log_dir()
    files = sorted(input_dir.glob("*.csv"))

    if not files:
        raise RuntimeError(f"No CSV files found in {input_dir}")

    for file in files:
        yield load_hwinfo_csv(file)


def load_all_hwinfo_logs(log_dir: Path | None = None) -> pd.DataFrame:
    """Load and concatenate all HWInfo logs in one dataframe."""

    return pd.concat(iter_hwinfo_logs(log_dir=log_dir), ignore_index=True)


def detect_core_columns(df: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """Detect per-core sensor columns from a wide HWInfo dataframe."""

    core_map: dict[int, dict[str, Any]] = {}

    for col in df.columns:
        vid_match = _CORE_VID_RE.match(col)
        if vid_match:
            core = int(vid_match.group(1))
            core_map.setdefault(core, {})["vid"] = col
            continue

        power_match = _CORE_POWER_RE.match(col)
        if power_match:
            core = int(power_match.group(1))
            core_map.setdefault(core, {})["power"] = col
            continue

        clock_match = _CORE_CLOCK_RE.match(col)
        if clock_match:
            core = int(clock_match.group(1))
            core_map.setdefault(core, {})["clock"] = col
            continue

        effective_match = _CORE_EFFECTIVE_RE.match(col)
        if effective_match:
            core = int(effective_match.group(1))
            core_map.setdefault(core, {}).setdefault("eff_clocks", []).append(col)
            continue

        effective_alt_match = _CORE_EFFECTIVE_ALT_RE.match(col)
        if effective_alt_match:
            core = int(effective_alt_match.group(1))
            core_map.setdefault(core, {}).setdefault("eff_clocks", []).append(col)
            continue

        temp_match = _CORE_TEMP_RE.match(col)
        if temp_match:
            core = int(temp_match.group(1))
            core_map.setdefault(core, {})["temp"] = col

    return core_map


def _get_first_present(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for candidate in candidates:
        if candidate in df.columns:
            return df[candidate]
    return None


def _get_first_matching(df: pd.DataFrame, needle: str) -> pd.Series | None:
    for column in df.columns:
        if needle in column:
            return df[column]
    return None


def build_core_dataframe(
    df: pd.DataFrame,
    core_map: dict[int, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Convert wide HWInfo dataframe into long per-core telemetry rows."""

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

        df[eff_cols] = df[eff_cols].apply(pd.to_numeric, errors="coerce")
        eff_clock = df[eff_cols].mean(axis=1)

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
    """Build long-format parquet files from all HWInfo CSV logs."""

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
    """Load all processed parquet files from ``VF_DATASET_DIR`` (or ``dataset_dir``)."""

    source_dir = dataset_dir if dataset_dir is not None else vf_dataset_dir()
    files = sorted(source_dir.glob(pattern))

    if not files:
        raise RuntimeError(f"No parquet files found in {source_dir}")

    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
