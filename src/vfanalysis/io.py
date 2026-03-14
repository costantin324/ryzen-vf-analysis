"""I/O utilities for HWInfo ingestion and processed parquet datasets."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Sequence
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
_NON_ALNUM_RE = re.compile(r"[^0-9A-Za-z]+")
_ENV_LOG_PREFIX = "HWINFO_LOG_DIR"

_GLOBAL_SENSOR_CANDIDATES: dict[str, tuple[list[str], list[str]]] = {
    "cpu_temp": (
        ["CPU (Tctl/Tdie) [°C]", "CPU (Tctl/Tdie)"],
        ["CPU (Tctl/Tdie)"],
    ),
    "cpu_package_power": (
        ["CPU Package Power [W]", "CPU Package Power"],
        ["CPU Package Power"],
    ),
    "cpu_ppt": (
        ["CPU PPT [W]", "CPU PPT"],
        ["CPU PPT"],
    ),
    "cpu_soc_power": (
        ["CPU SoC Power [W]", "CPU SoC Power"],
        ["CPU SoC Power"],
    ),
    "cpu_soc_misc_power": (
        ["CPU SoC + MISC Power [W]", "CPU SoC + MISC Power"],
        ["CPU SoC + MISC Power"],
    ),
    "cpu_ppt_limit": (
        ["CPU PPT Limit [W]", "CPU PPT Limit"],
        ["CPU PPT Limit"],
    ),
    "cpu_tdc_limit": (
        ["CPU TDC Limit [A]", "CPU TDC Limit"],
        ["CPU TDC Limit"],
    ),
    "cpu_edc_limit": (
        ["CPU EDC Limit [A]", "CPU EDC Limit"],
        ["CPU EDC Limit"],
    ),
    "cpu_ppt_fast_limit": (
        ["CPU PPT FAST Limit [W]", "CPU PPT FAST Limit"],
        ["CPU PPT FAST Limit"],
    ),
    "thermal_limit": (
        ["Thermal Limit [°C]", "Thermal Limit"],
        ["Thermal Limit"],
    ),
    "cpu_core_current": (
        [
            "CPU Core Current (SVI3 TFN) [A]",
            "CPU Core Current [A]",
            "CPU Core Current",
        ],
        ["CPU Core Current"],
    ),
    "soc_current": (
        [
            "CPU SoC Current (SVI3 TFN) [A]",
            "SoC Current [A]",
            "SoC Current",
        ],
        ["CPU SoC Current", "SoC Current"],
    ),
    "cpu_tdc": (
        ["CPU TDC [A]", "CPU TDC"],
        ["CPU TDC"],
    ),
    "cpu_edc": (
        ["CPU EDC [A]", "CPU EDC"],
        ["CPU EDC"],
    ),
    "gpu_temp": (
        ["GPU Temperature [°C]", "GPU Temperature"],
        ["GPU Temperature"],
    ),
    "gpu_memory_temp": (
        ["GPU Memory Temperature [°C]", "GPU Memory Temperature"],
        ["GPU Memory Temperature"],
    ),
    "gpu_core_voltage": (
        ["GPU Core Voltage [V]", "GPU Core Voltage", "GPU Core Voltage (VDDCR_GFX) [V]"],
        ["GPU Core Voltage", "VDDCR_GFX"],
    ),
    "gpu_clock": (
        ["GPU Clock [MHz]", "GPU Clock"],
        ["GPU Clock"],
    ),
    "gpu_memory_clock": (
        ["GPU Memory Clock [MHz]", "GPU Memory Clock"],
        ["GPU Memory Clock"],
    ),
    "gpu_utilization": (
        ["GPU Utilization [%]", "GPU Utilization"],
        ["GPU Utilization"],
    ),
    "framerate": (
        ["Framerate [FPS]", "Framerate"],
        ["Framerate"],
    ),
    "framerate_presented": (
        ["Framerate (Presented) [FPS]", "Framerate (Presented)"],
        ["Framerate (Presented)"],
    ),
    "framerate_displayed": (
        ["Framerate (Displayed) [FPS]", "Framerate (Displayed)"],
        ["Framerate (Displayed)"],
    ),
    "frame_time": (
        ["Frame Time [ms]", "Frame Time"],
        ["Frame Time"],
    ),
    "gpu_busy": (
        ["GPU Busy [%]", "GPU Busy"],
        ["GPU Busy"],
    ),
    "gpu_wait": (
        ["GPU Wait [%]", "GPU Wait"],
        ["GPU Wait"],
    ),
    "cpu_busy": (
        ["CPU Busy [%]", "CPU Busy"],
        ["CPU Busy"],
    ),
    "cpu_wait": (
        ["CPU Wait [%]", "CPU Wait"],
        ["CPU Wait"],
    ),
}


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


def _deduplicate_paths(paths: Sequence[Path]) -> list[Path]:
    """Return unique paths while preserving input order."""

    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(path)
    return ordered


def _sanitize_label(value: str) -> str:
    """Convert a filesystem label into a stable ASCII slug."""

    slug = _NON_ALNUM_RE.sub("_", value).strip("_").lower()
    return slug or "logs"


def hwinfo_log_dir() -> Path:
    """Return the configured primary raw HWInfo log directory.

    Args:
        None.

    Returns:
        The path stored in ``HWINFO_LOG_DIR``.

    Assumptions:
        ``HWINFO_LOG_DIR`` is defined and points to an existing directory.
    """

    return _read_env_path(_ENV_LOG_PREFIX)


def hwinfo_log_dirs() -> list[Path]:
    """Return all configured raw HWInfo log directories.

    Args:
        None.

    Returns:
        A deterministic list of configured raw log directories.

    Assumptions:
        At least one environment variable named ``HWINFO_LOG_DIR`` or prefixed
        with ``HWINFO_LOG_DIR_`` is defined and points at an existing directory.
    """

    paths: list[Path] = []

    if os.getenv(_ENV_LOG_PREFIX):
        paths.append(_read_env_path(_ENV_LOG_PREFIX))

    for name in sorted(os.environ):
        if not name.startswith(f"{_ENV_LOG_PREFIX}_"):
            continue
        paths.append(_read_env_path(name))

    if not paths:
        raise RuntimeError("No HWINFO_LOG_DIR environment variables are defined")

    return _deduplicate_paths(paths)


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


def _log_group_for_file(file: Path, log_roots: Sequence[Path]) -> str:
    """Return a stable label describing which configured log directory a file came from."""

    resolved_file = file.resolve()
    for root in log_roots:
        resolved_root = root.resolve()
        if resolved_root == resolved_file.parent.resolve():
            return _sanitize_label(root.name)
    return _sanitize_label(file.parent.name)


def _output_stem_for_file(file: Path, log_roots: Sequence[Path]) -> str:
    """Return a parquet output stem that remains unique across multiple log roots."""

    if len(log_roots) == 1:
        return file.stem
    return f"{_log_group_for_file(file, log_roots)}__{file.stem}"


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


def iter_hwinfo_log_files(log_dir: Path | None = None) -> Iterator[Path]:
    """Yield raw HWInfo CSV paths one by one.

    Args:
        log_dir: Optional directory override. When omitted, all configured raw
            log directories are used.

    Returns:
        An iterator of CSV paths.

    Assumptions:
        Every selected directory contains at least one ``.csv`` file exported by
        HWInfo.
    """

    log_dirs = [log_dir] if log_dir is not None else hwinfo_log_dirs()
    files: list[Path] = []
    for directory in log_dirs:
        files.extend(sorted(directory.glob("*.csv")))

    if not files:
        joined = ", ".join(str(directory) for directory in log_dirs)
        raise RuntimeError(f"No CSV files found in {joined}")

    yield from files


def iter_hwinfo_logs(log_dir: Path | None = None) -> Iterator[pd.DataFrame]:
    """Yield parsed HWInfo logs one by one.

    Args:
        log_dir: Optional directory override. When omitted, all configured raw
            log directories are used.

    Returns:
        An iterator of parsed dataframes, one per CSV file.

    Assumptions:
        The selected directories contain readable HWInfo CSV exports.
    """

    for file in iter_hwinfo_log_files(log_dir=log_dir):
        yield load_hwinfo_csv(file)


def load_all_hwinfo_logs(log_dir: Path | None = None) -> pd.DataFrame:
    """Load and concatenate all HWInfo logs into a single dataframe.

    Args:
        log_dir: Optional directory override.

    Returns:
        Concatenated HWInfo telemetry data.

    Assumptions:
        All selected CSV files share a compatible schema.
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


def _get_first_present(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series | None:
    """Return the first exact column present from a candidate list.

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


def _get_first_matching(df: pd.DataFrame, needles: Sequence[str]) -> pd.Series | None:
    """Return the first column whose name contains any requested substring.

    Args:
        df: Input dataframe.
        needles: Substrings to search for in column names.

    Returns:
        The first matching series, or ``None`` when no column matches.

    Assumptions:
        Case-insensitive substring matching is sufficient to identify the desired
        HWInfo sensor.
    """

    lowered_needles = [needle.casefold() for needle in needles]
    for column in df.columns:
        lowered_column = str(column).casefold()
        if any(needle in lowered_column for needle in lowered_needles):
            return df[column]
    return None


def extract_global_telemetry(df: pd.DataFrame) -> dict[str, pd.Series | None]:
    """Extract run-level CPU, GPU, and frame telemetry series from a wide log.

    Args:
        df: Wide-form HWInfo dataframe.

    Returns:
        A mapping from normalized telemetry names to matching series, or ``None``
        when a sensor is unavailable.

    Assumptions:
        Global telemetry sensors can be identified from exact names or stable
        substrings in the HWInfo export.
    """

    telemetry: dict[str, pd.Series | None] = {}
    for output_name, (exact_candidates, substring_candidates) in _GLOBAL_SENSOR_CANDIDATES.items():
        series = _get_first_present(df, exact_candidates)
        if series is None:
            series = _get_first_matching(df, substring_candidates)
        telemetry[output_name] = series
    return telemetry


def build_core_dataframe(
    df: pd.DataFrame,
    core_map: dict[int, CoreSensors] | None = None,
    *,
    log_group: str | None = None,
) -> pd.DataFrame:
    """Convert a wide HWInfo dataframe into long per-core telemetry rows.

    Args:
        df: Wide-form HWInfo dataframe.
        core_map: Optional precomputed sensor-column mapping.
        log_group: Optional label identifying which configured log directory
            produced the dataframe.

    Returns:
        Long-form per-core telemetry data sorted by timestamp and core.

    Assumptions:
        The dataframe contains timestamp metadata columns and at least one core
        with VID, clock, and effective-clock sensors.
    """

    map_to_use = core_map if core_map is not None else detect_core_columns(df)
    global_telemetry = extract_global_telemetry(df)
    group_label = log_group if log_group is not None else "logs"

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
                "log_group": group_label,
                "source_file": df["source_file"],
                "core": core,
                "vid": df[sensors["vid"]],
                "clock": df[sensors["clock"]],
                "eff_clock": eff_clock,
                "temp": df[sensors["temp"]] if "temp" in sensors else None,
                "power": df[sensors["power"]] if "power" in sensors else None,
                "cpu_package_power": global_telemetry["cpu_package_power"],
                "ppt": global_telemetry["cpu_ppt"],
                "cpu_ppt": global_telemetry["cpu_ppt"],
                "cpu_soc_power": global_telemetry["cpu_soc_power"],
                "cpu_soc_misc_power": global_telemetry["cpu_soc_misc_power"],
                "cpu_ppt_limit": global_telemetry["cpu_ppt_limit"],
                "cpu_tdc_limit": global_telemetry["cpu_tdc_limit"],
                "cpu_edc_limit": global_telemetry["cpu_edc_limit"],
                "cpu_ppt_fast_limit": global_telemetry["cpu_ppt_fast_limit"],
                "thermal_limit": global_telemetry["thermal_limit"],
                "cpu_core_current": global_telemetry["cpu_core_current"],
                "soc_current": global_telemetry["soc_current"],
                "cpu_tdc": global_telemetry["cpu_tdc"],
                "cpu_edc": global_telemetry["cpu_edc"],
                "cpu_temp": global_telemetry["cpu_temp"],
                "gpu_temp": global_telemetry["gpu_temp"],
                "gpu_memory_temp": global_telemetry["gpu_memory_temp"],
                "gpu_core_voltage": global_telemetry["gpu_core_voltage"],
                "gpu_clock": global_telemetry["gpu_clock"],
                "gpu_memory_clock": global_telemetry["gpu_memory_clock"],
                "gpu_utilization": global_telemetry["gpu_utilization"],
                "framerate": global_telemetry["framerate"],
                "framerate_presented": global_telemetry["framerate_presented"],
                "framerate_displayed": global_telemetry["framerate_displayed"],
                "frame_time": global_telemetry["frame_time"],
                "gpu_busy": global_telemetry["gpu_busy"],
                "gpu_wait": global_telemetry["gpu_wait"],
                "cpu_busy": global_telemetry["cpu_busy"],
                "cpu_wait": global_telemetry["cpu_wait"],
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

    log_roots = [log_dir] if log_dir is not None else hwinfo_log_dirs()
    outputs: list[Path] = []

    for file in iter_hwinfo_log_files(log_dir=log_dir):
        log_df = load_hwinfo_csv(file)
        outfile = out_dir / f"{_output_stem_for_file(file, log_roots)}.parquet"

        if outfile.exists() and not overwrite:
            outputs.append(outfile)
            continue

        core_map = detect_core_columns(log_df)
        core_df = build_core_dataframe(
            log_df,
            core_map,
            log_group=_log_group_for_file(file, log_roots),
        )
        core_df.to_parquet(outfile, compression=compression, index=False)
        outputs.append(outfile)

    return outputs


def aggregate_parquet_dataset(
    dataset_dir: Path | None = None,
    output_file: Path | None = None,
    pattern: str = "*.parquet",
    compression: str = "zstd",
) -> Path:
    """Aggregate processed parquet shards into one parquet file.

    Args:
        dataset_dir: Optional parquet directory override.
        output_file: Optional output parquet path. Defaults to a sibling file next
            to the dataset directory.
        pattern: Glob pattern selecting parquet shard files.
        compression: Parquet compression codec.

    Returns:
        The path to the aggregate parquet file.

    Assumptions:
        The selected parquet files share a compatible long-format schema and can
        be concatenated row-wise.
    """

    source_dir = dataset_dir if dataset_dir is not None else vf_dataset_dir()
    target = (
        output_file
        if output_file is not None
        else source_dir.parent / f"{source_dir.name}_aggregate.parquet"
    )
    aggregated = load_processed_dataset(dataset_dir=source_dir, pattern=pattern)
    target.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_parquet(target, compression=compression, index=False)
    return target


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


