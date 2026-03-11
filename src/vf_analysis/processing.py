import re
import pandas as pd
from typing import Dict


def detect_core_columns(df: pd.DataFrame) -> Dict[int, dict]:
    """
    Detect HWInfo columns belonging to each CPU core.
    """

    core_map: Dict[int, dict] = {}

    for col in df.columns:

        m = re.match(r"Core (\d+) VID", col)
        if m:
            core = int(m.group(1))
            core_map.setdefault(core, {})["vid"] = col
            continue

        m = re.match(r"Core (\d+) Power", col)
        if m:
            core = int(m.group(1))
            core_map.setdefault(core, {})["power"] = col
            continue

        m = re.match(r"Core (\d+) Clock", col)
        if m:
            core = int(m.group(1))
            core_map.setdefault(core, {})["clock"] = col
            continue

        m = re.match(r"Core (\d+) T\d Effective Clock", col)
        if m:
            core = int(m.group(1))
            core_map.setdefault(core, {}).setdefault("eff_clocks", []).append(col)
            continue

        m = re.match(r"Core(\d+) \(CCD", col)
        if m:
            core = int(m.group(1))
            core_map.setdefault(core, {})["temp"] = col
            continue

    return core_map


def build_core_dataframe(df: pd.DataFrame, core_map: dict) -> pd.DataFrame:
    """
    Convert wide HWInfo dataframe into long per-core dataset.
    """

    core_dfs = []

    ppt = df.get("CPU PPT [W]")
    cpu_temp = df.get("CPU (Tctl/Tdie) [°C]")

    for core, sensors in core_map.items():

        eff_cols = sensors["eff_clocks"]

        df[eff_cols] = df[eff_cols].apply(pd.to_numeric, errors="coerce")

        eff_clock = df[eff_cols].mean(axis=1)

        core_df = pd.DataFrame({
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
            "cpu_temp": cpu_temp
        })

        core_dfs.append(core_df)

    result = pd.concat(core_dfs, ignore_index=True)

    return result.sort_values(["timestamp", "core"])

