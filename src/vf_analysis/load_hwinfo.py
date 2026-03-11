from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

data_dir_str: str | None = os.getenv("HWINFO_LOG_DIR")

if data_dir_str is None:
    raise RuntimeError("HWINFO_LOG_DIR not defined in environment")

DATA_DIR: Path = Path(data_dir_str)

if not DATA_DIR.exists():
    raise FileNotFoundError(f"{DATA_DIR} does not exist")


def load_hwinfo_csv(file: Path) -> pd.DataFrame:

    df = pd.read_csv(
        file,
        encoding="cp1252",
        low_memory=False,
        on_bad_lines="skip",
        engine="c"
    )

    # metadata
    df["source_file"] = file.name
    df["run_id"] = file.stem

    # build timestamp
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna(subset=["timestamp"])

    if df.empty:
        raise RuntimeError(f"{file} produced empty dataframe after timestamp parsing")

    # relative time
    df["t"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

    # convert telemetry columns to numeric
    exclude = {"Date", "Time", "timestamp", "t", "run_id", "source_file"}

    numeric_cols = [c for c in df.columns if c not in exclude]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.copy()

    return df


def load_all_logs() -> pd.DataFrame:
    """
    Load and concatenate all HWInfo logs.
    """

    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise RuntimeError(f"No CSV files found in {DATA_DIR}")

    dfs = [load_hwinfo_csv(f) for f in files]

    return pd.concat(dfs, ignore_index=True)

def iter_hwinfo_logs():
    """
    Yield HWInfo logs one-by-one instead of loading all at once.
    """
    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise RuntimeError(f"No CSV files found in {DATA_DIR}")

    for file in files:
        yield load_hwinfo_csv(file)