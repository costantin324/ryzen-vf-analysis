from pathlib import Path
import os
from dotenv import load_dotenv

from vf_analysis.load_hwinfo import iter_hwinfo_logs
from vf_analysis.processing import detect_core_columns, build_core_dataframe

load_dotenv()

OUTPUT_DIR = Path(os.getenv("VF_DATASET_DIR"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_dataset():

    for df in iter_hwinfo_logs():

        run_id = df["run_id"].iloc[0]

        print(f"Processing {run_id}")

        core_map = detect_core_columns(df)
        core_df = build_core_dataframe(df, core_map)

        outfile = OUTPUT_DIR / f"{run_id}.parquet"
        if outfile.exists():
            print(f"Skipping {run_id}, already processed")
            continue

        core_df.to_parquet(outfile, compression="zstd")

        print(f"Saved {len(core_df):,} rows → {outfile}")


if __name__ == "__main__":
    build_dataset()