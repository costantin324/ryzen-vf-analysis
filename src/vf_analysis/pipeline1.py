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



    """Good instinct — documenting the **data lineage** now will save you a lot of confusion later. What you want is essentially a short **“data processing contract”** describing what each stage does and what it does *not* do.

Below is a clean structure you can drop into a README or notebook.

---

# Data Processing Pipeline (HWInfo → V/F Dataset)

## Stage 1 — Raw Data (HWInfo CSV)

Source: HWiNFO sensor logging.

Characteristics:

* wide table format (hundreds of columns)
* sampling interval ≈ 1–2 s
* each column = one sensor
* timestamps split into `Date` and `Time`

Example columns:

```
Date
Time
Core 0 VID
Core 0 Clock
Core 0 Effective Clock
Core 0 Power
CPU PPT
CPU (Tctl/Tdie)
...
```

No filtering occurs at this stage.

---

# Stage 2 — CSV Loading

Function: `load_hwinfo_csv()`

Operations performed:

1. Load CSV

```
pd.read_csv(... encoding="cp1252")
```

2. Construct timestamp

```
timestamp = Date + Time
```

3. Convert numeric sensor columns

```
pd.to_numeric(... errors="coerce")
```

4. Compute relative time

```
t = seconds since first sample
```

5. Add metadata

```
run_id
source_file
```

### Rows removed

Rows with invalid timestamps:

```
df = df.dropna(subset=["timestamp"])
```

No other filtering occurs here.

---

# Stage 3 — Core Dataset Construction

Function: `build_core_dataframe()`

This converts the wide table into a **long per-core dataset**.

Each row represents:

```
(timestamp, core)
```

rather than one row containing all cores.

Example transformation:

```
Wide format:
timestamp | core0_clock | core1_clock | core2_clock ...

↓

Long format:
timestamp | core | clock
timestamp | core | clock
timestamp | core | clock
```

---

## Sensors extracted per core

From `core_map`:

```
VID
Clock
Effective Clock
Temperature
Power
```

Global sensors:

```
CPU PPT
CPU (Tctl/Tdie)
```

---

## Derived features

Effective clock is averaged if multiple sensors exist:

```
eff_clock = mean(effective clock sensors)
```

Derived ratio:

```
eff_ratio = eff_clock / clock
```

Interpretation:

```
1.0  → fully active core
0.5  → half utilization
0.0  → idle
```

---

# Stage 4 — Parquet Output

Output format: **Parquet**

Each run becomes:

```
YYYY_MM_DD_HHMM.parquet
```

Rows represent:

```
(timestamp × cores)
```

Example:

```
88200 timestamps × 8 cores
→ 705,600 rows
```

Columns:

```
timestamp
t
run_id
source_file
core

vid
clock
eff_clock
eff_ratio

temp
power
ppt
cpu_temp
```

---

# Important: No performance filtering in pipeline

The processing pipeline is **lossless**.

No filtering is applied based on:

```
clock
power
eff_clock
VID
temperature
```

This ensures the parquet dataset contains **all telemetry samples**.

---

# Stage 5 — Analysis Filtering (Notebook)

Filtering occurs **only during analysis**.

Example filters used for V/F analysis:

```
eff_clock > 3000 MHz
power > 4.5 W
vid > 1.0 V
eff_ratio > 0.6
```

Purpose:

Remove

```
idle states
ramp transitions
low-load artifacts
scheduler noise
```

This isolates **stable boost states**.

---

# Why two clocks exist

Two frequency metrics are recorded:

```
clock        = instantaneous SMU boost frequency
eff_clock    = effective instruction retirement frequency
```

Interpretation:

```
clock      → silicon boost behavior
eff_clock  → workload utilization
```

Example:

```
clock = 5600 MHz
eff_clock = 3000 MHz
```

means the core boosted but was mostly idle.

---

# Resulting dataset

The final parquet dataset allows analysis of:

```
V/F curves
core quality differences
boost behavior
thermal limits
scheduler utilization
PBO tuning
```


  """