# Ryzen 9700X V/F Analysis

Reusable tooling for Ryzen 9700X voltage/frequency analysis from HWInfo logs and processed parquet datasets.

## Repository Structure

- `src/vfanalysis/`
  - Main reusable analysis package.
- `src/vfanalysis/io.py`
  - Raw HWInfo CSV ingestion, environment-path loading, sensor detection, long-format dataframe construction, parquet shard creation, and parquet aggregation.
- `src/vfanalysis/features.py`
  - Deterministic row-wise feature engineering such as effective-clock ratios and voltage/power efficiency helpers.
- `src/vfanalysis/filters.py`
  - Reusable threshold-based cleanup and V/F-focused filtering.
- `src/vfanalysis/metrics.py`
  - Aggregate metrics and per-core summaries used by the notebook and scripts.
- `src/vfanalysis/plots.py`
  - Shared plotting helpers for ridge curves, per-core fits, telemetry views, and residual surfaces.
- `src/vfanalysis/regressions.py`
  - Linear and power-law regression helpers plus notebook-oriented model summaries.
- `src/vfanalysis/vfcurve.py`
  - V/F curve extraction, min-voltage frontiers, and power-law fit helpers.
- `src/vfanalysis/workflow.py`
  - High-level notebook workflow helpers that combine loading, feature engineering, filtering, and display formatting.
- `src/vf_analysis/`
  - Compatibility import shims for older module paths.
- `notebooks/vf_analysis.ipynb`
  - Main exploratory notebook built on the reusable package.
- `scripts/quick_plots.py`
  - Short end-to-end smoke script for loading the processed dataset and rendering common plots.
- `README.md`
  - High-level repository and pipeline documentation.
- `.env`
  - Local environment configuration for raw CSV directories and processed parquet output. This file is gitignored and must be created locally per machine.

## Environment Configuration

The package loads environment variables from a repo-root `.env` file via `python-dotenv`.

Expected variables:

- `HWINFO_LOG_DIR`
  - Primary raw HWInfo CSV directory.
- `HWINFO_LOG_DIR_CO30ALLCORE`
  - Additional raw HWInfo CSV directory for the CO-30 all-core undervolting telemetry.
- `VF_DATASET_DIR`
  - Output directory for processed parquet shards.

`src/vfanalysis/io.py` now supports one primary raw-log directory plus any additional directories whose variable names start with `HWINFO_LOG_DIR_`.

## Internal Data Pipeline

The intended processing pipeline is:

1. Configure raw and processed paths in `.env`.
2. Read raw HWInfo CSV logs from all configured log directories.
3. Parse timestamps and coerce sensor columns to numeric values.
4. Detect per-core columns such as VID, core clock, effective clocks, per-core temperature, and per-core power.
5. Extract global CPU, package, SoC, GPU, and frame-timing telemetry columns when present.
6. Convert each wide HWInfo log into a long-format per-core dataframe.
7. Write one parquet shard per raw CSV file.
8. Load parquet shards for analysis, add deterministic row-wise features, then apply notebook-specific filters.
9. Run summary metrics, model fitting, and plotting helpers on the filtered frames.

## Processed Dataset Shape

The processed parquet schema is intentionally long-format and per-core. Each row represents one timestamp/core observation.

Core-specific fields include:

- `core`
- `vid`
- `clock`
- `eff_clock`
- `temp`
- `power`
- `eff_ratio`

Run-level metadata includes:

- `timestamp`
- `t`
- `run_id`
- `log_group`
- `source_file`

Global CPU telemetry that is duplicated across core rows for the same timestamp includes:

- `cpu_package_power`
- `cpu_ppt` and legacy alias `ppt`
- `cpu_soc_power`
- `cpu_soc_misc_power`
- `cpu_ppt_limit`
- `cpu_tdc_limit`
- `cpu_edc_limit`
- `cpu_ppt_fast_limit`
- `thermal_limit`
- `cpu_core_current`
- `soc_current`
- `cpu_tdc`
- `cpu_edc`
- `cpu_temp`

Global GPU and frame telemetry that is duplicated across core rows for the same timestamp includes:

- `gpu_temp`
- `gpu_memory_temp`
- `gpu_core_voltage`
- `gpu_clock`
- `gpu_memory_clock`
- `gpu_utilization`
- `framerate`
- `framerate_presented`
- `framerate_displayed`
- `frame_time`
- `gpu_busy`
- `gpu_wait`
- `cpu_busy`
- `cpu_wait`

This keeps the existing analysis workflow simple because one dataframe still contains the per-core CPU telemetry and the system-level GPU/frame context. If storage duplication becomes a problem later, the next refinement would be to split out a separate system-telemetry parquet keyed by timestamp/run.

## Parquet Helpers

Key ingestion helpers in `src/vfanalysis/io.py`:

- `build_parquet_dataset`
  - Build one processed parquet shard per raw CSV log.
- `load_processed_dataset`
  - Concatenate processed parquet shards into one dataframe.
- `aggregate_parquet_dataset`
  - Write one aggregated parquet file from all processed shards.

The compatibility entrypoint `src/vf_analysis/pipeline1.py` exposes both `build_dataset()` and `aggregate_dataset()` for older usage patterns.

## Analysis Flow

The standard notebook/script flow is:

1. `load_processed_dataset` or `load_analysis_frames`
2. `add_all_features`
3. `filter_df` / `prepare_analysis_frames`
4. metrics such as `core_summary`, `boost_ridge`, or regression summaries
5. plotting helpers such as `plot_vf_hexbin_per_core`, `plot_vf_curves`, and the per-core power-law fit panels

## Quick Start

Build parquet shards from raw logs:

```bash
uv run python -c "from vfanalysis.io import build_parquet_dataset; print(build_parquet_dataset())"
```

Build one aggregate parquet file from processed shards:

```bash
uv run python -c "from vfanalysis.io import aggregate_parquet_dataset; print(aggregate_parquet_dataset())"
```

Run the quick plot smoke script:

```bash
uv run python scripts/quick_plots.py
```

## Development

Runtime dependencies and developer tooling are split in `pyproject.toml`.

- Main scientific/runtime packages stay in `project.dependencies`.
- `jupyterlab`, `ruff`, `pyright`, and `pylint` now live in the `dev` dependency group.

That means a smaller runtime-only environment is available with:

```bash
uv sync --no-dev
```

And the full development environment is restored with:

```bash
uv sync
```

## Scientific Scope and Uncertainty

This repository is intended for exploratory telemetry analysis. Interpretations that imply per-core Curve Optimizer stability should be treated as hypotheses unless validated by dedicated stability testing.

