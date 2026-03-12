# Ryzen 9700X V/F Analysis

Reusable tooling for Ryzen 9700X voltage/frequency analysis from HWInfo logs.

## Structure

- `src/vfanalysis/`: reusable analysis package
- `notebooks/vf_analysis.ipynb`: cleaned notebook workflow
- `scripts/quick_plots.py`: quick visual sanity check script
- `src/vf_analysis/`: compatibility import shims for older code

## Pipeline

The intended analysis flow is:

1. `load_processed_dataset` (or build from raw via `build_parquet_dataset`)
2. `add_all_features`
3. `filter_df`
4. metrics (`core_summary`, `dv_df_slope`, `boost_ridge`, ...)
5. plots (`plot_vf_hexbin_per_core`, `plot_vf_curves`, ...)

## Environment Variables

Set these in `.env`:

- `HWINFO_LOG_DIR`: raw HWInfo CSV files
- `VF_DATASET_DIR`: processed parquet dataset directory

## Scientific Scope and Uncertainty

This repository is intended for exploratory telemetry analysis. Interpretations that imply per-core Curve Optimizer stability should be treated as hypotheses unless validated by dedicated stability testing.

## Quick Start

```bash
uv run python scripts/quick_plots.py
```

## Linting

```bash
ruff check .
pylint src/vfanalysis scripts/quick_plots.py
```
