# Duplicate Functions Report

## Consolidated duplicates

| Duplicate logic | File paths | Overlap | Refactoring suggestion |
| --- | --- | --- | --- |
| Aggregate V/F curve extraction (`extract_vf_curve_all`, `extract_vf_curve_all_cores`) | `notebooks/vf_analysis.ipynb`, `src/vfanalysis/vfcurve.py` | Multiple notebook cells binned `vid` and took a clock quantile across all cores. | Keep using `vfanalysis.vfcurve.extract_aggregate_vf_curve` and avoid redefining aggregate curve helpers in notebooks. |
| Minimum-voltage frontier extraction (`extract_min_voltage_curve`) | `notebooks/vf_analysis.ipynb`, `src/vfanalysis/vfcurve.py` | Two notebook cells implemented the same clock-bin to low-quantile VID curve, with only optional-core handling differing. | Reuse `vfanalysis.vfcurve.extract_min_voltage_curve` for both aggregate and per-core views. |
| Power-law V/F model helper (`vf_model`) | `notebooks/vf_analysis.ipynb`, `src/vfanalysis/vfcurve.py` | Several notebook cells redefined the same `a * (v - vt)^b` function, with only clipping constants varying slightly. | Reuse `vfanalysis.vfcurve.vf_power_law`; if the clipping policy must differ later, expose it as a named parameter instead of redefining the function. |
| Residual summary math for fitted models | `src/vfanalysis/regressions.py`, `src/vfanalysis/ridge.py`, `src/vfanalysis/vfcurve.py` | The modules computed `r2`, residual standard deviation, and residual variance independently. | This is now shared through `vfanalysis._shared.residual_diagnostics`; keep new model helpers on the same path. |
| Per-core row selection | `src/vfanalysis/metrics.py`, `src/vfanalysis/regressions.py` | Multiple modules implemented the same optional `core` filtering behavior. | This is now shared through `vfanalysis._shared.subset_core`; use that helper for future per-core utilities. |
| Residual heatmap binning | `notebooks/vf_analysis.ipynb`, `src/vfanalysis/plots.py` | The notebook built the same `qcut` + median pivot used for residual surface plots. | Reuse `vfanalysis.plots.residual_heatmap` and keep notebook cells focused on rendering choices. |
| CO estimate smoothing (`smooth_curve`) | `notebooks/vf_analysis.ipynb`, `src/vfanalysis/workflow.py` | The notebook applied the same monotonic step-limited smoothing pattern to each grouped curve. | Reuse `vfanalysis.workflow.smooth_co_estimate_curve` for grouped CO estimate tables. |

## Remaining duplicates left unchanged

| Duplicate-looking logic | File paths | Why left unchanged | Suggestion |
| --- | --- | --- | --- |
| Notebook residual-surface experiments over `vid`, `v^2`, and `power` bins | `notebooks/vf_analysis.ipynb` | These cells appear to explore related ideas, but they do not clearly encode the same intended model or output table. Removing or merging them would risk changing exploratory behavior. | If these plots are still important, promote the repeated bin-and-pivot patterns into a dedicated exploratory module after deciding which formulation is canonical. |
| Repeated workload-style filtering expressions in late notebook cells | `notebooks/vf_analysis.ipynb` | Some later cells use slightly different inclusive thresholds, so it is unclear which differences are intentional versus incidental. | Prefer `vfanalysis.workflow.select_workload_dataset` for future edits and only consolidate more of these cells after the thresholds are reviewed. |
| Compatibility package shims mirroring `vfanalysis` | `src/vf_analysis/*.py`, `src/vfanalysis/*.py` | The duplicate import surface is intentional for backward compatibility. | Keep the shim package thin; do not add new logic under `src/vf_analysis` unless compatibility requires it. |
