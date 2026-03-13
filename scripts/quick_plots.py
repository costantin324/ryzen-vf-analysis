"""Quick plotting entrypoint for the reusable V/F analysis pipeline."""

from __future__ import annotations

import matplotlib.pyplot as plt

from vfanalysis.metrics import core_summary
from vfanalysis.plots import plot_boost_ridge, plot_vf_curves, plot_vf_hexbin_per_core
from vfanalysis.workflow import load_analysis_frames


def main() -> None:
    """Run a short end-to-end analysis and display common plots.

    Args:
        None.

    Returns:
        None.

    Assumptions:
        The processed parquet dataset is available via the configured dataset
        directory.
    """

    frames = load_analysis_frames()

    print("Core summary (VF-filtered):")
    print(core_summary(frames.vf).round(4).to_string(index=False))

    plot_vf_hexbin_per_core(frames.vf, freq_col="eff_clock")
    plot_vf_curves(frames.vf)
    plot_boost_ridge(frames.base, power_min=10.0)

    plt.show()


if __name__ == "__main__":
    main()
