"""Quick plotting entrypoint for reusable V/F analysis pipeline."""

from __future__ import annotations

import matplotlib.pyplot as plt

from vfanalysis.features import add_all_features
from vfanalysis.filters import base_filter_config, filter_df, vf_filter_config
from vfanalysis.io import load_processed_dataset
from vfanalysis.metrics import core_summary
from vfanalysis.plots import plot_boost_ridge, plot_vf_curves, plot_vf_hexbin_per_core


def main() -> None:
    """Run a short end-to-end analysis and display common plots."""

    df = load_processed_dataset()
    df = add_all_features(df)

    df_base = filter_df(df, base_filter_config())
    df_vf = filter_df(df_base, vf_filter_config())

    print("Core summary (VF-filtered):")
    print(core_summary(df_vf).round(4).to_string(index=False))

    plot_vf_hexbin_per_core(df_vf, freq_col="eff_clock")
    plot_vf_curves(df_vf)
    plot_boost_ridge(df_base, power_min=10.0)

    plt.show()


if __name__ == "__main__":
    main()
