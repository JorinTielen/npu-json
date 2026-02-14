#!/usr/bin/env python3
"""Plot selectivity benchmark as normalized throughput percentages.

Each type is normalized to its 0% throughput (100%).
Adjustable plot settings are grouped near the top of main().
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize_to_zero_percent(df):
    """Return a DataFrame with throughput normalized to percent=0 per type."""
    baseline = df[df["percent"] == 0][["type", "throughput_gb_s"]].rename(
        columns={"throughput_gb_s": "baseline_gb_s"}
    )
    merged = df.merge(baseline, on="type", how="left")
    merged["norm_pct"] = merged["throughput_gb_s"] / merged["baseline_gb_s"] * 100.0
    return merged


def main():
    # ---- Adjustable settings ----
    data_path = Path("selectivity_benchmark_results.tsv")
    title = "Selectivity Benchmark with T1 Query: Normalized to 0%"

    # Figure size: change the tuple to adjust width/height in inches.
    figsize = (12, 6)

    # Global font size: tweak this value to scale all text.
    plt.rcParams.update({"font.size": 11})

    # Separate font sizes for each text group.
    title_fontsize = 18
    axis_fontsize = 15
    legend_fontsize = 15

    # Color-blind-friendly palette and distinct markers.
    colors = ["#0072B2", "#E69F00", "#009E73"]
    markers = ["o", "s", "^"]
    # -----------------------------

    df = pd.read_csv(data_path, sep="\t", encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    df = normalize_to_zero_percent(df)

    percent_values = np.sort(df["percent"].to_numpy())
    types = sorted(df["type"].unique())

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for idx, item_type in enumerate(types):
        series = df[df["type"] == item_type].set_index("percent")["norm_pct"]
        series = series.reindex(percent_values)
        ax.plot(
            percent_values,
            series.to_numpy(),
            label=item_type,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            linewidth=1.8,
            markersize=6,
        )

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Modification (%)", fontsize=axis_fontsize)
    ax.set_ylabel("Throughput (0% modification = 100%)", fontsize=axis_fontsize)
    ax.tick_params(axis="both", labelsize=axis_fontsize)
    ax.set_ylim(bottom=0, top=120)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.legend(loc="lower left", fontsize=legend_fontsize)

    # To save instead of showing, uncomment the next line and adjust filename/dpi.
    plt.savefig("selectivity_benchmark_normalized.png", dpi=250)
    # plt.show()


if __name__ == "__main__":
    main()
