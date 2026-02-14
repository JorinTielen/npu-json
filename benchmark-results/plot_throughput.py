#!/usr/bin/env python3
"""Plot throughput benchmarks from throughput-8845.tsv.

Adjustable settings are grouped near the top of main().
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def annotate_bars(ax, bars, precision=2, fontsize=None, scales=None):
    assert scales is None or len(scales) == len(bars), "Length of values must match number of bars"
    if scales is None:
        scales = [f"{bar.get_height():.{precision}f}" for bar in bars]
    """Place value labels on top of each bar."""
    for bar, value in zip(bars, scales):
        height = bar.get_height()
        if math.isnan(height):
            continue
        ax.annotate(
            value,
            (bar.get_x() + bar.get_width() / 2, height),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def plot_grouped_bars(
    ax,
    df,
    title,
    engines,
    color_map,
    hatch_map,
    precision=2,
    title_fontsize=None,
    axis_fontsize=None,
    legend_fontsize=None,
    value_fontsize=None,
):
    benchmarks = df["Benchmark"].tolist()
    x = np.arange(len(benchmarks))
    width = 0.27
    
    base_values = df[engines[0]].to_numpy(dtype=float)

    for idx, engine in enumerate(engines):
        values = df[engine].to_numpy(dtype=float)
        bars = ax.bar(
            x + (idx - 1) * width,
            values,
            width,
            label=engine,
            color=color_map[engine],
            edgecolor="#1a1a1a",
            linewidth=0.5,
            hatch=hatch_map[engine],
        )
        annotate_bars(ax, bars, precision=precision, fontsize=value_fontsize, scales=[f"{value:.{precision}f}x" for value in base_values/values])

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=axis_fontsize)
    ax.set_ylabel("Throughput (GB/s)", fontsize=axis_fontsize)
    ax.tick_params(axis="y", labelsize=axis_fontsize)
    ax.margins(y=0.15)
    ax.legend(loc="upper right", fontsize=legend_fontsize)


def main():
    # ---- Adjustable settings ----
    data_path = Path("throughput-8845.tsv")
    title_1 = "Query Throughput on 8845HS against JSONPath Engines"
    title_2 = "Query Throughput on 8845HS against non-JSONPath Engines"

    # Figure size: change the tuple to adjust width/height in inches.
    figsize = (14, 8)

    # Global font size: tweak this value to scale all text.
    plt.rcParams.update({"font.size": 10})
    plt.rcParams['hatch.linewidth'] = 0.3

    # Separate font sizes for each text group.
    title_fontsize = 18
    axis_fontsize = 13
    legend_fontsize = 15
    value_fontsize = 9

    # Precision of value labels on bars.
    value_precision = 2

    # Color-blind-friendly palette with distinct hatches.
    color_map = {
        "npu-json": "#89c9ee",
        "jsonski": "#f1cf84",
        "rsonpath": "#ff9191",
        "simdjson": "#feb68f",
        "pison": "#a5ebd2",
    }
    hatch_map = {
        "npu-json": "///",
        "jsonski": "\\\\",
        "rsonpath": "xx",
        "simdjson": "..",
        "pison": "++",
    }
    # -----------------------------

    df = pd.read_csv(data_path, sep="\t")

    # Normalize column names (strip accidental whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Ensure missing cells are treated as NaN
    df = df.replace(r"^\s*$", np.nan, regex=True)

    fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)

    plot_grouped_bars(
        axes[0],
        df,
        title_1,
        engines=["npu-json", "jsonski", "rsonpath"],
        color_map=color_map,
        hatch_map=hatch_map,
        precision=value_precision,
        title_fontsize=title_fontsize,
        axis_fontsize=axis_fontsize,
        legend_fontsize=legend_fontsize,
        value_fontsize=value_fontsize,
    )

    plot_grouped_bars(
        axes[1],
        df,
        title_2,
        engines=["npu-json", "simdjson", "pison"],
        color_map=color_map,
        hatch_map=hatch_map,
        precision=value_precision,
        title_fontsize=title_fontsize,
        axis_fontsize=axis_fontsize,
        legend_fontsize=legend_fontsize,
        value_fontsize=value_fontsize,
    )

    # To save instead of showing, uncomment the next line and adjust filename/dpi.
    plt.savefig("throughput-8845.png", dpi=250)
    # plt.show()


if __name__ == "__main__":
    main()
