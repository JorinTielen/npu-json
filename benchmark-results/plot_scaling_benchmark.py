#!/usr/bin/env python3
"""Plot scaling benchmark as normalized throughput percentages.

Each query is normalized to its baseline size throughput (100%).
Adjustable plot settings are grouped near the top of main().
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize_to_baseline(df, baseline_size=1024):
    """Return a DataFrame with throughput normalized to baseline size."""
    baseline = df[df["size_mb"] == baseline_size][["query", "throughput_gb_s"]].rename(
        columns={"throughput_gb_s": "baseline_gb_s"}
    )
    merged = df.merge(baseline, on="query", how="left")
    merged["norm_pct"] = merged["throughput_gb_s"] / merged["baseline_gb_s"] * 100.0
    return merged


def main():
    # ---- Adjustable settings ----
    data_path = Path("scaling_benchmark_results.tsv")
    title = "Scalability on JSON Document size: Normalized to 4096MB"

    # Column names in the input TSV (edit if your file uses different headers).
    column_map = {
        "query": "query",
        "size_mb": "size_mb",
        "throughput_gb_s": "throughput_gb_s",
    }

    # Figure size: change the tuple to adjust width/height in inches.
    figsize = (14, 5)

    # Global font size: tweak this value to scale all text.
    plt.rcParams.update({"font.size": 11})

    # Separate font sizes for each text group.
    title_fontsize = 18
    axis_fontsize = 15
    legend_fontsize = 15
    value_fontsize = 9

    # Baseline size for normalization (MB).
    baseline_size = 4096

    # Color-blind-friendly palette and distinct markers.
    colors = [
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#D55E00",
        "#CC79A7",
        "#56B4E9",
        "#F0E442",
        "#000000",
        "#7F7F7F",
        "#332288",
    ]
    markers = ["o", "s", "^", "D", "P", "X", "v", "*", ">", "<"]
    # -----------------------------

    df = pd.read_csv(data_path, sep="\t", encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    # Normalize columns to expected names (case-insensitive match).
    lower_map = {str(c).lower(): c for c in df.columns}
    rename_map = {}
    missing = []
    for target, source in column_map.items():
        key = source.lower()
        if key not in lower_map:
            missing.append(source)
            continue
        rename_map[lower_map[key]] = target

    if missing:
        # Fallback: assume first 3 columns are query, size_mb, throughput_gb_s.
        if len(df.columns) >= 3:
            df = df.rename(
                columns={
                    str(df.columns[0]): "query",
                    str(df.columns[1]): "size_mb",
                    str(df.columns[2]): "throughput_gb_s",
                }
            )
        else:
            available = ", ".join(df.columns)
            missing_list = ", ".join(missing)
            raise ValueError(
                "Missing required columns: "
                f"{missing_list}. Available columns: {available}"
            )
    else:
        df = df.rename(columns=rename_map)

    df = normalize_to_baseline(df, baseline_size=baseline_size)

    # Prepare data in wide form for plotting.
    sizes = np.sort(df["size_mb"].to_numpy())
    queries = sorted(df["query"].unique())

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for idx, query in enumerate(queries):
        series = df[df["query"] == query].set_index("size_mb")["norm_pct"]
        series = series.reindex(sizes)
        ax.plot(
            sizes,
            series.to_numpy(),
            label=query,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            linewidth=1.8,
            markersize=6,
        )

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Input size (MB)", fontsize=axis_fontsize)
    ax.set_ylabel(
        f"Throughput ({baseline_size}MB = 100%)",
        fontsize=axis_fontsize,
    )
    ax.tick_params(axis="both", labelsize=axis_fontsize)
    ax.set_ylim(bottom=40, top=120)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.legend(loc="lower left", fontsize=legend_fontsize, ncol=2)

    # To save instead of showing, uncomment the next line and adjust filename/dpi.
    plt.savefig("scaling_benchmark_normalized.png", dpi=200)
    # plt.show()


if __name__ == "__main__":
    main()
