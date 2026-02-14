#!/usr/bin/env python3
"""Plot chunk size vs throughput for both machines.

Chunk sizes are bytes (powers of 2). The x-axis is log2-scaled with
human-readable tick labels (e.g., 16MB).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def format_bytes(value):
    """Format bytes using binary units (KB, MB, GB)."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    if size.is_integer():
        return f"{int(size)}{units[unit_index]}"
    return f"{size:.1f}{units[unit_index]}"


def main():
    # ---- Adjustable settings ----
    data_path = Path("chunksize-throughput.tsv")
    title = "Chunk Size vs Throughput"

    # Figure size: change the tuple to adjust width/height in inches.
    figsize = (12, 6)

    # Global font size: tweak this value to scale all text.
    plt.rcParams.update({"font.size": 11})

    # Separate font sizes for each text group.
    title_fontsize = 14
    axis_fontsize = 11
    legend_fontsize = 10

    # Colors and markers per machine.
    color_map = {
        "HX370": "#0072B2",
        "8845HS": "#E69F00",
    }
    marker_map = {
        "HX370": "o",
        "8845HS": "s",
    }
    # -----------------------------

    df = pd.read_csv(data_path, sep="\t", encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    # Normalize the chunk size column name if needed.
    if "chunk size" in df.columns and "chunk_size" not in df.columns:
        df = df.rename(columns={"chunk size": "chunk_size"})

    sizes = df["chunk_size"].to_numpy(dtype=float)
    machines = [c for c in df.columns if c != "chunk_size"]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for machine in machines:
        ax.plot(
            sizes,
            df[machine].to_numpy(dtype=float),
            label=machine,
            color=color_map.get(machine, "#555555"),
            marker=marker_map.get(machine, "o"),
            linewidth=1.8,
            markersize=6,
        )

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Chunk size", fontsize=axis_fontsize)
    ax.set_ylabel("Throughput", fontsize=axis_fontsize)
    ax.tick_params(axis="both", labelsize=axis_fontsize)

    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([format_bytes(v) for v in sizes])
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.legend(loc="upper left", fontsize=legend_fontsize)

    # To save instead of showing, uncomment the next line and adjust filename/dpi.
    plt.savefig("chunksize_throughput.png", dpi=200)
    # plt.show()


if __name__ == "__main__":
    main()
