#!/usr/bin/env python3
"""Plot Query vs Indexing utilization from trace CSVs.

Each workload is a stacked bar where Query + Indexing = 100%.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


QUERY_TASKS = ["automaton"]
INDEXING_TASKS = [
    "read_kernel_output",
    "npu_kernel_run",
    "prepare_kernel_input",
    "construct_escape_carry_index",
]


def workload_name(path: Path) -> str:
    """Extract workload name from filename like traces-t1.csv -> t1."""
    stem = path.stem
    if stem.startswith("traces-"):
        return stem.split("traces-", 1)[1]
    return stem


def main():
    # ---- Adjustable settings ----
    traces_dir = Path(".")
    title = "Query vs Indexing Utilization"
    
    plt.rcParams['hatch.linewidth'] = 0.3

    # Figure size: change the tuple to adjust width/height in inches.
    figsize = (12, 6)

    # Global font size: tweak this value to scale all text.
    plt.rcParams.update({"font.size": 11})

    # Separate font sizes for each text group.
    title_fontsize = 17
    axis_fontsize = 15
    legend_fontsize = 15
    bar_width = 0.5

    # Colors for stacked bars.
    query_color = "#f5c791"
    indexing_color = "#79a9c5"
    # -----------------------------

    trace_files = sorted(traces_dir.glob("traces-*.csv"))
    if not trace_files:
        raise FileNotFoundError("No trace CSV files found (traces-*.csv)")

    rows = []
    for path in trace_files:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

        query_time = df[df["task"].isin(QUERY_TASKS)]["duration_ns"].sum()
        indexing_time = df[df["task"].isin(INDEXING_TASKS)]["duration_ns"].sum()

        total = query_time + indexing_time
        if total == 0:
            query_pct = 0.0
            indexing_pct = 0.0
        else:
            query_pct = query_time / total * 100.0
            indexing_pct = indexing_time / total * 100.0

        rows.append(
            {
                "workload": workload_name(path),
                "query_pct": query_pct,
                "indexing_pct": indexing_pct,
            }
        )

    order = ["t1", "t2", "b1", "b2", "g1", "g2", "n1", "n2", "w1", "wi"]
    summary = pd.DataFrame(rows)
    summary["workload"] = pd.Categorical(summary["workload"], order, ordered=True)
    summary = summary.sort_values("workload")

    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    query_bars = ax.bar(
        x,
        summary["query_pct"],
        label="Query",
        color=query_color,
        edgecolor="#1a1a1a",
        linewidth=0.3,
        hatch="//",
        width=bar_width,
    )
    indexing_bars = ax.bar(
        x,
        summary["indexing_pct"],
        bottom=summary["query_pct"],
        label="Indexing",
        color=indexing_color,
        edgecolor="#1a1a1a",
        linewidth=0.3,
        hatch="\\\\",
        width=bar_width,
    )

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Workload", fontsize=axis_fontsize)
    ax.set_ylabel("Utilization (%)", fontsize=axis_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels([task.upper() for task in summary["workload"].tolist()], fontsize=axis_fontsize)
    ax.tick_params(axis="y", labelsize=axis_fontsize)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.legend(
        [indexing_bars, query_bars],
        ["Indexing", "Query"],
        loc="lower left",
        fontsize=legend_fontsize,
    )

    # To save instead of showing, uncomment the next line and adjust filename/dpi.
    plt.savefig("query_indexing_utilization.png", dpi=200)
    # plt.show()


if __name__ == "__main__":
    main()
