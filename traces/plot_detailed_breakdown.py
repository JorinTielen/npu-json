#!/usr/bin/env python3
"""Plot detailed performance breakdown from trace CSVs.

For each workload, plot two adjacent bars:
- Query: sum of automaton time
- Indexing: stacked sum of construct_escape_carry_index, prepare_kernel_input,
  npu_kernel_run, read_kernel_output
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def workload_name(path: Path) -> str:
    """Extract workload name from filename like traces-t1.csv -> t1."""
    stem = path.stem
    if stem.startswith("traces-"):
        return stem.split("traces-", 1)[1]
    return stem


def main():
    # ---- Adjustable settings ----
    traces_dir = Path(".")
    title = "Detailed Performance Breakdown"

    # Workload order on x axis.
    workload_order = ["t1", "t2", "b1", "b2", "g1", "g2", "n1", "n2", "w1", "wi"]

    # Figure size: change the tuple to adjust width/height in inches.
    figsize = (13, 6)

    # Global font size: tweak this value to scale all text.
    plt.rcParams.update({"font.size": 11})
    plt.rcParams['hatch.linewidth'] = 0.3

    # Separate font sizes for each text group.
    title_fontsize = 19
    axis_fontsize = 15
    legend_fontsize = 15

    # Bar layout.
    bar_width = 0.25
    bar_gap = 0.04

    # Task definitions (keys must match trace task names).
    query_task = "automaton"
    indexing_tasks = [
        "prepare_kernel_input",
        "npu_kernel_run",
        "read_kernel_output",
    ]

    # Legend labels (editable without touching task names).
    legend_labels = {
        "query": "Query: automaton",
        "prepare_kernel_input": "Index: prepare input",
        "npu_kernel_run": "Index: NPU kernel",
        "read_kernel_output": "Index: read output",
    }

    # Colors and hatches (adjustable per bar/stack).
    query_color = "#65b8e7"
    query_hatch = "///"
    indexing_colors = {
        "prepare_kernel_input": "#7ed4bc",
        "npu_kernel_run": "#e9b286",
        "read_kernel_output": "#be8bd1",
    }
    indexing_hatches = {
        "prepare_kernel_input": "xx",
        "npu_kernel_run": "-/",
        "read_kernel_output": "\\\\",
    }
    # -----------------------------

    trace_files = sorted(traces_dir.glob("traces-*.csv"))
    if not trace_files:
        raise FileNotFoundError("No trace CSV files found (traces-*.csv)")

    rows = []
    for path in trace_files:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

        row = {"workload": workload_name(path)}
        row["query_ms"] = (
            df.loc[df["task"] == query_task, "duration_ns"].sum() / 1_000_000.0
        )
        for task in indexing_tasks:
            row[f"{task}_ms"] = (
                df.loc[df["task"] == task, "duration_ns"].sum() / 1_000_000.0
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["workload"] = pd.Categorical(
        summary["workload"], workload_order, ordered=True
    )
    summary = summary.sort_values("workload")

    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    query_x = x - (bar_width + bar_gap) / 2
    indexing_x = x + (bar_width + bar_gap) / 2

    query_bars = ax.bar(
        query_x,
        summary["query_ms"],
        width=bar_width,
        color=query_color,
        edgecolor="#1a1a1a",
        linewidth=0.2,
        hatch=query_hatch,
        label=legend_labels["query"],
    )

    bottoms = np.zeros(len(summary))
    stack_handles = []
    for task in indexing_tasks:
        values = summary[f"{task}_ms"].to_numpy(dtype=float)
        bars = ax.bar(
            indexing_x,
            values,
            width=bar_width,
            bottom=bottoms,
            color=indexing_colors[task],
            edgecolor="#1a1a1a",
            linewidth=0.2,
            hatch=indexing_hatches[task],
            label=legend_labels[task],
        )
        stack_handles.append(bars)
        bottoms += values

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Workload", fontsize=axis_fontsize)
    ax.set_ylabel("Execution time (ms)", fontsize=axis_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels([task.upper() for task in summary["workload"].tolist()], fontsize=axis_fontsize)
    ax.tick_params(axis="y", labelsize=axis_fontsize)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.4)

    legend_handles = [query_bars] + stack_handles
    legend_texts = [legend_labels["query"]] + [legend_labels[t] for t in indexing_tasks]
    ax.legend(legend_handles, legend_texts, loc="upper right", fontsize=legend_fontsize)

    # To save instead of showing, uncomment the next line and adjust filename/dpi.
    plt.savefig("detailed_breakdown.png", dpi=200)
    # plt.show()


if __name__ == "__main__":
    main()
