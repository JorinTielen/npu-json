#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


QUERY_TASKS = ["automaton"]
NPU_WALL_TASK = "construct_combined_index_npu"
NPU_KERNEL_TASK = "npu_kernel_run"


def load_task_sums(trace_path: Path) -> dict[str, float]:
    sums: dict[str, float] = {}
    with trace_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row["task"].strip()
            duration_ns = float(row["duration_ns"])
            sums[task] = sums.get(task, 0.0) + duration_ns
    return sums


def workload_name(trace_path: Path) -> str:
    stem = trace_path.stem
    if stem.startswith("traces-"):
        return stem.split("traces-", 1)[1]
    return stem


def build_rows(trace_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(trace_dir.glob("traces-*.csv")):
        sums = load_task_sums(path)

        query_ns = sum(sums.get(task, 0.0) for task in QUERY_TASKS)
        escape_ns = sums.get("construct_escape_carry_index", 0.0)
        prep_ns = sums.get("prepare_kernel_input", 0.0)
        npu_wall_ns = sums.get(NPU_WALL_TASK, 0.0)
        npu_kernel_ns = sums.get(NPU_KERNEL_TASK, 0.0)
        read_ns = sums.get("read_kernel_output", 0.0)
        npu_component_ns = npu_kernel_ns if npu_kernel_ns > 0.0 else npu_wall_ns
        indexing_ns = escape_ns + prep_ns + npu_component_ns + read_ns
        total_ns = query_ns + indexing_ns

        if total_ns <= 0:
            continue

        rows.append(
            {
                "workload": workload_name(path),
                "query_pct": f"{query_ns / total_ns * 100.0:.4f}",
                "indexing_pct": f"{indexing_ns / total_ns * 100.0:.4f}",
                "escape_pct": f"{escape_ns / total_ns * 100.0:.4f}",
                "prep_pct": f"{prep_ns / total_ns * 100.0:.4f}",
                "npu_wall_pct": f"{npu_wall_ns / total_ns * 100.0:.4f}",
                "npu_kernel_pct": f"{npu_kernel_ns / total_ns * 100.0:.4f}",
                "npu_component_pct": f"{npu_component_ns / total_ns * 100.0:.4f}",
                "read_pct": f"{read_ns / total_ns * 100.0:.4f}",
                "query_ms": f"{query_ns / 1_000_000.0:.4f}",
                "indexing_ms": f"{indexing_ns / 1_000_000.0:.4f}",
                "npu_wall_ms": f"{npu_wall_ns / 1_000_000.0:.4f}",
                "npu_kernel_ms": f"{npu_kernel_ns / 1_000_000.0:.4f}",
                "total_ms": f"{total_ns / 1_000_000.0:.4f}",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trace_component_breakdown",
        description="Summarize query/indexing component shares from trace CSVs",
    )
    parser.add_argument("--trace-dir", type=Path, required=True, help="Directory with traces-*.csv files")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    rows = build_rows(args.trace_dir)
    if not rows:
        raise FileNotFoundError(f"No trace data found in {args.trace_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "workload",
            "query_pct",
            "indexing_pct",
            "escape_pct",
            "prep_pct",
            "npu_wall_pct",
            "npu_kernel_pct",
            "npu_component_pct",
            "read_pct",
            "query_ms",
            "indexing_ms",
            "npu_wall_ms",
            "npu_kernel_ms",
            "total_ms",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
