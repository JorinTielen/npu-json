#!/usr/bin/env python3

import argparse
import csv
import math
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


WORKLOADS = {
    "T1": ("twitter.json", "$[*].user.lang"),
    "T2": ("twitter.json", "$[*].entities.urls[*].url"),
    "B1": ("bestbuy.json", "$.products[*].categoryPath[1:3].id"),
    "B2": ("bestbuy.json", "$.products[*].videoChapters[*].chapter"),
    "G1": ("googlemaps.json", "$[*].routes[*].legs[*].steps[*].distance.text"),
    "G2": ("googlemaps.json", "$[*].available_travel_modes"),
    "N1": ("nspl.json", "$.meta.view.columns[*].name"),
    "N2": ("nspl.json", "$.data[*][*][*]"),
    "W1": ("walmart.json", "$.items[*].bestMarketplacePrice.price"),
    "Wi": ("wikipedia.json", "$[*].claims.P150[*].mainsnak.property"),
}


TIME_RE = re.compile(r"performed query on average in\s+([0-9.]+)s")
THROUGHPUT_RE = re.compile(r"GB/s:\s+([0-9.]+)")


@dataclass
class RunResult:
    workload: str
    run: int
    time_s: float
    gbps: float


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of empty list")
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * p
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_values[low]
    weight = pos - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def run_benchmark_once(binary: Path, dataset_path: Path, query: str) -> tuple[float, float]:
    cmd = [str(binary), str(dataset_path), query, "--bench"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")

    if proc.returncode != 0:
        raise RuntimeError(
            f"Benchmark command failed (code {proc.returncode}): {' '.join(cmd)}\n{output}"
        )

    time_match = TIME_RE.search(output)
    throughput_match = THROUGHPUT_RE.search(output)
    if time_match is None or throughput_match is None:
        raise RuntimeError(
            f"Could not parse benchmark output for command: {' '.join(cmd)}\n{output}"
        )

    return float(time_match.group(1)), float(throughput_match.group(1))


def parse_workloads(raw: list[str]) -> list[str]:
    if len(raw) == 1 and raw[0].lower() == "all":
        return list(WORKLOADS.keys())

    unknown = [w for w in raw if w not in WORKLOADS]
    if unknown:
        valid = ", ".join(WORKLOADS.keys())
        raise ValueError(f"Unknown workload(s): {', '.join(unknown)}. Valid: {valid}, all")

    return raw


def print_summary(label: str, rows: list[RunResult], workloads: list[str]) -> None:
    if label:
        print(f"\n[{label}] throughput distribution")
    else:
        print("\nThroughput distribution")

    print("workload,runs,mean_gbps,median_gbps,p10_gbps,p90_gbps,stdev_gbps,min_gbps,max_gbps")
    for workload in workloads:
        vals = [r.gbps for r in rows if r.workload == workload]
        if not vals:
            continue
        stdev = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(
            f"{workload},{len(vals)},{statistics.mean(vals):.4f},{statistics.median(vals):.4f},"
            f"{percentile(vals, 0.10):.4f},{percentile(vals, 0.90):.4f},{stdev:.4f},"
            f"{min(vals):.4f},{max(vals):.4f}"
        )


def write_runs_csv(path: Path, label: str, rows: list[RunResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "workload", "run", "time_s", "gbps"])
        for r in rows:
            writer.writerow([label, r.workload, r.run, f"{r.time_s:.9f}", f"{r.gbps:.9f}"])


def write_summary_csv(path: Path, label: str, rows: list[RunResult], workloads: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "workload",
                "runs",
                "mean_gbps",
                "median_gbps",
                "p10_gbps",
                "p90_gbps",
                "stdev_gbps",
                "min_gbps",
                "max_gbps",
            ]
        )
        for workload in workloads:
            vals = [r.gbps for r in rows if r.workload == workload]
            if not vals:
                continue
            stdev = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            writer.writerow(
                [
                    label,
                    workload,
                    len(vals),
                    f"{statistics.mean(vals):.9f}",
                    f"{statistics.median(vals):.9f}",
                    f"{percentile(vals, 0.10):.9f}",
                    f"{percentile(vals, 0.90):.9f}",
                    f"{stdev:.9f}",
                    f"{min(vals):.9f}",
                    f"{max(vals):.9f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bench_distribution",
        description="Run benchmarks repeatedly and report throughput distribution.",
    )
    parser.add_argument("--binary", type=Path, default=Path("build/nj"), help="Path to nj binary")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets"),
        help="Directory containing benchmark JSON datasets",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["all"],
        help="Workload keys to run (T1 T2 ... Wi) or 'all'",
    )
    parser.add_argument("--runs", type=int, default=5, help="Measured runs per workload")
    parser.add_argument("--warmups", type=int, default=0, help="Warmup runs per workload")
    parser.add_argument("--cooldown-ms", type=int, default=0, help="Sleep between measured runs")
    parser.add_argument("--label", type=str, default="", help="Optional label stored in CSV output")
    parser.add_argument("--runs-csv", type=Path, default=None, help="Optional CSV file for per-run data")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional CSV file for per-workload summary",
    )
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be > 0")
    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0")
    if args.cooldown_ms < 0:
        raise ValueError("--cooldown-ms must be >= 0")

    workload_keys = parse_workloads(args.workloads)

    if not args.binary.exists():
        raise FileNotFoundError(f"Binary not found: {args.binary}")

    rows: list[RunResult] = []
    for workload in workload_keys:
        dataset, query = WORKLOADS[workload]
        dataset_path = args.dataset_dir / dataset
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        for _ in range(args.warmups):
            _ = run_benchmark_once(args.binary, dataset_path, query)

        for run in range(1, args.runs + 1):
            time_s, gbps = run_benchmark_once(args.binary, dataset_path, query)
            rows.append(RunResult(workload=workload, run=run, time_s=time_s, gbps=gbps))
            print(f"{workload} run {run}/{args.runs}: {gbps:.4f} GB/s ({time_s:.6f}s)")
            if args.cooldown_ms > 0 and run < args.runs:
                time.sleep(args.cooldown_ms / 1000.0)

    print_summary(args.label, rows, workload_keys)

    if args.runs_csv is not None:
        args.runs_csv.parent.mkdir(parents=True, exist_ok=True)
        write_runs_csv(args.runs_csv, args.label, rows)
        print(f"Wrote per-run CSV: {args.runs_csv}")

    if args.summary_csv is not None:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        write_summary_csv(args.summary_csv, args.label, rows, workload_keys)
        print(f"Wrote summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()
