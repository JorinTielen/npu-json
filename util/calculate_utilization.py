import argparse
import csv
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt

from plot_trace import Trace, read_traces_from_csv


DESCRIPTION = "Calculate the NPU and CPU utilization of a trace."


NPU_TRACE_TYPES = ["construct_combined_index_npu"]
CPU_TRACE_TYPES = ["automaton"]


def print_utilization(traces: List[Trace]):
    total_duration = 0
    cpu_duration = 0
    npu_duration = 0
    for trace in traces:
        if trace.task not in NPU_TRACE_TYPES + CPU_TRACE_TYPES:
            continue
        duration = float(trace.duration_ns) / 1000 / 1000 / 1000
        total_duration += duration
        if trace.task in CPU_TRACE_TYPES:
            cpu_duration += duration
        elif trace.task in NPU_TRACE_TYPES:
            npu_duration += duration
    cpu_percent = cpu_duration / total_duration * 100
    npu_percent = npu_duration / total_duration * 100
    print("Utilization:")
    print(f"CPU: {cpu_percent:.3f}% ({cpu_duration:.3f}s)")
    print(f"NPU: {npu_percent:.3f}% ({npu_duration:.3f}s)")


def main():
    parser = argparse.ArgumentParser(prog="plot_trace", description=DESCRIPTION)
    parser.add_argument("-t", "--traces", required=True, type=str, help="CSV file containing traces")
    args = parser.parse_args()

    traces = read_traces_from_csv(args.traces)
    print_utilization(traces)


if __name__ == "__main__":
    main()
