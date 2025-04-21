import argparse
import csv
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt


@dataclass
class Trace:
    task: str
    start_ns: int
    duration_ns: int


DESCRIPTION = "Plot the traces exported as CSV from the tracer utility as a Gantt chart."


def read_traces_from_csv(file_name: str) -> List[Trace]:
    traces = []
    with open(file_name, "+r") as csv_file:
        trace_reader = csv.reader(csv_file)
        # Skip headers
        next(trace_reader)
        for row in trace_reader:
            traces.append(Trace(row[0], int(row[1]), int(row[2])))
    return traces


def save_schedule_diagram(output_file_name: str, traces: List[Trace]):
    _, ax = plt.subplots(1, figsize=(10, 5))
    for trace in traces:
        ax.barh(
            y=trace.task,
            left=float(trace.start_ns) / 1000 / 1000 / 1000,
            width=float(trace.duration_ns) / 1000 / 1000 / 1000)
    ax.grid(True, linestyle="--", axis="y")
    ax.set_axisbelow(True)
    plt.xlabel("Execution time (s)", )
    plt.ylabel("Task")
    plt.yticks(list({ trace.task for trace in traces }))
    plt.tight_layout()
    plt.savefig(output_file_name, dpi=250)


def main():
    parser = argparse.ArgumentParser(prog="plot_trace", description=DESCRIPTION)
    parser.add_argument("-t", "--traces", required=True, type=str, help="CSV file containing traces")
    parser.add_argument("-o", "--output", required=True, type=str, help="Output file name (PDF)")
    args = parser.parse_args()

    traces = read_traces_from_csv(args.traces)
    save_schedule_diagram(args.output, traces)


if __name__ == "__main__":
    main()
