import argparse
import csv
from dataclasses import dataclass
from typing import List


@dataclass
class Trace:
    task: str
    start_ns: int
    duration_ns: int


DESCRIPTION = "Calculate throughput statistics for traces."


def read_traces_from_csv(file_name: str) -> List[Trace]:
    traces = []
    with open(file_name, "+r") as csv_file:
        trace_reader = csv.reader(csv_file)
        # Skip headers
        next(trace_reader)
        for row in trace_reader:
            traces.append(Trace(row[0], int(row[1]), int(row[2])))
    return traces


def calculate(traces: List[Trace], chunk_size_mb: int) -> dict[str, float]:
    counts = {}
    aggregrates = {}
    for trace in traces:
        if trace.task in aggregrates:
            aggregrates[trace.task] += trace.duration_ns
            counts[trace.task] += 1
        else:
            aggregrates[trace.task] = trace.duration_ns
            counts[trace.task] = 1
    throughputs = {}
    for task in aggregrates.keys():
        average_duration_ns = aggregrates[task] / counts[task]
        throughputs[task] = (chunk_size_mb / 1000) / (average_duration_ns / (10**9))
    return throughputs


def main():
    parser = argparse.ArgumentParser(prog="plot_trace", description=DESCRIPTION)
    parser.add_argument("-t", "--traces", required=True, type=str, help="CSV file containing traces")
    parser.add_argument("-s", "--chunk-size", required=True, type=int, help="Chunk size that was used")
    args = parser.parse_args()

    traces = read_traces_from_csv(args.traces)
    throughputs = calculate(traces, args.chunk_size)
    for task, throughput in throughputs.items():
        print(f"{task}: {throughput:.03f} GB/s")


if __name__ == "__main__":
    main()
