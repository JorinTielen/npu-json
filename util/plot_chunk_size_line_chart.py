import argparse
import csv
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DESCRIPTION = "Plot chunk size experiment results in TSV format as a line chart."


ResultData = Dict[str, List[float]]


def read_results_from_tsv(file_name: str, skip_geomean=True) -> Tuple[List[str], ResultData]:
    benches = []
    results: ResultData = dict()
    with open(file_name, "+r") as tsv_file:
        results_reader = csv.DictReader(tsv_file, delimiter="\t")
        engines = results_reader.fieldnames[1:]
        for results_row in results_reader:
            benches.append(results_row["Chunk size"])
            for engine in engines:
                if engine not in results:
                    results[engine] = []
                measurement = results_row[engine].replace(",", ".")
                if measurement == "":
                        results[engine].append(np.nan)
                else:
                    results[engine].append(float(measurement))
    return benches, results


def save_line_diagram(output_file_name: str, benches: List[str],
                      results: ResultData):
    x = np.arange(len(benches))

    print(results)
    print(benches)

    plt.style.use("tableau-colorblind10")
    _, ax = plt.subplots(figsize=(7.5,5))
    measurements = results["Throughput"]
    print(measurements)
    ax.plot(x, measurements, linestyle="dashed", marker="o")

    ax.set_facecolor('whitesmoke')
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle="dashed", axis="y")
    ax.set_ylabel("Throughput (GB/s)")
    ax.set_xlabel("Chunk size")
    ax.set_xticks(x, benches)
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_ylim(0, 12)

    plt.tight_layout()
    plt.savefig(output_file_name, dpi=80)

def main():
    parser = argparse.ArgumentParser(prog="plot_bench_results", description=DESCRIPTION)
    parser.add_argument("-r", "--results", required=True, type=str, help="TSV file containing chunk size results")
    parser.add_argument("-o", "--output", required=True, type=str, help="Output file name (PDF)")
    args = parser.parse_args()

    benches, results = read_results_from_tsv(args.results)
    save_line_diagram(args.output, benches, results)


if __name__ == "__main__":
    main()
