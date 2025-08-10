import argparse
import csv
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DESCRIPTION = "Plot experiment results in TSV format as a line chart."


ResultData = Dict[str, List[float]]


def read_results_from_tsv(file_name: str, experiment_name: str) -> Tuple[List[str], ResultData]:
    benches = []
    results: ResultData = dict()
    with open(file_name, "+r") as tsv_file:
        results_reader = csv.DictReader(tsv_file, delimiter="\t")
        engines = results_reader.fieldnames[1:]
        for results_row in results_reader:
            benches.append(results_row[experiment_name])
            for engine in engines:
                if engine not in results:
                    results[engine] = []
                measurement = results_row[engine].replace(",", ".")
                if measurement == "":
                        results[engine].append(np.nan)
                else:
                    results[engine].append(float(measurement))
    return benches, results


def save_line_diagram(output_file_name: str, experiment_name: str,
                      benches: List[str], results: ResultData):
    x = np.arange(len(benches))
    width = 0.2
    multiplier = 0

    plt.style.use("tableau-colorblind10")
    _, ax = plt.subplots(figsize=(7.5,5))
    for i, (engine, measurements) in enumerate(results.items()):
        offset = width * multiplier
        ax.plot(x, measurements, linestyle="dashed", label=engine, marker="o")
        multiplier += 1

    ax.set_facecolor('whitesmoke')
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle="dashed", axis="y")
    ax.set_ylabel("Throughput (GB/s)")
    ax.set_xlabel(experiment_name)
    ax.set_xticks(x, benches)
    ax.tick_params(axis='x', labelrotation=45)
    legend = ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncols=4, fontsize="small")
    frame = legend.get_frame()
    frame.set_facecolor('whitesmoke')
    ax.set_ylim(0, 12)

    plt.tight_layout()
    plt.savefig(output_file_name, dpi=80)

def main():
    parser = argparse.ArgumentParser(prog="plot_bench_results", description=DESCRIPTION)
    parser.add_argument("-r", "--results", required=True, type=str, help="TSV file containing chunk size results")
    parser.add_argument("-n", "--name", required=True, type=str, help="Experiment name")
    parser.add_argument("-o", "--output", required=True, type=str, help="Output file name (PDF)")
    args = parser.parse_args()

    benches, results = read_results_from_tsv(args.results, args.name)
    save_line_diagram(args.output, args.name, benches, results)


if __name__ == "__main__":
    main()
