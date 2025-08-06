import argparse
import csv
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DESCRIPTION = "Plot utilization results in TSV format as a bar chart."


ResultData = Dict[str, List[float]]


def read_results_from_tsv(file_name: str) -> Tuple[List[str], ResultData]:
    benches = []
    results: ResultData = dict()
    with open(file_name, "+r") as tsv_file:
        results_reader = csv.DictReader(tsv_file, delimiter="\t")
        components = results_reader.fieldnames[1:]
        for results_row in results_reader:
            query_code = re.search(r"\((.*)\)", results_row["Benchmark"])
            if query_code is None:
                benches.append(results_row["Benchmark"])
            else:
                benches.append(query_code.group(1))
            for component in components:
                if component not in results:
                    results[component] = []
                measurement = results_row[component].replace(",", ".")
                if measurement == "":
                        results[component].append(np.nan)
                else:
                    results[component].append(float(measurement))
    return benches, results


def save_utilization_diagram(output_file_name: str, benches: List[str],
                             results: ResultData):
    x = np.arange(len(benches))
    width = 0.4
    multiplier = 0

    patterns = ["//", "\\\\", None, None]
    plt.rcParams.update({'hatch.color': '0.8'})

    bottom = np.zeros(len(benches))

    plt.style.use("tableau-colorblind10")
    _, ax = plt.subplots(figsize=(7.5,5))
    for i, (component, measurements) in enumerate(results.items()):
        ax.bar(x + width, measurements, width, label=component, hatch=patterns[i], bottom=bottom)
        bottom += measurements
        multiplier += 1

    ax.set_facecolor('whitesmoke')
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle="dashed", axis="y")
    ax.set_ylabel("Utilization %")
    ax.set_xticks(x + width, benches)
    legend = ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncols=4, fontsize="small")
    frame = legend.get_frame()
    frame.set_facecolor('whitesmoke')
    ax.set_ylim(0, 100)

    plt.savefig(output_file_name, dpi=80)


def main():
    parser = argparse.ArgumentParser(prog="plot_utilization", description=DESCRIPTION)
    parser.add_argument("-u", "--utilization", required=True, type=str, help="TSV file containing utilizations")
    parser.add_argument("-o", "--output", required=True, type=str, help="Output file name (PDF)")
    args = parser.parse_args()

    benches, results = read_results_from_tsv(args.utilization)
    save_utilization_diagram(args.output, benches, results)


if __name__ == "__main__":
    main()
