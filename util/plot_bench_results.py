import argparse
import csv
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DESCRIPTION = "Plot benchmark results in TSV format as a bar chart."


ResultData = Dict[str, List[float]]


def read_results_from_tsv(file_name: str, skip_geomean=True) -> Tuple[List[str], ResultData]:
    benches = []
    results: ResultData = dict()
    with open(file_name, "+r") as tsv_file:
        results_reader = csv.DictReader(tsv_file, delimiter="\t")
        engines = results_reader.fieldnames[1:]
        for results_row in results_reader:
            if skip_geomean and results_row["Benchmark"] == "geomean":
                continue
            query_code = re.search(r"\((.*)\)", results_row["Benchmark"])
            if query_code is None:
                benches.append(results_row["Benchmark"])
            else:
                benches.append(query_code.group(1))
            for engine in engines:
                if engine not in results:
                    results[engine] = []
                measurement = results_row[engine].replace(",", ".")
                if measurement == "":
                        results[engine].append(np.nan)
                else:
                    results[engine].append(float(measurement))
    return benches, results


def save_bench_diagram(output_file_name: str, benches: List[str], results: ResultData):
    x = np.arange(len(benches))
    width = 0.2
    multiplier = 0

    patterns = ["//", "\\\\", None, None]
    plt.rcParams.update({'hatch.color': '0.8'})

    plt.style.use("tableau-colorblind10")
    _, ax = plt.subplots(figsize=(7.5,4))
    for i, (engine, measurements) in enumerate(results.items()):
        offset = width * multiplier
        ax.bar(x + offset, measurements, width, label=engine, hatch=patterns[i])
        multiplier += 1

    ax.set_facecolor('whitesmoke')
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle="dashed", axis="y")
    ax.set_ylabel("Throughput (GB/s)")
    ax.set_xticks(x + width, benches)
    legend = ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncols=4, fontsize="small")
    frame = legend.get_frame()
    frame.set_facecolor('whitesmoke')
    ax.set_ylim(0, 16)

    plt.savefig(output_file_name, dpi=80)

def main():
    parser = argparse.ArgumentParser(prog="plot_bench_results", description=DESCRIPTION)
    parser.add_argument("-r", "--results", required=True, type=str, help="TSV file containing benchmark results")
    parser.add_argument("-o", "--output", required=True, type=str, help="Output file name (PDF)")
    args = parser.parse_args()

    benches, results = read_results_from_tsv(args.results)
    save_bench_diagram(args.output, benches, results)


if __name__ == "__main__":
    main()
