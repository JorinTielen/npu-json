import argparse
import json
import random


DESCRIPTION = "Generate an artificial dataset of a particular size based with statuses from the twitter dataset"


def main():
    parser = argparse.ArgumentParser(prog="plot_bench_results", description=DESCRIPTION)
    parser.add_argument("-i", "--id", required=True, type=bool, help="Whether to add incrementing IDs to statuses")
    parser.add_argument("-s", "--size", required=True, type=int, help="Size of artificial dataset (KB)")
    parser.add_argument("-o", "--output", required=True, type=str, help="Output filename (JSON)")
    args = parser.parse_args()

    with open(args.output, "w") as artificial_dataset:
        with open("datasets/twitter.json", "r") as twitter_dataset:
            statuses = json.load(twitter_dataset)

            id_counter = 1
            total_size = 2
            size_reached = False
            artificial_dataset.write("[\n")
            while not size_reached:
                status_idx = random.randrange(len(statuses))
                status = statuses[status_idx]
                if args.id:
                    status["id"] = id_counter
                line = json.dumps(status) + "\n,\n"
                line_size = len(line.encode("utf-8"))
                if (total_size + line_size) > args.size * 1024:
                    line = json.dumps(status) + "\n"
                    line_size = len(line.encode("utf-8"))
                    size_reached = True
                artificial_dataset.write(line)
                total_size += line_size
                id_counter += 1
            artificial_dataset.write("]\n")


if __name__ == "__main__":
    main()
