# Datasets

The datasets used for benchmarking can be automatically downloaded and checked
for validity using their sha256 hash using a script before running the
benchmarks. The specific benchmarks and their original sources (first paper to
use it) are listed here:

| Dataset name | Size | Description | (Original) Source |
|--------------|------|-------------|-------------------|
| `ast`        | 25 MB | JSON representation of the AST of an arbitrary popular C file from Software Heritage. | [`rsonpath`](https://github.com/rsonquery/rsonpath) benchmarks. |
| `twitter`    | 750 KB | Tweets containing mostly Japanese characters, taken from the Twitter API. | [`simdjson`](https://github.com/simdjson/simdjson) benchmarks. |
| `bestbuy` | 997 MB | BestBuy product dataset. | [GPJSON](https://github.com/koesie10/gpsjon) benchmarks. |
| `nspl` | 1.2 GB | National Statistics Postcode Lookup (NSPL) dataset from United Kingdom. | [JSONSki](https://github.com/automatalab/jsonski) benchmarks. |
| `walmart` | 950 MB | Walmart product dataset. | [GPJSON](https://github.com/koesie10/gpjson) benchmarks. |

The datasets are downloaded from the rsonpath benchmarking dataset open archive
on Zenodo:

> Charles Paperman, Mateusz Gienieczko, & Filip Murlak. (2023). Dataset for benchmarking rsonpath [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8395641

## Prerequisites

The script to download the datasets depends on the following utilities:

- `curl`
- `gunzip`
- `sha256sum`
