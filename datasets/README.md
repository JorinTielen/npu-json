# Datasets

The datasets used for benchmarking are automatically downloaded using
using a script before running the benchmarks. The specific benchmarks and
their original sources are listed here:

| Dataset name | Size | Description | Source |
|--------------|------|-------------|--------|
| `ast`        | 25 MB | JSON representation of the AST of an arbitrary popular C file from Software Heritage. | [`rsonpath`](https://github.com/rsonquery/rsonpath) benchmarks. |
| `twitter`    | 750 KB | Tweets containing mostly Japanese characters, taken from the Twitter API. | [`simdjson`](https://github.com/simdjson/simdjson) benchmarks. |
| `bestbuy_large_record` | ? | BestBuy product dataset. | [JSONSki](https://github.com/automatalab/jsonski) benchmarks. |
| `openfood`   | ? | Data extracted from Open Food Facts API. | [`rsonpath`](https://github.com/rsonquery/rsonpath) benchmarks. |

The datasets are downloaded from the rsonpath benchmarking dataset open archive on Zenodo:

> Charles Paperman, Mateusz Gienieczko, & Filip Murlak. (2023). Dataset for benchmarking rsonpath [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8395641

## Prerequisites

The script to download the datasets depends on the following utilities:

- `curl`
- `sha256sum`
