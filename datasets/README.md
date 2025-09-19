# Datasets

The datasets used for benchmarking can be automatically downloaded and checked
for validity using their sha256 hash using a script before running the
benchmarks. The specific benchmarks and their original sources (first paper to
use it) are listed here:

| Dataset name | Size | Description |
|--------------|------|-------------|
| `twitter`    | 843 MB | Tweets containing mostly Japanese characters, taken from the Twitter API. |
| `bestbuy` | 1045 MB | BestBuy product dataset. |
| `googlemaps` | 1136 MB | Google Maps geographic dataset. |
| `nspl` | 1210 MB | National Statistics Postcode Lookup (NSPL) dataset from United Kingdom. |
| `walmart` | 995 MB | Walmart product dataset. |
| `wikipedia` | 1099 MB | Wikipedia entity dataset. |

The datasets are downloaded from the rsonpath benchmarking dataset open archive
on Zenodo:

> Charles Paperman, Mateusz Gienieczko, & Filip Murlak. (2023). Dataset for benchmarking rsonpath [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8395641

## Prerequisites

The script to download the datasets depends on the following utilities:

- `curl`
- `gunzip`
- `sha256sum`
