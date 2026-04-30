#!/bin/sh

set -e

MODE="${1:-warm}"

if [ "$MODE" = "cold" ]; then
  BENCH_FLAG="--bench cold"
elif [ "$MODE" = "warm" ]; then
  BENCH_FLAG="--bench"
else
  echo "Usage: $0 [cold|warm]"
  exit 1
fi

echo "=== $MODE benchmarks ==="
echo ""

echo "Running benchmark: twitter (T1)"
build/nj datasets/twitter.json "\$[*].user.lang" $BENCH_FLAG
echo "Running benchmark: twitter (T2)"
build/nj datasets/twitter.json "\$[*].entities.urls[*].url" $BENCH_FLAG
echo "Running benchmark: bestbuy (B1)"
build/nj datasets/bestbuy.json "\$.products[*].categoryPath[1:3].id" $BENCH_FLAG
echo "Running benchmark: bestbuy (B2)"
build/nj datasets/bestbuy.json "\$.products[*].videoChapters[*].chapter" $BENCH_FLAG
echo "Running benchmark: googlemaps (G1)"
build/nj datasets/googlemaps.json "\$[*].routes[*].legs[*].steps[*].distance.text" $BENCH_FLAG
echo "Running benchmark: googlemaps (G2)"
build/nj datasets/googlemaps.json "\$[*].available_travel_modes" $BENCH_FLAG
echo "Running benchmark: nspl (N1)"
build/nj datasets/nspl.json "\$.meta.view.columns[*].name" $BENCH_FLAG
echo "Running benchmark: nspl (N2)"
build/nj datasets/nspl.json "\$.data[*][*][*]" $BENCH_FLAG
echo "Running benchmark: walmart (W1)"
build/nj datasets/walmart.json "\$.items[*].bestMarketplacePrice.price" $BENCH_FLAG
echo "Running benchmark: wikipedia (Wi)"
build/nj datasets/wikipedia.json "\$[*].claims.P150[*].mainsnak.property" $BENCH_FLAG
