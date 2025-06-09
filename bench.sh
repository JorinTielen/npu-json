#!/bin/sh

set -e

echo "Running benchmark: twitter (T1)"
build/nj datasets/twitter.json "\$[*].user.lang" --bench
echo "Running benchmark: twitter (T2)"
build/nj datasets/twitter.json "\$[*].entities.urls[*].url" --bench
echo "Running benchmark: bestbuy (B2)"
build/nj datasets/bestbuy.json "\$.products[*].videoChapters[*].chapter" --bench
echo "Running benchmark: googlemaps (G1)"
build/nj datasets/googlemaps.json "\$[*].routes[*].legs[*].steps[*].distance.text" --bench
echo "Running benchmark: googlemaps (G2)"
build/nj datasets/googlemaps.json "\$[*].available_travel_nodes" --bench
echo "Running benchmark: nspl (N1)"
build/nj datasets/nspl.json "\$.meta.view.columns[*].name" --bench
echo "Running benchmark: nspl (N2)"
build/nj datasets/nspl.json "\$.data[*][*][*]" --bench
echo "Running benchmark: walmart (W1)"
build/nj datasets/walmart.json "\$.items[*].bestMarketplacePrice.price" --bench
echo "Running benchmark: wikipedia (Wi)"
build/nj datasets/wikipedia.json "\$[*].claims.P150[*].mainsnak.property" --bench
