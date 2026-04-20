#!/bin/sh

set -e

expected_log="correct-answer.log"
actual_log="$(mktemp)"

cleanup() {
  rm -f "$actual_log"
}

trap cleanup EXIT

run_query() {
  json_path="$1"
  query_raw="$2"
  query_print="$3"

  if [ "$first_query" = "0" ]; then
    printf '\n' >> "$actual_log"
  fi
  first_query=0

  printf 'build/nj %s "%s"\n' "$json_path" "$query_print" >> "$actual_log"
  build/nj "$json_path" "$query_raw" >> "$actual_log"
}

first_query=1

run_query datasets/twitter.json '$[*].user.lang' '\$[*].user.lang'
run_query datasets/twitter.json '$[*].entities.urls[*].url' '\$[*].entities.urls[*].url'
run_query datasets/bestbuy.json '$.products[*].categoryPath[1:3].id' '\$.products[*].categoryPath[1:3].id'
run_query datasets/bestbuy.json '$.products[*].videoChapters[*].chapter' '\$.products[*].videoChapters[*].chapter'
run_query datasets/googlemaps.json '$[*].routes[*].legs[*].steps[*].distance.text' '\$[*].routes[*].legs[*].steps[*].distance.text'
run_query datasets/googlemaps.json '$[*].available_travel_modes' '\$[*].available_travel_modes'
run_query datasets/nspl.json '$.meta.view.columns[*].name' '\$.meta.view.columns[*].name'
run_query datasets/nspl.json '$.data[*][*][*]' '\$.data[*][*][*]'
run_query datasets/walmart.json '$.items[*].bestMarketplacePrice.price' '\$.items[*].bestMarketplacePrice.price'
run_query datasets/wikipedia.json '$[*].claims.P150[*].mainsnak.property' '\$[*].claims.P150[*].mainsnak.property'
printf '\n' >> "$actual_log"

if ! diff -u "$expected_log" "$actual_log"; then
  printf 'big e2e results differ from %s\n' "$expected_log" >&2
  exit 1
fi

printf 'big e2e results match %s\n' "$expected_log"
