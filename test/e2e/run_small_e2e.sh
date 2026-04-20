#!/bin/sh

set -e

expected_log="test/e2e/correct-small.log"
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

run_query test/e2e/fixtures/people.json '$.people[*].name' '\$.people[*].name'
run_query test/e2e/fixtures/people.json '$.people[*].age' '\$.people[*].age'
run_query test/e2e/fixtures/people.json '$.people[*].tags' '\$.people[*].tags'
run_query test/e2e/fixtures/people.json '$.active' '\$.active'

run_query test/e2e/fixtures/store.json '$.store.book[*].title' '\$.store.book[*].title'
run_query test/e2e/fixtures/store.json '$.store.book[1:2].price' '\$.store.book[1:2].price'
run_query test/e2e/fixtures/store.json '$.store.bicycle.color' '\$.store.bicycle.color'
run_query test/e2e/fixtures/store.json '$.ids[1:3]' '\$.ids[1:3]'

run_query test/e2e/fixtures/events.json '$[*].type' '\$[*].type'
run_query test/e2e/fixtures/events.json '$[*].meta.x' '\$[*].meta.x'
run_query test/e2e/fixtures/events.json '$[*].ok' '\$[*].ok'

if ! diff -u "$expected_log" "$actual_log"; then
  printf 'small e2e results differ from %s\n' "$expected_log" >&2
  exit 1
fi

printf 'small e2e results match %s\n' "$expected_log"
