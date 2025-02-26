#!/usr/bin/env bash

set -euo pipefail

while read -r file url
do
  if [ ! -f "$file" ]; then
    curl -L --compressed "$url" -o - | gunzip > "$file"
  fi
done < datasets.tsv

sha256sum -c checksums.sha256
