#!/usr/bin/env bash
# Build full-size 23andMe-style test files from per-chromosome 1kg data (~1.3M SNPs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KG="$ROOT/1kg"
OUT="$ROOT/test_data/large"
PLINK="$ROOT/tools/plink"
mkdir -p "$OUT"

build_large() {
  local id="$1"
  local pop_label="$2"
  local outfile="$OUT/${id}_full_23andme.txt"
  echo "Building $outfile ($pop_label)..."
  echo "$id $id" > "$OUT/${id}.keep"

  {
    echo "# This data file generated for Ancestry-Finder testing"
    echo "# Simulates a large consumer raw export (~1.3M SNPs, hg19)"
    echo "# Sample: $id ($pop_label)"
    echo "#"
    echo "# rsid	chromosome	position	genotype"
  } > "$outfile"

  ls "$KG"/1KGPhase3.w_hm3.chr*.bed | sed 's/.bed$//' | sort -V | while read -r prefix; do
    "$PLINK" --bfile "$prefix" --keep "$OUT/${id}.keep" --recode 23 \
      --out "$OUT/.tmp_${id}" >/dev/null 2>&1
    recode_file="$OUT/.tmp_${id}.txt"
    test -f "$recode_file" || recode_file="$OUT/.tmp_${id}.23"
    grep -v '^#' "$recode_file" >> "$outfile"
    rm -f "$OUT/.tmp_${id}".*
  done

  rm -f "$OUT/${id}.keep"
  wc -l "$outfile"
  ls -lh "$outfile"
}

build_large HG00096 "British / EUR"
build_large NA19238 "Yoruba / AFR"
build_large HG00419 "Han Chinese / EAS"

echo "Done. Files in test_data/large/"
