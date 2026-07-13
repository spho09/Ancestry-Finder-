#!/usr/bin/env bash
# Remove reproducible PLINK intermediates. Safe after reference_pruned.* exists.
set -euo pipefail
cd "$(dirname "$0")"

rm -f merge_list.txt
rm -f reference.bed reference.bim reference.fam reference.log reference.nosex
rm -f prune.log prune.nosex prune.prune.in prune.prune.out
rm -f reference_pruned.log reference_pruned.nosex

echo "Cleaned offline build artifacts."
echo "Kept: reference_pruned.*, model/, integrated_call_samples.panel"
