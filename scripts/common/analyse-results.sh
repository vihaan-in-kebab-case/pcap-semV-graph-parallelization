#!/usr/bin/env bash
# Median/IQR tables + speedups. Usage: analyse-results.sh [CSV ...]   (default: results/raw/$RUN_TAG_*.csv)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
ensure_venv
if [ "$#" -gt 0 ]; then files=("$@"); else files=("$RESULTS_DIR"/${RUN_TAG}_*.csv); fi
if [ ! -e "${files[0]}" ]; then echo "no result CSVs found for RUN_TAG=$RUN_TAG" >&2; exit 1; fi
for f in "${files[@]}"; do
  echo; echo "################ $f"
  python "$SCRIPTS_DIR/analyze_results.py" "$f"
done