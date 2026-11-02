#!/usr/bin/env bash
# Figures + processed summary. Usage: plot-results.sh [CSV ...]   (default: results/raw/$RUN_TAG_*.csv)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
ensure_venv
if [ "$#" -gt 0 ]; then files=("$@"); else files=("$RESULTS_DIR"/${RUN_TAG}_*.csv); fi
if [ ! -e "${files[0]}" ]; then echo "no result CSVs found for RUN_TAG=$RUN_TAG" >&2; exit 1; fi
python "$SCRIPTS_DIR/plot_results.py" "${files[@]}" \
  --out "$ROOT/results/figures/$RUN_TAG" --processed "$ROOT/results/processed"