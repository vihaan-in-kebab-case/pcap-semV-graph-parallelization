#!/usr/bin/env bash
# Full figure set for three scenarios: Mac, 5090 with GPU, 5090 with the GPU removed.
#   scripts/make-all-figures.sh            representative per-graph plots
#   ALL=1 scripts/make-all-figures.sh      per-graph plots for every graph (slower)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ -d .venv ]; then set +u; . .venv/bin/activate; set -u; fi
REP="facebook web-google uniform_n20_d8 rmat_n20_d8 grid_n20 smallworld_n20_d8"

run_scenario() {   # label tag selector_dir [plot flags...]
  local label="$1" tag="$2" seldir="$3"
  shift 3
  local base="results/figures/$label"
  echo; echo "=== $label ($tag) ==="
  mkdir -p "$base"
  python scripts/plot_suite.py --tag "$tag" --out "$base/suite" --title "$label" "$@"
  local graphs="$REP" files=() g f
  if [ "${ALL:-0}" = "1" ]; then graphs="$(awk -F, 'NR>1 {print $1}' datasets/manifest.csv)"; fi
  for g in $graphs; do
    f="results/raw/${tag}_${g}.csv"
    if [ -f "$f" ]; then files+=("$f"); fi
  done
  if [ "${#files[@]}" -gt 0 ]; then
    python scripts/plot_results.py "${files[@]}" --out "$base/per_graph" --processed "$base" "$@"
  fi
  if [ -d "results/figures/$seldir" ]; then
    rm -rf "$base/selector"
    cp -r "results/figures/$seldir" "$base/selector"
  else
    echo "note: results/figures/$seldir not found, selector figures not copied"
  fi
}

run_scenario mac sweep_mac selector_sweep_mac
run_scenario 5090_gpu sweep_5090 selector_sweep_5090 --drop-impl mpi
run_scenario 5090_nogpu sweep_5090 selector_sweep_5090_nogpu --drop-impl mpi --drop-impl cuda
echo; echo "done: results/figures/{mac,5090_gpu,5090_nogpu}/"