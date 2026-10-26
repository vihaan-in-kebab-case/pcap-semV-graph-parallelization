#!/usr/bin/env bash
# Shared settings and helpers. SOURCED (not executed) by every script in scripts/.
# Override from the environment, e.g.:  TRIALS=3 GRAPHS=facebook scripts/openmp/run-bfs.sh
#   RUN_TAG    results file prefix (default phase1)      GRAPHS   facebook and/or web-google
#   TRIALS     timed trials (default 5, QUICK=1 -> 3)    WARMUP   untimed warm-ups (default 1)
#   THREADS    OpenMP thread counts (default 1,2,4.. up to cores)   RANKS  MPI ranks (default = THREADS)
#   CUDA_DEVICE GPU index (default 0)                    BUILD_DIR, GRAPHS_DIR, RESULTS_DIR
set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$COMMON_DIR")"
ROOT="$(dirname "$SCRIPTS_DIR")"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
GRAPHS_DIR="${GRAPHS_DIR:-$ROOT/datasets/graphs}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/raw}"
RUN_TAG="${RUN_TAG:-phase1}"
GRAPHS="${GRAPHS:-facebook web-google}"
WARMUP="${WARMUP:-1}"
TRIALS="${TRIALS:-5}"
if [ "${QUICK:-0}" = "1" ]; then TRIALS=3; fi
CUDA_DEVICE="${CUDA_DEVICE:-0}"
mkdir -p "$RESULTS_DIR"
if [ "$(uname)" = "Darwin" ]; then export TMPDIR=/tmp; fi   # Open MPI session-dir path length on macOS

ncores() { nproc 2>/dev/null || sysctl -n hw.physicalcpu; }
worker_counts() {
  local max c=1 out=""
  max="$(ncores)"
  while [ "$c" -le "$max" ]; do out="$out $c"; c=$((c * 2)); done
  echo "${out# }"
}
THREADS="${THREADS:-$(worker_counts)}"
RANKS="${RANKS:-$THREADS}"

graph_file() {
  case "$1" in
    facebook)   echo "$GRAPHS_DIR/facebook_combined.txt" ;;
    web-google) echo "$GRAPHS_DIR/web-Google.txt" ;;
    *) echo "unknown graph '$1' (known: facebook web-google)" >&2; return 1 ;;
  esac
}
graph_flags() { case "$1" in facebook) echo "--undirected" ;; *) echo "" ;; esac; }
csv_for() { echo "$RESULTS_DIR/${RUN_TAG}_$1.csv"; }

# true if the CSV already has rows for (algo, impl, workers)
have_rows() {
  local csv="$1" impl="$2" algo="$3" workers="$4"
  [ -f "$csv" ] || return 1
  awk -F, -v a="$algo" -v i="$impl" -v w="$workers" \
    'NR>1 && $1==a && $2==i && $7==w {f=1} END{exit !f}' "$csv"
}

mpi_flags() {   # $1 = ranks; Open MPI only (needed when root in containers / oversubscribed)
  local flags=""
  if mpirun --version 2>&1 | grep -qi "open.mpi"; then
    if [ "$(id -u)" = "0" ]; then flags="$flags --allow-run-as-root"; fi
    if [ "$1" -gt "$(ncores)" ]; then flags="$flags --oversubscribe"; fi
  fi
  echo "$flags"
}

collect_environment() { "$COMMON_DIR/collect-environment.sh" "$RESULTS_DIR/${RUN_TAG}_environment.txt"; }

require_bin() {
  local impl="$1" bin
  if [ ! -x "$BUILD_DIR/gps_bench" ]; then
    echo "error: $BUILD_DIR/gps_bench not found. Run scripts/build.sh first." >&2; exit 1
  fi
  case "$impl" in
    openmp) return 0 ;;
    mpi)  bin=gps_bench_mpi ;;
    cuda) bin=gps_bench_cuda ;;
    *) echo "unknown impl '$impl'" >&2; exit 2 ;;
  esac
  if [ ! -x "$BUILD_DIR/$bin" ]; then
    echo "error: $BUILD_DIR/$bin not found (MPI/CUDA not available at build time?). Run scripts/build.sh." >&2; exit 1
  fi
  if [ "$impl" = "mpi" ] && ! command -v mpirun >/dev/null 2>&1; then echo "error: mpirun not found" >&2; exit 1; fi
  if [ "$impl" = "cuda" ] && ! command -v nvidia-smi >/dev/null 2>&1; then echo "error: no NVIDIA GPU/driver found" >&2; exit 1; fi
}

run_seq_ref() {   # run_seq_ref <algo> <graph>
  local algo="$1" g="$2" csv
  csv="$(csv_for "$g")"
  if have_rows "$csv" seq "$algo" 1; then echo "[seq ] reference for $algo/$g already recorded"; return 0; fi
  echo "--- seq reference ($algo, $g)"
  "$BUILD_DIR/gps_bench" --graph "$(graph_file "$g")" $(graph_flags "$g") --algo "$algo" \
    --impl seq --trials "$TRIALS" --warmup "$WARMUP" --out "$csv"
}

run_openmp() {    # run_openmp <algo> <graph>
  local algo="$1" g="$2" csv t
  csv="$(csv_for "$g")"
  if [ "$(uname)" = "Linux" ]; then export OMP_PROC_BIND=close OMP_PLACES=cores; fi
  for t in $THREADS; do
    echo "--- openmp threads=$t ($algo, $g)"
    "$BUILD_DIR/gps_bench" --graph "$(graph_file "$g")" $(graph_flags "$g") --algo "$algo" \
      --impl omp --threads "$t" --trials "$TRIALS" --warmup "$WARMUP" --out "$csv"
  done
}

run_mpi() {       # run_mpi <algo> <graph>
  local algo="$1" g="$2" csv r
  csv="$(csv_for "$g")"
  for r in $RANKS; do
    echo "--- mpi ranks=$r ($algo, $g)"
    mpirun -np "$r" $(mpi_flags "$r") "$BUILD_DIR/gps_bench_mpi" \
      --graph "$(graph_file "$g")" $(graph_flags "$g") --algo "$algo" \
      --trials "$TRIALS" --warmup "$WARMUP" --out "$csv"
  done
}

run_cuda() {      # run_cuda <algo> <graph>
  local algo="$1" g="$2" csv
  csv="$(csv_for "$g")"
  echo "--- cuda device=$CUDA_DEVICE ($algo, $g)"
  "$BUILD_DIR/gps_bench_cuda" --graph "$(graph_file "$g")" $(graph_flags "$g") --algo "$algo" \
    --device "$CUDA_DEVICE" --trials "$TRIALS" --warmup "$WARMUP" --out "$csv"
}

run_sequential() {   # run_sequential <algo>
  local algo="$1" g file
  require_bin openmp   # only checks that gps_bench exists
  collect_environment
  for g in $GRAPHS; do
    file="$(graph_file "$g")"
    if [ ! -f "$file" ]; then echo "error: $file missing. Run scripts/fetch-datasets.sh" >&2; exit 1; fi
    echo "=== sequential / $algo / $g  (RUN_TAG=$RUN_TAG) ==="
    run_seq_ref "$algo" "$g"
  done
}

run_baseline() {  # run_baseline <openmp|mpi|cuda> <bfs|pagerank|cc|sssp|tc>
  local impl="$1" algo="$2" g file
  require_bin "$impl"
  collect_environment
  for g in $GRAPHS; do
    file="$(graph_file "$g")"
    if [ ! -f "$file" ]; then echo "error: $file missing. Run scripts/fetch-datasets.sh" >&2; exit 1; fi
    echo "=== $impl / $algo / $g  (RUN_TAG=$RUN_TAG) ==="
    run_seq_ref "$algo" "$g"
    case "$impl" in
      openmp) run_openmp "$algo" "$g" ;;
      mpi)    run_mpi "$algo" "$g" ;;
      cuda)   run_cuda "$algo" "$g" ;;
    esac
  done
}

ensure_venv() {   # python env for analysis/plots
  if [ ! -d "$ROOT/.venv" ]; then python3 -m venv "$ROOT/.venv"; fi
  set +u; . "$ROOT/.venv/bin/activate"; set -u
  if ! python -c "import pandas, matplotlib, numpy" 2>/dev/null; then
    pip install --quiet pandas matplotlib numpy
  fi
}