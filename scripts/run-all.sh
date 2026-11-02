#!/usr/bin/env bash
# scripts/run-all.sh [openmp|mpi|cuda ...]   default: openmp + mpi (+ cuda if built) x 5 algorithms
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common/env.sh"
targets=("$@")
if [ "${#targets[@]}" -eq 0 ]; then
  targets=(sequential openmp)
  if [ -x "$BUILD_DIR/gps_bench_mpi" ]; then targets+=(mpi); fi
  if [ -x "$BUILD_DIR/gps_bench_cuda" ] && command -v nvidia-smi >/dev/null 2>&1; then targets+=(cuda); fi
fi
echo "running: ${targets[*]}  (RUN_TAG=$RUN_TAG, graphs: $GRAPHS, trials: $TRIALS)"
for t in "${targets[@]}"; do
  for a in bfs pagerank cc sssp tc; do
    "$SCRIPTS_DIR/$t/run-$a.sh"
  done
done
echo "done. results: $RESULTS_DIR/${RUN_TAG}_*.csv"