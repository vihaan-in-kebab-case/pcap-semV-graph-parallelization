#!/usr/bin/env bash
# Runs every correctness test that was built (sequential, OpenMP, MPI at 1-4 ranks, CUDA if a GPU exists).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
fail=0
run() { echo "--- $*"; if ! "$@"; then echo "FAILED: $*" >&2; fail=1; fi; }
run "$BUILD_DIR/test_sequential"
run "$BUILD_DIR/test_openmp"
if [ -x "$BUILD_DIR/test_mpi" ]; then
  for np in 1 2 3 4; do run mpirun -np "$np" $(mpi_flags "$np") "$BUILD_DIR/test_mpi"; done
else echo "[skip] test_mpi not built"; fi
if [ -x "$BUILD_DIR/test_cuda" ] && command -v nvidia-smi >/dev/null 2>&1; then
  run "$BUILD_DIR/test_cuda"
else echo "[skip] test_cuda (not built or no GPU)"; fi
if [ "$fail" = "0" ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED" >&2; fi
exit "$fail"