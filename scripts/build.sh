#!/usr/bin/env bash
# Configure + build everything available on this machine (seq, OpenMP; MPI and CUDA if found).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)}"

args=(-S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release)
if [ "$(uname)" = "Darwin" ]; then
  LIBOMP="$(brew --prefix libomp)"
  args+=(-DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$LIBOMP/include"
         -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_omp_LIBRARY="$LIBOMP/lib/libomp.dylib")
fi
# CUDA architecture from the installed GPU (e.g. 7.5 -> 75); override with CUDA_ARCH=89
if [ -z "${CUDA_ARCH:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ' || true)"
fi
if [ -n "${CUDA_ARCH:-}" ]; then args+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"); fi

cmake "${args[@]}"
cmake --build "$BUILD_DIR" -j"$JOBS"

echo; echo "built targets:"
for b in gps_bench gps_bench_mpi gps_bench_cuda test_sequential test_openmp test_mpi test_cuda; do
  if [ -x "$BUILD_DIR/$b" ]; then echo "  [x] $b"; else echo "  [ ] $b (not available here)"; fi
done