#!/usr/bin/env bash
# One-shot session for the GPU machine (Linux or WSL2).
#   scripts/laptop-session.sh                 everything
#   STAGE=sweep scripts/laptop-session.sh     resume at: check | build | test | sweep | package
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export RUN_TAG="${RUN_TAG:-sweep_5090}" GRAPHS="${GRAPHS:-all}" TRIALS="${TRIALS:-3}"
STAGE="${STAGE:-check}"
order="check build test sweep package"
idx() { local i=0 s; for s in $order; do if [ "$s" = "$1" ]; then echo "$i"; return; fi; i=$((i + 1)); done; echo 99; }
want() { [ "$(idx "$1")" -ge "$(idx "$STAGE")" ]; }
if [ -d "$ROOT/.venv" ]; then set +u; . "$ROOT/.venv/bin/activate"; set -u; fi

if want check; then
  echo "== check: GPU, driver, toolkit =="
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: the GPU is not visible here" >&2; exit 1; }
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
  command -v nvcc >/dev/null || { echo "nvcc not found: install the CUDA toolkit 12.8+ (toolkit only)" >&2; exit 1; }
  nvcc --version | tail -1
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
  read -r maj min <<<"$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p' | head -1)"
  if [ "${cap%%.*}" -ge 12 ] && { [ "$maj" -lt 12 ] || { [ "$maj" -eq 12 ] && [ "$min" -lt 8 ]; }; }; then
    echo "GPU compute capability $cap needs CUDA toolkit 12.8 or newer (found $maj.$min)" >&2; exit 1
  fi
  python3 -c "import numpy, pandas, matplotlib, sklearn" 2>/dev/null || echo "warning: python packages missing (activate .venv and pip install -r selector/requirements.txt)"
fi

if want build; then
  echo "== build =="
  scripts/fetch-datasets.sh
  if [ ! -f datasets/graphs/uniform_n10_d8.txt ]; then python datasets/generators/generate_graphs.py --preset demo; fi
  sha256sum datasets/graphs/uniform_n10_d8.txt
  echo "(compare this hash with the same file on the Mac; if it differs, copy datasets/graphs/ from the Mac)"
  scripts/build.sh
  [ -x build/gps_bench_cuda ] || { echo "gps_bench_cuda was not built: read the CUDA errors above" >&2; exit 1; }
fi

if want test; then
  echo "== tests =="
  scripts/common/run-tests.sh
fi

if want sweep; then
  echo "== sweep (CUDA first; resumable; RUN_TAG=$RUN_TAG) =="
  echo "reminder: plugged in, maximum-performance power mode, other apps closed"
  scripts/extract-features.sh
  scripts/run-all.sh cuda openmp mpi
  python selector/build_dataset.py --tag "$RUN_TAG"
fi

if want package; then
  echo "== package =="
  out="$ROOT/results_${RUN_TAG}.tgz"
  files="$(ls results/raw/${RUN_TAG}_* results/features/${RUN_TAG}_features.jsonl results/processed/${RUN_TAG}_dataset.csv 2>/dev/null || true)"
  if [ -d results/traces ]; then files="$files results/traces"; fi
  tar czf "$out" $files
  echo "created $out"
  if command -v wslpath >/dev/null 2>&1; then echo "Windows path: $(wslpath -w "$out")"; fi
  echo "before returning the laptop: run demo queries with --verify and keep results/traces"
fi