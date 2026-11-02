#!/usr/bin/env bash
# Writes machine/software details next to the results (Brief 01 environment record).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/results/raw/environment.txt}"
if [ -f "$OUT" ] && [ "${FORCE:-0}" != "1" ]; then exit 0; fi
mkdir -p "$(dirname "$OUT")"
{
  echo "date_utc: $(date -u +%FT%TZ)"
  echo "git_commit: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_uncommitted_files: $(git -C "$ROOT" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  echo "uname: $(uname -a)"
  if [ "$(uname)" = "Darwin" ]; then
    echo "cpu: $(sysctl -n machdep.cpu.brand_string)"
    echo "cores_physical: $(sysctl -n hw.physicalcpu)"
    echo "cores_logical: $(sysctl -n hw.logicalcpu)"
    echo "memory_bytes: $(sysctl -n hw.memsize)"
  else
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
    echo "cores_logical: $(nproc)"
    echo "memory_kb: $(grep MemTotal /proc/meminfo | awk '{print $2}')"
  fi
  echo "cmake: $(cmake --version | head -1)"
  echo "cxx: $(${CXX:-c++} --version | head -1)"
  if command -v mpirun >/dev/null 2>&1; then echo "mpi: $(mpirun --version 2>&1 | head -1)"; fi
  if command -v nvcc >/dev/null 2>&1; then echo "nvcc: $(nvcc --version | tail -1)"; fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /'
  fi
  echo "omp_env: OMP_PROC_BIND=${OMP_PROC_BIND:-unset} OMP_PLACES=${OMP_PLACES:-unset}"
} > "$OUT"
echo "[env ] wrote $OUT"