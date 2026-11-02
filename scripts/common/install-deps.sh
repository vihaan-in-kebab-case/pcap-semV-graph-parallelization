#!/usr/bin/env bash
# Fresh machine setup (macOS: Homebrew; Debian/Ubuntu: apt). The CUDA toolkit is NOT installed here.
set -euo pipefail
if [ "$(uname)" = "Darwin" ]; then
  brew install cmake libomp open-mpi
elif command -v apt-get >/dev/null 2>&1; then
  SUDO=""
  if [ "$(id -u)" != "0" ]; then SUDO="sudo"; fi
  $SUDO apt-get update
  $SUDO apt-get install -y build-essential cmake git curl wget python3 python3-venv libopenmpi-dev openmpi-bin
else
  echo "unsupported system: install a C++17 compiler, cmake, OpenMP, MPI and python3 manually" >&2; exit 1
fi
if command -v nvcc >/dev/null 2>&1; then echo "CUDA: $(nvcc --version | tail -1)"; else echo "note: nvcc not found, CUDA targets will be skipped"; fi