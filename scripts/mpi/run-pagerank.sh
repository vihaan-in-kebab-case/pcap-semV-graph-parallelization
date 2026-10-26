#!/usr/bin/env bash
# mpi baseline: pagerank on every graph in $GRAPHS (options: see scripts/common/env.sh)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"
run_baseline mpi pagerank
