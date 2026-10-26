#!/usr/bin/env bash
# sequential reference: bfs on every graph in $GRAPHS (options: see scripts/common/env.sh)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"
run_sequential bfs
