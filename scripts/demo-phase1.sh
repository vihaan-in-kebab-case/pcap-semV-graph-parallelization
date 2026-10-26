#!/usr/bin/env bash
# One-shot Phase 1 demo: build -> tests -> datasets -> baselines -> tables -> figures.
#   QUICK=1 for fewer trials; SKIP_BUILD=1 / SKIP_TESTS=1 to skip steps; pass targets to limit, e.g. demo-phase1.sh openmp
set -euo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_TAG="${RUN_TAG:-demo_$(date +%Y%m%d_%H%M%S)}"
echo "RUN_TAG=$RUN_TAG"
if [ "${SKIP_BUILD:-0}" != "1" ]; then "$S/build.sh"; fi
if [ "${SKIP_TESTS:-0}" != "1" ]; then "$S/common/run-tests.sh"; fi
"$S/fetch-datasets.sh"
"$S/run-all.sh" "$@"
"$S/common/analyse-results.sh"
"$S/common/plot-results.sh"
echo; echo "figures: results/figures/$RUN_TAG/"