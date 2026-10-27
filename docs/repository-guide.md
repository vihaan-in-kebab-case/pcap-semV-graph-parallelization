# Repository guide

What each main folder is for, why it exists, and the rules that keep the experiment trustworthy.

> There are no top-level `benchmarks/` or `analysis/` directories. Those roles are served today by
> `scripts/` + `config/` (benchmarks) and `scripts/analyze_results.py` + `scripts/plot_results.py`
> + `results/processed|figures` (analysis). They are described separately below because the
> responsibilities differ; they can become real directories later without changing the design.

## How the pieces depend on each other

    include/  ->  src/  ->  drivers and tests  ->  results/raw  ->  analysis  ->  results/processed, figures
    scripts/ (+ config/) launch everything and contain no algorithm logic.

## include/ — the contracts
**Role.** Defines what every implementation must agree on: CSR graph layout, result types per
algorithm, the correctness checkers, and the benchmark record (timer plus CSV schema).
Also holds the MPI partition helpers (`mpi_dist.hpp`) and CUDA helpers (`cuda_common.hpp`).
**Why it matters.** Because sequential, OpenMP, MPI and CUDA all compile against the same headers,
"BFS level" or "compute_s" means the same thing in every comparison. This is what makes the
cross-model results comparable at all.
**Rules.** A header change can alter every baseline, so treat it as a contract change: re-run the
tests and note it in the deviation log (`docs/experimental-protocols.md`).

## src/ — the implementations
**Role.** `common/` (graph loader, CSR build, transpose, symmetrize, synthetic weights),
`sequential/`, `openmp/`, `mpi/`, `cuda/` (the vanilla baselines), driver mains
(`bench_main.cpp`, `mpi_bench_main.cpp`, `cuda_bench_main.cpp`), and `policies/`
(reserved for Phase 2: optimized CUDA BFS and PageRank, and the lookahead classifier).
**Why it matters.** Keeping vanilla baselines and optimized code in different directories means an
optimization can never silently overwrite the thing it is compared against.
**Rules.** Once a baseline is validated it is frozen. Optimizations go in `policies/` as separate
variants.

## tests/ — the correctness gate
**Role.** `tiny_graphs/` (small, hand-checkable, committed), `correctness/` (`test_sequential`,
`test_openmp`, `test_mpi`, `test_cuda`), `regression/` (reserved).
Tests compare against independent oracles (Bellman-Ford and Dijkstra for SSSP, brute-force triangles,
flood-fill components, a separate pull PageRank) and run with hub-skewed random graphs, edge cases
(single vertex, empty MPI blocks) and several thread or rank counts.
**Why it matters.** The project rule is "validate correctness before comparing performance". A
speedup from an incorrect implementation is meaningless, and parallel bugs are often
intermittent and size-dependent.
**Rules.** `scripts/common/run-tests.sh` runs before any benchmark in the demo flow. A failing test
means no results are produced.

## config/ — what to run
**Role.** `datasets.yaml` (graph sources, checksums, family, directedness) and `benchmark.yaml`
(trials, warm-up, sweeps, algorithm parameters, validation tolerances).
**Why it matters.** A run should be reproducible from a file plus a git commit, and a reviewer should
be able to read the experiment definition without reading shell scripts.
**Status.** Not yet read by the scripts; the scripts are the source of truth until that is
wired up. Each file says so at the top.

## Benchmarks (scripts/ and config/) — how to run
**Role.** `build.sh`, `fetch-datasets.sh`, `run-all.sh`, `demo-phase1.sh`, `common/` helpers, and one
`run-<algo>.sh` per baseline and algorithm. They build, test, fetch, sweep thread/rank counts, record
the environment, and write CSVs tagged with `RUN_TAG`.
**Why it matters.** The measurement procedure lives in code, not in someone's memory, so the same
command repeats the same experiment on another machine.
**Rules.** Sequential reference runs are recorded on the same machine and tag as the parallel runs
they are compared with.

## Analysis (scripts/analyze_results.py, plot_results.py) — from rows to claims
**Role.** Median and IQR per configuration, speedup and efficiency versus the sequential reference,
a processed summary CSV, and five figures per graph (speedup, compute comparison, end-to-end
breakdown, efficiency, MPI communication share).
**Why it matters.** Raw rows are not evidence until summarized consistently. The scripts flag
incorrect runs and show the end-to-end breakdown, which stops compute-only speedups from being
over-read.
**Rules.** Never hand-edit outputs; regenerate them. Do not mix CSVs from different machines in one plot.

## results/ — the evidence trail
**Role.** `raw/` (append-only CSV per tag and graph, plus `<tag>_environment.txt`), `processed/`
(summary tables), `figures/` (plots).
**Why it matters.** Every claim in the report must trace back to raw rows. Keeping negative and
inconclusive runs is a stated project principle.
**Rules.** Raw files are immutable. Processed files and figures are reproducible outputs.

## Supporting folders
`datasets/` (README with sources and checksums; graphs in gitignored `graphs/`), `docs/` (scope,
briefs, protocols, this guide), `environment/` (machine and software record, Dockerfile).

## Where Phase 2 plugs in
Optimized CUDA variants go in `src/policies/` with `include/execution_policy.hpp`; their tests in
`tests/regression/`; their runs under new `RUN_TAG`s; and the vanilla baselines stay untouched.