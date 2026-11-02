# Experimental protocols

How every reported number is produced. If a run deviates, record it in the deviation log at the end.

## 1. Graph and algorithm contracts
Graphs are stored as CSR with 32-bit vertex ids and 64-bit offsets. Vertex ids are compacted to
0..n-1; self-loops and duplicate edges are removed; neighbour lists are sorted. Graphs are directed
as loaded unless run with `--undirected`. SSSP weights are integers 1..100 derived from a hash of
the seed and the unordered endpoint pair (seed 42), so they are independent of vertex id, degree
and edge order.

| Algorithm | Graph used | Result convention |
|---|---|---|
| BFS | as loaded | level per vertex, -1 unreachable; source = highest out-degree vertex |
| PageRank | as loaded, pull over transposed CSR | d = 0.85, initial 1/n, dangling mass spread uniformly, stop at L1 residual < 1e-8 or 100 iterations |
| CC | symmetrized (weakly connected) | component membership; labels may differ across implementations |
| SSSP | as loaded, weighted | uint64 distance, max value = unreachable, non-negative weights |
| TC | symmetrized simple graph | each triangle counted once |

Algorithms are identical across sequential, OpenMP and CUDA (frontier BFS, pull PageRank, union-find CC,
frontier label-correcting SSSP, merge-intersection TC). MPI uses distributed formulations; its CC differs
(label propagation) and TC replicates the graph. Results mixing algorithm and execution model must be
labelled as such.

## 2. Correctness gates
- Unit tests (`scripts/common/run-tests.sh`) must pass before any benchmark.
- Each benchmark run validates trial 0: BFS levels and SSSP distances via independent optimality
  checks, CC by partition equivalence with a flood fill, TC by an independent 3x edge-wise count, PageRank
  by sum = 1, non-negativity, convergence, and (for parallel implementations) L1 distance to the sequential
  result <= 10 x tolerance (1e-7).
- A run that fails validation is recorded with `correct=0` and excluded from summaries.
- Validation covers trial 0 only; deterministic algorithms are assumed to repeat.

## 3. Timing boundaries
| Field | Meaning |
|---|---|
| load_s | file parse and CSR build |
| preprocess_s | per algorithm: PageRank transpose, CC/TC symmetrization (each includes the symmetry check), SSSP weight generation; BFS none |
| setup_s | MPI: graph distribution; |
| alloc_s, h2d_s | CUDA: device allocation and graph upload |
| compute_s | algorithm only: wall clock (sequential, OpenMP); max over ranks incl. communication (MPI); CUDA events from first to last kernel incl. per-round control copies (CUDA) |
| comm_s | MPI: max over ranks of time inside MPI calls |
| d2h_s | CUDA: result download |
| e2e_s | load + preprocess + setup (MPI) or graph alloc/upload + trial wall time (CUDA) + compute |

Compute-only and end-to-end figures are never compared with each other. Speedup is reported on compute
time by default, with end-to-end alongside; for CUDA, kernel-only and kernel + transfers are both shown.

## 4. Measurement procedure
- One untimed warm-up run, then the timed trials. Development runs use 3 to 5 trials; reported runs
  use at least 10. Median and IQR are computed per configuration.
- OpenMP threads and MPI ranks sweep powers of two up to the physical core count. On Linux, threads are
  pinned (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`).
- The sequential reference is measured on the same machine, under the same `RUN_TAG`, as the parallel
  runs it is compared with.
- Machines are otherwise idle during runs; no results from different machines are pooled.

## 5. Metrics
- Speedup = `T_sequential / T_parallel` (median compute time).
- Parallel efficiency = speedup / workers, for OpenMP and MPI only (not meaningful for a GPU).
- MPI communication share = `comm_s / compute_s`.
- Regressions (speedup < 1) and failed runs are reported, not removed.

## 6. Running an experiment
    export RUN_TAG=<name>
    scripts/build.sh && scripts/common/run-tests.sh
    scripts/fetch-datasets.sh
    scripts/run-all.sh                 # or scripts/<impl>/run-<algo>.sh
    scripts/common/analyse-results.sh
    scripts/common/plot-results.sh
Output: `results/raw/<RUN_TAG>_<graph>.csv`, `<RUN_TAG>_environment.txt` (commit, CPU, versions),
`results/processed/`, `results/figures/<RUN_TAG>/`.

## 7. Data handling
Raw CSVs and environment files are append-only and never edited. Processed tables and figures are regenerated
from raw data. Every reported figure names the `RUN_TAG` and the git commit it came from.

## 8. Profiling protocol (CUDA BFS and PageRank)
Vanilla implementations are profiled on the reference GPU with Nsight Systems (timeline, transfers) and
Nsight Compute (kernel metrics). Bottlenecks are recorded as measurements and handed to Phase 2 with the
profile files. No optimization is chosen before the profile exists. `<fill: tool versions and
counter availability once profiled>`

## 9. Phase 2 protocol (outline)
Each optimization is a separate variant in `src/policies/`, validated with the same gates, and
compared with the unchanged vanilla baseline (`T_vanilla / T_optimized`). Combinations come after
individual evaluation. The lookahead classifier is trained and tuned on a set of graphs, evaluated on
held-out graphs with policy and switching overhead included, and compared with fixed strategies, a
rule-based policy and an offline oracle. Details go in Brief 02.

## 10. Threats to validity
- Few graph families so far: no structural claims beyond what was tested.
- Single-node MPI hides network cost.
- Development-machine variability (hybrid cores, thermal throttling, no pinning on macOS).
- Few trials make IQR coarse; use more for reported results.
- CUDA kernel-only timing excludes transfers; both views are reported.
- Dynamic scheduling and atomics make parallel PageRank results differ slightly in floating point.