# Algorithm and graph contracts (Track 1, Step 1)

**Graph:** CSR, 32-bit vertex ids, 64-bit edge offsets. Vertex ids are compacted to 0..n-1.
Self-loops and duplicate edges are removed; neighbour lists are sorted. Directed as loaded
unless `--undirected` (symmetrized). Weights: integer in [1,100], hash of (seed, unordered pair).

| Algorithm | Graph used | Output / convention | Validation |
|---|---|---|---|
| BFS | as loaded (directed unless --undirected) | level per vertex, -1 = unreachable; source = highest out-degree unless --source | independent level/parent validator |
| PageRank | as loaded; pull over transposed CSR | d=0.85, init 1/n, dangling mass spread uniformly, stop at L1 residual < 1e-8 or 100 iters | sums to 1; parallel vs sequential L1 distance < 10*tol |
| CC | symmetrized, so **weakly** connected components | component membership (labels may differ across implementations) | partition equivalence vs flood fill |
| SSSP | as loaded, weighted | uint64 distance, UINT64_MAX = unreachable, non-negative weights | optimality validator (relaxed + tight predecessor) |
| TC | symmetrized simple graph | each triangle counted once | independent 3x edge-wise count |

## Algorithm choices (identical in every implementation except MPI)
BFS: level-synchronous frontier. PageRank: pull over transposed CSR. CC: union-find,
larger root under smaller. SSSP: frontier-based label-correcting. TC: sorted-list merge
intersection (u < v < w). Sequential = the same algorithm on one thread. Dijkstra and
Bellman-Ford exist only as test oracles.

## (exception) MPI (single node, multi-process)
1D block vertex partition; each rank holds only its slice of the CSR (TC replicates it, see below).
BFS, PageRank, SSSP: same algorithm as seq/OpenMP in distributed form. **CC uses min-label
propagation (differs from union-find elsewhere), so CC speedups mix algorithm and execution
mode.** TC replicates the CSR on every rank (memory grows with ranks). `threads` column = ranks
when `impl=mpi`. `compute_s` = max over ranks incl. communication; `comm_s` = max time in MPI
calls; `setup_s` = graph distribution; `e2e_s = load + preprocess + setup + compute`.

## CUDA
Same algorithms as seq/OpenMP; vanilla thread-per-vertex mapping, no tuning. `compute_s` = CUDA
events (first kernel to last, incl. per-round control copies). `alloc_s`, `d2h_s` = wall time
around cudaMalloc / result downloads; `h2d_s` = graph upload (once per run). `e2e_s` = load +
preprocess + graph alloc/upload + trial wall time (includes cudaFree).

**Timing:** `load_s` = file parse + CSR build. `preprocess_s` = symmetrization / transpose /
weight generation for that algorithm. `compute_s` = algorithm call incl. its internal
allocation. `e2e_s = load_s + preprocess_s + compute_s`. Warm-up runs are discarded.
Speedups use `compute_s` unless labelled end-to-end.