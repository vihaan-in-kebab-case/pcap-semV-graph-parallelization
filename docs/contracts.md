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

**Timing:** `load_s` = file parse + CSR build. `preprocess_s` = symmetrization / transpose /
weight generation for that algorithm. `compute_s` = algorithm call incl. its internal
allocation. `e2e_s = load_s + preprocess_s + compute_s`. Warm-up runs are discarded.
Speedups use `compute_s` unless labelled end-to-end.