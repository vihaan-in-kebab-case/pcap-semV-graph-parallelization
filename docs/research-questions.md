# Research questions

Answered only as far as the experiments support. Phase 1 covers RQ1 and produces the gap list that
feeds RQ2. Phase 2 optimizes CUDA only.

## RQ1 — Comparative performance (Phase 1)
How do sequential, OpenMP, MPI and CUDA implementations of BFS, PageRank, Connected Components, SSSP
and Triangle Counting compare across graph sizes and structures?

Sub-questions
1. When does parallel execution beat sequential, per algorithm, graph and worker count?
2. How much do loading, preprocessing, setup/transfer and communication change the picture?
   (compute-only versus end-to-end)
3. How does scaling behave: efficiency, saturation, and the communication share under MPI?
4. Which implementation differences confound the comparison? Known ones: MPI connected components uses
   label propagation instead of union-find; MPI triangle counting replicates the graph; block vertex
   partitioning is unbalanced on skewed graphs.

Evidence: speedup (`T_seq / T_parallel`, compute basis, end-to-end reported alongside), parallel
efficiency for OpenMP and MPI, median and IQR over repeated trials, communication share, end-to-end
breakdown. Figures 1 to 5 from `scripts/plot_results.py`.

Expectations to test, not assumed: small graphs are dominated by parallel overhead; end-to-end time
is dominated by loading and preprocessing for the larger graph; scaling saturates before the core
count; communication-heavy MPI algorithms scale worst.

Scope limits: two graphs so far (a small undirected social graph and a larger directed web graph) are not
enough to make claims about graph structure. More families are needed (to be selected). Single-node
MPI only. GPU model `<fill>`.

## RQ2 — Optimization effectiveness (Phase 2, CUDA only)
Can profiling-guided optimizations improve CUDA BFS and PageRank over their vanilla implementations,
and under what workload conditions?

Evidence required: profiling of the vanilla implementations first (bottlenecks measured, not assumed),
each optimization evaluated individually against the unchanged baseline, then combinations;
`T_vanilla / T_optimized`, regression rate, and the lookahead classifier measured end to end with its
own overhead against fixed strategies, a rule-based policy and an offline oracle, on held-out graphs.
Phase 1 supplies the baselines and gap candidates (see the repository guide and the gap log below).

## RQ3 — Generalization
To what extent do the observed benefits generalize across graph structures, algorithms and execution
models?

Only claims supported by what was tested: CUDA BFS and PageRank results do not extend to other
algorithms or to OpenMP/MPI. Generalization across graphs is tested on held-out graphs.

## Gap log (Phase 1 findings to carry forward)
- Partition imbalance under equal-vertex block partitioning.
- BFS and SSSP send one message per cross-rank edge with no combining.
- Label propagation needs diameter-many rounds with a full allgather each round.
- Triangle counting replicates the CSR on every rank.
- Frontier SSSP can re-relax vertices; delta-stepping is a candidate improvement.