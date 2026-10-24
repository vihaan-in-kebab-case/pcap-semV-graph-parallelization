# Experimental Evaluation of Parallel Graph Analytics Using OpenMP, MPI, and CUDA

**A reproducible study of graph algorithm performance, profiling-guided optimization, and adaptive execution strategies.**

> **Project status:** In development  
> **Focus:** Parallel computing · Graph analytics · GPU optimization · Performance evaluation

## Overview

Graph analytics workloads can behave very differently depending on graph structure, algorithm design, memory access patterns, and execution hardware. Parallelization can accelerate computation, but synchronization, communication, workload imbalance, and setup overhead can also limit performance.

This project investigates these trade-offs through a comparative study of sequential and parallel graph algorithms, followed by targeted GPU optimization and adaptive strategy selection.

The goal is not simply to implement algorithms in multiple programming models, but to understand **when different execution strategies perform well, why they do, and whether measured performance differences can guide better execution decisions.**

The project combines three main components:

1. **Comparative benchmarking** — establish validated sequential and parallel baselines across graph algorithms, graph structures, and available hardware.
2. **Profiling-guided optimization** — identify bottlenecks in CUDA BFS and PageRank, implement targeted optimizations, and evaluate their impact.
3. **Adaptive optimization** — build a lookahead classifier that determines whether switching optimization strategies during execution is worth the expected cost.

A separate execution-selection pipeline is also planned as an extension. It will use empirical benchmark evidence to choose among measured algorithm implementations and execution frameworks for a given task and environment.

## Algorithms and execution models

### Algorithms

| Algorithm | Purpose |
|---|---|
| Breadth-First Search (BFS) | Explore graph neighborhoods and compute unweighted shortest-path distances from a source |
| PageRank | Estimate node importance through iterative graph-based ranking |
| Connected Components (CC) | Identify connected vertex groups under a specified connectivity definition |
| Single-Source Shortest Paths (SSSP) | Compute shortest-path distances from a source in a weighted graph |
| Triangle Counting | Measure the number of triangles in a graph under a documented counting convention |

### Execution models

| Model | Approach |
|---|---|
| Sequential CPU | Single-threaded reference implementation |
| OpenMP | Shared-memory CPU parallelism |
| MPI | Process-based parallelism using message passing |
| CUDA | Parallel execution on NVIDIA GPUs |

The sequential implementations provide correctness and performance references. OpenMP, MPI, and CUDA implementations will be evaluated where the required software and hardware are available.

The intended comparison covers graph sizes and structural characteristics rather than relying on a single benchmark graph.

## Research questions

**RQ1 — Comparative performance**

How do sequential, OpenMP, MPI, and CUDA implementations compare across algorithms, graph sizes, and graph structures?

**RQ2 — Optimization effectiveness**

Can profiling-guided optimizations improve CUDA BFS and PageRank over their vanilla implementations, and under what workload conditions?

**RQ3 — Generalization**

To what extent do observed optimization benefits generalize across graph structures, algorithms, and execution models?

These questions will be answered through measured experiments. Results will not be generalized to algorithms, hardware, or execution models that have not been tested.

## Optimization and adaptive execution

### Profiling-guided optimization

BFS and PageRank are the primary targets for deeper CUDA optimization.

Potential optimization areas include:

- Memory access and data locality
- Work scheduling and load balancing
- Frontier management and queue construction
- Allocation and temporary-buffer reuse
- Synchronization and reduction overhead

These are candidate directions, not assumed solutions. Profiling will guide which techniques are implemented, and every retained optimization will be compared against a stable vanilla baseline.

### Lookahead classifier

The lookahead classifier is part of the **core project scope**.

During execution, it evaluates whether switching from the current optimization strategy to an alternative is likely to improve the remaining runtime. Its decision accounts for factors such as:

- Current execution state and workload characteristics
- Estimated remaining work
- Predicted benefit of the alternative strategy
- Strategy setup and switching costs
- Feature-extraction and classifier overhead

The classifier will be evaluated against fixed strategies and rule-based alternatives, with end-to-end overheads included. It will be tested on held-out graphs to assess whether its decisions generalize beyond the data used during development.

### Execution-selection pipeline

A separate planned extension will use benchmark evidence to select an appropriate measured implementation or execution framework for a task, graph, and available hardware.

This is different from the lookahead classifier:

- **Lookahead classifier:** chooses whether to switch optimization strategies during execution.
- **Execution-selection pipeline:** chooses among measured implementations or execution frameworks for a task and environment.

The pipeline is a separate artifact and does not replace the core benchmarking, optimization, or classifier work.

## Methodology

The project follows a measurement-driven workflow:

1. **Establish correctness.** Implement sequential references and validate parallel implementations against them.
2. **Build reproducible baselines.** Benchmark the available execution models under documented configurations.
3. **Profile targeted workloads.** Identify reproducible bottlenecks in vanilla CUDA BFS and PageRank.
4. **Implement optimizations.** Test targeted strategies independently before evaluating combinations.
5. **Develop the lookahead classifier.** Evaluate whether adaptive switching improves end-to-end performance.
6. **Evaluate generalization.** Use held-out graphs, ablations, and workload-specific analysis.
7. **Document findings.** Report improvements, regressions, overheads, limitations, and reproducibility instructions.

### Evaluation metrics

Depending on the experiment, measurements will include:

- Runtime and runtime variability
- Speedup relative to the appropriate baseline
- Parallel efficiency and scaling where meaningful
- Kernel or compute time versus end-to-end runtime
- Memory, allocation, synchronization, transfer, and communication overheads
- Optimization and classifier overhead
- Policy regret and performance regression rate
- Correctness and numerical tolerance

Repeated experiments will report median runtime and interquartile range (IQR). CUDA kernel timing and end-to-end timing will use appropriate synchronization and clearly documented measurement boundaries.

## Repository structure

The repository is organized to separate stable implementations, optimization strategies, experiments, and analysis.

```text
graph-parallel-study/
├── README.md
├── CMakeLists.txt
├── config/
│   ├── benchmark.yaml
│   ├── datasets.yaml
│   ├── hardware.yaml
│   └── policies/
├── docs/
│   ├── project-scope.md
│   ├── research-questions.md
│   ├── related-work.md
│   ├── experimental-protocol.md
│   └── briefs/
│       ├── 01-baselines.md
│       ├── 02-optimization.md
│       └── 03-evaluation-generalization.md
├── include/
├── src/
│   ├── common/
│   ├── sequential/
│   ├── openmp/
│   ├── mpi/
│   ├── cuda/
│   ├── optimizations/
│   └── selection_pipeline/
├── tests/
├── benchmarks/
├── scripts/
├── datasets/
├── results/
├── analysis/
└── environment/
```

The exact contents will evolve as implementation progresses. Large datasets and generated benchmark outputs should not be committed indiscriminately; dataset sources, experiment configurations, and reproduction steps should be documented instead.

## Compute environment

The project is designed to accommodate cloud-based experimentation.

- **CPU and OpenMP:** a suitable multi-core CPU environment.
- **MPI:** an MPI-enabled environment; multi-node experiments require access to multiple machines or nodes.
- **CUDA:** a compatible NVIDIA GPU and CUDA toolkit.

GPU availability and multi-node MPI access may constrain the final experiment matrix. Unsupported configurations will be documented rather than inferred from results collected on different hardware.

Environment setup, software versions, benchmark configurations, and relevant hardware specifications will be recorded to support reproducibility.

## Current status and roadmap

The intended milestones are:

- [ ] Shared graph representation and correctness infrastructure
- [ ] Sequential reference implementations for all five algorithms
- [ ] Vanilla OpenMP, MPI, and CUDA implementations where feasible
- [ ] Reproducible baseline benchmark suite
- [ ] Profiling of CUDA BFS and PageRank
- [ ] Profiling-guided optimization implementations
- [ ] Lookahead classifier and policy evaluation
- [ ] Held-out evaluation, ablations, and generalization analysis
- [ ] Final research report and reproducibility package
- [ ] Separate execution-selection pipeline and evaluation

This checklist represents the planned work, not a claim that the milestones are already complete.

## Research integrity

Performance comparisons are only meaningful when correctness, hardware conditions, and timing boundaries are clearly defined.

This project aims to:

- Preserve vanilla baselines for fair comparison.
- Include relevant overheads in end-to-end measurements.
- Report negative results and regressions.
- Distinguish demonstrated findings from hypotheses and future work.
- Avoid unsupported claims about novelty or generalization.
- Provide sufficient code, configuration, and experimental records to reproduce key results.

The intended contribution is a reproducible empirical study of parallel graph analytics, measured optimization strategies, and adaptive execution decisions—not an assumed claim of inventing a new graph algorithm.

Contributions, issues, and suggestions related to correctness, benchmark methodology, optimization strategies, and reproducibility are welcome. Proposed performance improvements should ideally include a clear hypothesis, a reproducible experiment, and results against an appropriate baseline.# pcap-semV-graph-parallelization
