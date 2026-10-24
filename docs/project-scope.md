# Project Overview — Experimental Evaluation of Parallel Graph Analytics Using OpenMP, MPI, and CUDA on Large-Scale Networks

## 1. Context and Motivation

Graph analytics algorithms are used to analyze networks in areas such as information retrieval, social networks, scientific computing, and machine learning. Their performance depends not only on the algorithm itself, but also on graph structure, workload distribution, memory access patterns, synchronization, and the execution hardware.

Although parallel computing can accelerate graph processing, the benefits are not uniform. Different algorithms, graph structures, and execution models can favor different implementation strategies, and optimization overhead can sometimes outweigh the gains.

This project investigates these trade-offs through an empirical study of sequential and parallel graph analytics, followed by profiling-guided optimization and an adaptive optimization policy.

## 2. Overall Goal

The primary goal is to **build, benchmark, optimize, and systematically evaluate graph analytics implementations across sequential CPU execution, OpenMP, MPI, and CUDA**, then investigate whether an adaptive lookahead classifier can improve performance by switching between optimization strategies when the expected benefit justifies the switching cost.

The project should produce a reproducible experimental framework, validated implementations, quantitative performance comparisons, and evidence-based conclusions about when particular execution strategies work best.

It is a systems-oriented experimental research project, not simply an implementation exercise. Correctness, controlled experiments, profiling, overhead accounting, ablation studies, and reproducibility are essential.

## 3. Core Scope

### Graph algorithms

Implement five algorithms:

1. Breadth-First Search (BFS)
2. PageRank
3. Connected Components
4. Single-Source Shortest Paths (SSSP)
5. Triangle Counting

Develop sequential reference implementations and vanilla parallel implementations using the available OpenMP, MPI, and CUDA environments. Use a common graph representation, primarily Compressed Sparse Row (CSR), and consistent correctness criteria.

### Comparative benchmarking

Evaluate performance across graph sizes and structural characteristics using repeated, reproducible experiments. Measure runtime, speedup, scaling where supported, correctness, and relevant execution overheads.

Distinguish compute time from end-to-end runtime, including preprocessing, allocation, transfers, communication, and other applicable costs. Hardware and execution configurations that cannot be tested must be documented explicitly.

### Profiling-guided optimization

Focus deeper optimization efforts on **CUDA BFS and PageRank**. Profile their vanilla implementations, identify reproducible bottlenecks, and implement actual optimization strategies based on the evidence.

Candidate areas include memory locality, work scheduling, load balancing, synchronization, allocation, and frontier or workload management. These are possibilities rather than assumptions: select techniques according to measured bottlenecks and validate their effects experimentally.

Preserve the vanilla implementations as stable baselines. Evaluate optimization variants independently before studying combinations.

### Lookahead classifier — core requirement

Implement a lookahead classifier that decides whether to continue with the current optimization strategy or switch to an alternative during execution.

The decision should consider available execution-state features, predicted remaining-work savings, setup and switching costs, and decision overhead. Its purpose is to improve realized end-to-end performance, not merely predict which strategy is fastest in hindsight.

Evaluate it against fixed strategies and a transparent rule-based policy, using held-out graphs and accounting for all policy overheads. An offline oracle may be included as an upper-bound reference.

The classifier is a required part of the core optimization work, not an optional extension. It must operate over genuinely implemented and measured strategies.

## Important for all implementations:
1. Timing contract
Every implementation should record the relevant components separately:
- Graph loading and preprocessing.
- Memory allocation.
- Host-to-device and device-to-host transfers.
- Policy feature extraction.
- Classifier inference.
- Strategy or work-queue setup.
- Kernel/compute execution.
- MPI communication, where applicable.
- End-to-end runtime.
Use CUDA events for GPU execution timing and synchronized wall-clock measurements for end-to-end time. Clearly define whether preprocessing is included in each headline result; never compare a kernel-only measurement against another implementation's end-to-end runtime.

2. Performance metrics
At minimum, the benchmark and evaluation briefs should require:
- Median runtime and IQR across repeated trials.
- Sequential speedup: \(T_{\text{seq}}/T_{\text{parallel}}\).
- Optimization speedup: \(T_{\text{vanilla}}/T_{\text{optimized}}\).
- Parallel efficiency where the worker count is meaningful.
- Edge throughput where the work definition is comparable.
- Policy overhead as a fraction of end-to-end runtime.
- Correctness, scaling behaviour and regression rate.

## 4. Separate Extension: Execution-Selection Pipeline

In addition to the core study, develop an execution-selection pipeline if time permits.

This pipeline uses empirical benchmark evidence to select a suitable measured algorithm implementation or execution framework for a requested task, graph characteristics, and available hardware. It may select an optimization configuration only where that configuration has been implemented and evaluated for the relevant workload.

This is distinct from the lookahead classifier:

- **Lookahead classifier:** decides whether to switch optimization strategies during an algorithm's execution.
- **Execution-selection pipeline:** chooses among measured implementations or execution frameworks for a task and environment.

The pipeline is a separate artifact. It must not replace the benchmarking study, actual optimization implementations, or core classifier. Evaluate it independently against simple selection policies and the best measured candidate, with selection overhead and performance regressions reported.

## 5. Research Questions

**RQ1 — Comparative performance:** How do sequential, OpenMP, MPI, and CUDA implementations of the five graph algorithms compare across graph sizes and structures?

**RQ2 — Optimization effectiveness:** Can profiling-guided execution optimizations improve CUDA BFS and PageRank over their vanilla implementations, and under what workload conditions?

**RQ3 — Generalization:** To what extent do the observed optimization benefits generalize across graph structures, algorithms, and execution models?

Answer each question only to the extent supported by the experiments. Results from CUDA BFS and PageRank alone do not establish generalization to other algorithms or execution models.

## 6. Experimental Principles

- Validate correctness before comparing performance.
- Profile before selecting optimizations; do not assume a bottleneck without evidence.
- Preserve baseline implementations and raw experimental records.
- Use repeated runs and report median runtime and interquartile range.
- Compare implementations using consistent timing boundaries and equivalent correctness requirements.
- Include optimization, classifier, and switching overhead in end-to-end results.
- Use held-out graphs to evaluate adaptive policies and avoid data leakage.
- Report regressions, negative results, and inconclusive experiments.
- Keep conclusions proportional to the hardware, graph families, algorithms, and configurations actually tested.
- Conduct a related-work review before making novelty claims.

The intended contribution is a reproducible comparative study, measured optimization improvements, an evaluated lookahead switching policy, and—if completed—a separate empirical execution-selection artifact. Novelty must be established through related-work analysis rather than assumed.

## 7. Deliverables and Constraints

The final project should contain:

- Sequential and feasible parallel implementations of the five algorithms
- Correctness tests and shared graph infrastructure
- Reproducible benchmark configurations, scripts, and raw results
- Profiling reports for CUDA BFS and PageRank
- Actual optimization implementations and their evaluations
- The lookahead classifier, policy traces, and overhead analysis
- Ablation and held-out evaluation results
- A final report or paper draft, figures, and documented limitations
- The separate execution-selection pipeline and evaluation, if completed

The project has a three-person team and an estimated four-to-six-week implementation window. Compute is expected to come from cloud resources; NVIDIA GPU availability and multi-node MPI access are uncertain. The implementation plan must therefore be adaptable to available hardware.

The five algorithms across the sequential reference and three parallel approaches define the target comparison matrix, not a guarantee that every combination will receive equal optimization effort. Prioritize correctness and reproducible baselines first, then the core optimization and classifier contributions. Reduce implementation breadth transparently if necessary rather than compromising experimental validity.

## 8. How to Use the Supporting Briefs

This document establishes the shared project context and scope. The three supporting briefs define the work in greater detail:

- **01 — Baselines:** shared infrastructure, correctness, vanilla implementations, and comparative benchmarking.
- **02 — Optimization:** profiling, actual CUDA BFS and PageRank optimization strategies, and the required lookahead classifier.
- **03 — Evaluation and Generalization:** experimental analysis, ablations, limits of generalization, final reporting, and the separate execution-selection extension.

All implementation decisions should remain consistent with this overview. If a supporting brief or implementation decision appears to change the scope, resolve the discrepancy explicitly rather than silently expanding or replacing the core objectives.