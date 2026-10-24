# Track 1 — Baseline Implementations and Comparative Benchmarking

## Objective

Build correct, reproducible sequential and parallel implementations of five
graph algorithms, then establish a trustworthy empirical baseline across
available execution models.

The baseline study must establish when parallel execution helps, how
performance changes with graph size and structure, and how implementation
and execution overheads affect observed performance.

This track provides the reference implementations and empirical evidence
needed by the optimization and evaluation tracks.

## Research Question

**RQ1:** How do sequential, OpenMP, MPI, and CUDA implementations of
common graph algorithms compare across graph sizes and graph structures?

The execution models included in the final comparison will depend on
what can be implemented and tested correctly in the available environments.
Unavailable hardware or execution configurations must be reported rather
than simulated or inferred.

## Algorithms

Implement the following algorithms:

1. Breadth-First Search (BFS)
2. PageRank
3. Connected Components (CC)
4. Single-Source Shortest Paths (SSSP)
5. Triangle Counting

Each algorithm must have a trusted sequential reference implementation.
Implement vanilla parallel versions using the available execution models:

- OpenMP
- MPI
- CUDA

The initial parallel implementations should be straightforward, correct
implementations rather than prematurely optimized versions. Keep these
baselines available throughout the project.

## Graph Representation and Input Contract

- Use a shared graph representation where practical.
- Use Compressed Sparse Row (CSR) as the primary representation for
  computation.
- Define consistent graph-loading, indexing, directedness, and weighting
  conventions.
- Separate graph loading and preprocessing from algorithm execution
  in the benchmark records.
- Document any format conversions required by individual implementations.
- Use a documented set of small correctness-test graphs and larger
  benchmark graphs spanning different sizes and structural properties.

Record relevant graph characteristics where available, including:

- Number of vertices and edges
- Directed or undirected status
- Weighted or unweighted status
- Degree distribution and degree skew
- Graph family or source
- Other structural features relevant to a particular algorithm

Do not commit large datasets to the repository. Document where they come
from, how to obtain them, and how to reproduce preprocessing.

## Implementation Workflow

### Step 1 — Define contracts

Specify the input/output contract for each algorithm, graph conventions,
correctness criteria, and benchmark timing boundaries.

### Step 2 — Build shared infrastructure

Implement:

- CSR graph representation
- Graph loading and preprocessing
- Input validation
- Correctness checks
- Benchmark configuration
- Machine-readable result output
- Reproducible run scripts

### Step 3 — Implement sequential references

Implement all five algorithms sequentially and validate them on small,
hand-checkable graphs and appropriate test cases.

### Step 4 — Implement vanilla parallel baselines

Implement the available OpenMP, MPI, and CUDA versions. Keep the
baseline versions separate from optimized implementations so that later
comparisons remain valid.

### Step 5 — Validate correctness

Compare each parallel implementation against the relevant reference.
Test edge cases and document numerical tolerances and algorithm-specific
semantics.

### Step 6 — Benchmark

Run repeated experiments across selected graph sizes, graph structures,
thread/rank configurations, and available hardware.

Use a reproducible benchmark runner that records configuration,
environment, timings, and correctness results.

### Step 7 — Profile targeted algorithms

Profile the vanilla CUDA BFS and PageRank implementations. Use the
measurements to identify potential bottlenecks and hand the evidence
to Track 2. Do not assume in advance that a particular bottleneck or
optimization opportunity exists.

## Correctness Requirements

### BFS

- Compare distances from the source vertex.
- Correctly represent unreachable vertices.
- Verify results across all implementations.

### PageRank

Use consistent:
- Damping factor
- Initialization
- Dangling-node handling
- Convergence criterion and tolerance
- Maximum iteration count

Compare results using a documented numerical tolerance and report
iteration counts where relevant.

### Connected Components

Compare component membership rather than raw component labels, because
different correct implementations may assign different numeric labels.

Specify whether the experiment concerns weakly or strongly connected
components for directed graphs. Do not mix these definitions.

### SSSP

Initially use nonnegative edge weights unless negative weights are
explicitly supported and the chosen algorithm is valid for them.
Specify how unreachable vertices and distance values are represented.

### Triangle Counting

Specify:
- Directed or undirected graph semantics
- Whether self-loops and duplicate edges are allowed
- Whether each triangle is counted once or multiple times

Apply the same convention to all implementations.

## Benchmarking and Measurements

Record, where applicable:

- Algorithm and implementation
- Graph ID, family, vertex count, and edge count
- Run ID and configuration
- CPU model, thread count, MPI ranks, GPU model, and relevant software versions
- Graph-loading and preprocessing time
- Allocation and setup time
- Host-to-device and device-to-host transfer time
- Compute or kernel time
- MPI communication time
- End-to-end runtime
- Correctness status and validation tolerance
- Git commit and exact run configuration

Use repeated runs and report the median and interquartile range (IQR).
Document warm-up, synchronization, and other measurement procedures.

Report sequential speedup as:

    Speedup = T_sequential / T_parallel

Report parallel efficiency where it is meaningful and the processor
configuration is clearly defined. Use TEPS only when the work-counting
convention is clearly specified and comparable.

Do not compare kernel-only timings against end-to-end timings as if
they were equivalent. Keep the timing boundaries consistent.

## Deliverables

- Correct sequential references for all five algorithms
- Vanilla parallel implementations for feasible execution models
- Shared graph representation and correctness tests
- Reproducible benchmark scripts and configurations
- Raw, machine-readable benchmark records
- Summary tables and plots for baseline comparisons
- Environment and dataset documentation
- Profiling evidence for vanilla CUDA BFS and PageRank
- A record of implementations or configurations that could not be tested,
  with the reason clearly stated

## Completion Criteria

This track is complete when:

1. All sequential reference implementations pass their correctness tests.
2. Available parallel implementations are implemented and validated.
3. Repeated benchmark runs produce machine-readable results.
4. Timing boundaries and hardware/software environments are documented.
5. Baseline results are reproducible from documented commands.
6. Profiling evidence is available to guide optimization work.

## Handoff to Track 2

Provide the validated baseline implementations, benchmark runner,
raw measurements, graph metadata, and profiling evidence.

Track 2 must preserve the vanilla baselines unchanged and compare every
optimized implementation against an appropriate baseline.