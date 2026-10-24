# Track 2 — Profiling-Guided Optimization and Lookahead Classification

## Objective

Improve the performance of CUDA BFS and PageRank through actual,
profiling-guided implementation of optimization techniques.

Build a lookahead classifier that can decide whether switching between
available optimization strategies is worthwhile during execution. The
decision must account for the predicted benefit, remaining work, and
the cost of switching or preparing the alternative strategy.

This track includes three required components:

1. Identifying and validating performance bottlenecks
2. Implementing and measuring actual optimization strategies
3. Implementing and evaluating the lookahead classifier

The classifier is not a substitute for optimization implementations.
It must choose among genuinely implemented strategies whose behavior
and costs have been measured.

## Research Questions

**RQ2:** Can targeted execution optimizations improve CUDA BFS and
PageRank over their vanilla implementations?

**RQ3:** To what extent do optimization principles generalize across
algorithms, graph structures, and execution models?

RQ3 must be answered according to the evidence actually collected.
If optimizations are implemented only for CUDA BFS and PageRank, the
project can establish findings about those workloads, but cannot claim
that the same improvements generalize to OpenMP, MPI, or other algorithms
without implementing and testing them there.

## Scope

- Preserve the vanilla CUDA BFS and PageRank implementations.
- Profile both before choosing optimization techniques.
- Implement a manageable number of real optimization variants.
- Measure each variant independently before combining techniques.
- Build a lookahead classifier that selects or switches between
  implemented strategies.
- Include feature extraction, setup, switching, and inference costs
  in end-to-end evaluation.
- Maintain correctness and reproducibility throughout.

The optimization work should be deep enough to support a meaningful
experimental comparison. Implementing many untested techniques is
less valuable than implementing a few techniques well and explaining
when they help or hurt.

## Stage 1 — Profile Before Optimizing

Profile the vanilla CUDA implementations and identify reproducible
bottlenecks.

### BFS profiling

Measure, where available:

- Per-level runtime
- Frontier size and frontier fraction
- Number of active edges processed per level
- Active-edge-to-vertex ratio
- Degree distribution and degree skew
- Work imbalance across threads, blocks, or partitions
- Queue construction and frontier compaction overhead
- Allocation and synchronization overhead
- GPU utilization, occupancy, memory throughput, and cache behavior
- Kernel-launch overhead and data-transfer costs

Use the measurements to form specific optimization hypotheses.

For example, a high-degree-skew graph with measurable load imbalance
may motivate an alternative work-assignment strategy. That strategy
should only be retained if experiments show a reproducible benefit
after accounting for its own overhead.

Do not assume BFS is always memory-bound, that frontier size alone
predicts performance, or that adaptive scheduling must outperform a
fixed strategy.

### PageRank profiling

Measure, where available:

- Runtime per iteration
- Edge-processing throughput
- Memory-access and cache behavior
- Work distribution and partition balance
- Synchronization and reduction costs
- Allocation and initialization costs
- Kernel-launch overhead
- Data-transfer and end-to-end runtime
- GPU utilization and memory throughput

Use these observations to select the optimization experiments.
Do not assume that a technique effective for BFS will also improve
PageRank.

## Stage 2 — Implement Optimization Strategies

Implement actual alternatives based on the measured bottlenecks.
Potential categories include:

- Memory access and data locality
- Work scheduling and load balancing
- Vertex-oriented versus edge-oriented work assignment
- Frontier representation and queue construction
- Allocation and temporary-buffer reuse
- Synchronization and reduction overhead
- Partitioning or other workload-distribution strategies

These are candidate categories, not a requirement to implement every
technique. Select techniques according to profiling evidence, expected
benefit, implementation cost, and available time.

### BFS examples

Depending on profiling results, possible alternatives include:

- Vertex-oriented traversal versus an edge-balanced strategy
- Alternative frontier construction or compaction approaches
- Alternative work scheduling for degree-skewed graphs
- Reuse of buffers or reductions in avoidable setup overhead

### PageRank examples

Depending on profiling results, possible alternatives include:

- Alternative edge partitioning
- Improved memory locality
- Reuse of allocated buffers
- Reduced synchronization or reduction overhead
- Alternative work-distribution strategies

For every implemented optimization:

1. State the observed bottleneck.
2. State the optimization hypothesis.
3. Implement the change behind a clearly defined interface.
4. Verify correctness.
5. Measure it against the vanilla baseline.
6. Record the workload regimes where it helps or hurts.
7. Measure the overhead it introduces.
8. Retain or reject it based on evidence.

Keep optimization components modular enough to support ablations.
Avoid duplicating entire baseline implementations when a feature can
be represented as a separate strategy or configurable component.

## Stage 3 — Build the Lookahead Classifier

### Purpose

The lookahead classifier is a core component of this track. It decides
whether to continue using the current strategy or switch to an
alternative strategy during execution.

Its purpose is not merely to predict which strategy has the lowest
runtime on an entire graph. It must estimate whether switching now is
likely to produce a net benefit, given the state of the current execution.

### Decision inputs

Use features that are genuinely available at the time of the decision,
such as:

- Current algorithm and optimization strategy
- Current frontier size or other algorithm-specific execution state
- Remaining estimated work
- Graph features and relevant workload statistics
- Recent execution timings
- Estimated remaining runtime under the current strategy
- Estimated remaining runtime under an alternative strategy
- Strategy setup, queue construction, allocation, or switching costs
- Classifier feature-extraction and inference costs

The precise feature set will depend on the algorithm and the implemented
strategies. Do not use future execution information unavailable at
decision time.

### Decision objective

The classifier should estimate whether the expected savings from
switching exceed the costs of switching and preparing the alternative.

A conceptual decision rule is:

    Switch if:
        predicted remaining-time savings
        > switching cost + setup cost + decision overhead + safety margin

The safety margin should help avoid switches based on small or uncertain
predicted gains. If the expected benefit is too small or uncertain,
fall back to the current or default strategy.

This is a conceptual objective, not a mandated mathematical model.
The final implementation must define how its estimates are computed
and validate them experimentally.

### Training and decision data

Before training a classifier, establish that different fixed strategies
actually win in different workload or execution-state regimes. If one
strategy dominates everywhere tested, a learned switching policy may
not be justified.

Create training examples from measured execution traces and controlled
strategy comparisons. Define the label or target from end-to-end cost,
including the overheads that would actually be incurred.

Use graph-level train/validation/test splits: all observations from
a given graph must remain in the same split. This reduces leakage
between correlated execution states from the same graph.

Start with an interpretable approach, such as a shallow decision tree
or a small rule-based model. Increase model complexity only if the
data and evaluation justify it.

### Evaluation

Compare at least:

1. A fixed default strategy
2. The best fixed strategy selected using training data
3. A transparent hand-written rule policy
4. The lookahead classifier policy
5. An offline oracle or best-measured policy as an upper-bound reference

The offline oracle is a comparison reference, not a deployable policy:
it may use information that is unavailable to an online decision-maker.
Clearly explain the distinction.

Measure:

- End-to-end runtime and speedup
- Classifier inference overhead
- Feature-extraction overhead
- Strategy setup and switching overhead
- Number of switches
- Policy decision accuracy, where a defensible label is available
- Regret relative to the best measured policy
- Regression rate: cases where the policy makes performance worse
- Net benefit after all policy overheads

Report results by graph size, graph structure, and execution regime,
not just as one aggregate average.

The classifier must be evaluated on held-out graphs. If there is
insufficient data to train a reliable model, reduce its complexity
or collect better data; do not present an unvalidated heuristic as
a successful learned classifier.

## Stage 4 — Ablations and Robustness

Where applicable, test:

- Each optimization independently
- Relevant combinations of optimizations
- The fixed strategy without the classifier
- The policy with and without selected features
- The effect of switching/setup overhead
- The effect of the decision threshold or safety margin
- Cases where the classifier declines to switch
- Cases where switching produces a regression

Keep raw results and decision traces so that policy behavior can be
reconstructed.

## Measurement Rules

Use repeated runs and report median and IQR.

For each relevant experiment, distinguish:

- Kernel or compute time
- Graph preprocessing and allocation
- Queue or strategy setup
- Feature extraction
- Classifier inference
- Switching overhead
- Data transfers
- Synchronization
- End-to-end runtime

Use CUDA events for kernel timing and appropriately synchronized
wall-clock timing for end-to-end measurements.

Include all policy and optimization overheads in the headline
end-to-end comparison. Do not claim an end-to-end improvement using
kernel-only timings.

Compare optimized and vanilla implementations on the same graph,
hardware, correctness requirements, and measurement boundaries.

## Deliverables

- Profiling reports for vanilla CUDA BFS and PageRank
- Documented bottleneck hypotheses
- Actual implemented optimization strategies
- Correctness tests for every variant
- Per-strategy benchmark results
- Lookahead classifier implementation
- Reproducible policy training and evaluation scripts
- Policy decision traces and overhead measurements
- Ablations and regression analysis
- Updated literature review documenting related approaches
- A record of unsuccessful optimizations and the conditions where
  they failed

## Completion Criteria

This track is complete when:

1. The selected bottlenecks are supported by measurements.
2. Multiple viable strategies are implemented where the classifier
   is expected to choose between them.
3. Every retained strategy is tested against the vanilla baseline.
4. The lookahead classifier is implemented and evaluated against
   fixed and rule-based alternatives.
5. All policy overheads are included in end-to-end results.
6. Improvements, regressions, and limitations are documented.
7. Code, data, and commands are sufficient to reproduce the experiments.

## Handoff to Track 3

Provide baseline and optimized implementations, raw measurements,
graph metadata, policy configurations, decision traces, and evaluation
scripts.

Track 3 will use these artifacts to assess the research questions,
robustness, and limits of generalization.