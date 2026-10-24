# Track 3 — Evaluation, Generalization, and Research Conclusions

## Objective

Evaluate the baseline implementations, implemented optimizations, and
lookahead classifier using a consistent experimental methodology.

Determine which performance differences are reproducible, which graph
or execution characteristics explain them, and how far the findings
generalize beyond the workloads used to develop the implementations.

The conclusions must be proportional to the experiments actually run.
A result observed on CUDA BFS, for example, is not evidence that the
same optimization improves MPI BFS or PageRank.

## Research Questions

**RQ1:** How do sequential, OpenMP, MPI, and CUDA implementations of
common graph algorithms compare across graph sizes and graph structures?

**RQ2:** Can targeted execution optimizations improve CUDA BFS and
PageRank over their vanilla implementations?

**RQ3:** To what extent do optimization principles generalize across
algorithms, graph structures, and execution models?

RQ1 is addressed by Track 1 and consolidated here.
RQ2 is addressed by Track 2 and consolidated here.
RQ3 is addressed only for the algorithms, graph structures, and
execution models directly tested.

## Experimental Design

### Graph coverage

Use a documented collection of graphs covering a range of sizes and
structural properties.

Where practical, include different graph families, degree distributions,
degree skew, sparsity patterns, and workload sizes.

Separate graphs used for development and tuning from graphs used for
held-out evaluation. Keep all observations from one graph within the
same split when evaluating learned policies.

Document missing coverage. If an important graph family or hardware
configuration cannot be tested, state that limitation rather than
extrapolating to it.

### Repeated measurements

- Run experiments multiple times under documented conditions.
- Report median runtime and interquartile range (IQR).
- Record hardware, software, configuration, and commit information.
- Document warm-up, synchronization, and timing procedures.
- Investigate noisy or anomalous measurements rather than silently
  discarding inconvenient results.

Use confidence intervals or additional statistical analysis where
appropriate and feasible.

### Fair comparisons

Ensure comparisons use consistent:

- Input graphs and graph semantics
- Correctness requirements
- Numerical tolerances
- Hardware and resource configurations
- Timing boundaries
- Warm-up and synchronization procedures
- Preprocessing and transfer accounting

Report the configurations that differ between implementations.
Do not imply that two execution models received equivalent resources
when their environments differ materially.

## Metrics

### Runtime and speedup

Report runtime using median and IQR.

Sequential speedup:

    Speedup = T_sequential / T_parallel

Optimization speedup:

    Optimization speedup = T_vanilla / T_optimized

A value greater than one indicates a speedup under the stated timing
definition. Clearly state which timing boundary is being compared.

### Scaling

Where supported by the available hardware and experiment design, report:

- Strong-scaling behavior
- Parallel efficiency
- Runtime as graph size increases
- The point at which parallel execution overhead outweighs its benefits

Do not infer multi-node MPI scaling from single-node MPI measurements.
If multi-node access is unavailable, restrict conclusions to the
configurations actually tested.

### Throughput

Report TEPS or other throughput measures only when the definition of
processed work is clear and comparable across implementations.

For iterative algorithms such as PageRank, report relevant iteration
counts and convergence criteria alongside timing.

### Overheads

Separate the relevant components of total runtime:

- Graph loading and preprocessing
- Memory allocation and initialization
- Host-to-device and device-to-host transfers
- Compute or kernel time
- Queue construction and strategy setup
- Synchronization and reductions
- MPI communication
- Feature extraction
- Classifier inference
- Strategy switching
- Validation and other end-to-end costs

Avoid double-counting overlapping timing components. Document the
measurement boundaries used for each reported result.

### Lookahead classifier

Evaluate:

- Realized end-to-end speedup against fixed policies
- Benefit after feature extraction, inference, and switching costs
- Regret relative to the best measured policy
- Regression rate and severity
- Number and timing of strategy switches
- Performance on held-out graphs
- Cases where the classifier makes a good decision and cases where
  it fails

Use an offline oracle only as an explicitly labelled upper-bound
reference, not as a directly deployable competitor.

## Generalization Experiments

### Across graph structures

Test whether observed improvements persist across held-out graphs and
structural regimes.

Distinguish between:
- Performance on graphs used during development
- Performance on held-out graphs from familiar families
- Performance on graph families not used during development, if tested

Do not describe evaluation on a few graphs as universal generalization.

### Across algorithms

Where time permits, directly test selected optimization principles
or policy components on another algorithm, such as Connected Components
or Triangle Counting.

This is an extension to the core BFS and PageRank optimization work,
unless explicitly added to the implementation plan.

Only claim cross-algorithm transfer when the relevant implementation
and evaluation have actually been completed.

### Across execution models

If an optimization is implemented only in CUDA, report the result as
a CUDA result.

To claim that a principle generalizes to OpenMP or MPI, implement and
test an appropriate version under those execution models. Differences
in synchronization, memory hierarchy, communication, and scheduling
may change the result.

## Ablations

Use ablations to identify which components account for observed
improvements.

Where relevant, compare:

- Vanilla implementation
- Each optimization independently
- Selected combinations of optimizations
- Fixed strategies versus the lookahead classifier
- Classifier decisions with and without selected features
- Performance with switching costs included
- Performance with different decision thresholds or safety margins

Ablations should answer specific questions. Do not include combinations
solely to increase the number of experiments.

## Separate Extension — Execution-Selection Pipeline

### Purpose

Build an additional artifact that uses empirical benchmark evidence
to select an appropriate measured implementation for a graph task
and available execution environment.

This is distinct from the lookahead classifier:

- **Lookahead classifier:** decides whether to switch optimization
  strategies during an algorithm's execution.
- **Execution-selection pipeline:** chooses among measured algorithm
  implementations or execution frameworks before a run, using task,
  graph, and hardware information.

The pipeline may select an applicable optimization configuration
only when that configuration has actually been implemented and
measured for the relevant combination.

### Inputs

Potential inputs include:

- Requested graph algorithm
- Graph size and structural features
- Available CPU, thread, GPU, and MPI configurations
- Relevant implementation constraints
- Benchmark-derived performance estimates

The pipeline must not recommend an implementation that is unavailable
or incompatible with the requested task and hardware.

### Implementation

1. Build a structured dataset from validated benchmark records.
2. Define which implementation/configuration combinations are feasible.
3. Define a selection objective, such as minimizing expected
   end-to-end runtime.
4. Implement a selection policy using measured data.
5. Evaluate the policy on held-out graphs and configurations.
6. Record its predictions, chosen implementation, and realized runtime.

A simple, interpretable policy is preferable to a complex learned
selector unless the data supports the added complexity.

### Evaluation

Compare the pipeline against:

- A fixed default implementation
- A simple rule-based selector, where appropriate
- The best measured candidate for each evaluation case as an
  explicitly labelled offline reference

Measure end-to-end runtime, selection overhead, selection regret,
regression rate, and the frequency with which the selector chooses
a slower feasible implementation.

The evaluation must not leak test-graph performance into the selector's
training or tuning process.

### Status and scope

This pipeline is a separate extension/artifact, not a replacement for
the baseline study, optimization implementations, or lookahead classifier.

The core project should remain scientifically complete if the extension
cannot be finished within the available time. If it is implemented,
its claims must be supported by a separate evaluation.

## Reporting and Research Integrity

The final report must:

- Answer each research question explicitly.
- Include results that show improvements and regressions.
- Distinguish measured results from hypotheses and future work.
- Explain implementation and hardware constraints.
- Report negative or inconclusive experiments where informative.
- Avoid claiming novelty until a targeted related-work review has
  established what is already known.
- Avoid claiming that an implementation is poorly optimized merely
  because it loses to another implementation.
- Distinguish algorithmic differences from implementation quality,
  hardware effects, and measurement overhead.

Potential contribution claims should focus on what the experiments
actually establish: a reproducible comparative study, profiling-guided
optimization results, a measured lookahead switching policy, and,
if completed, an empirical execution-selection artifact.

Do not claim a novel graph algorithm unless the work genuinely develops
one and the related-work review supports that claim.

## Deliverables

- Final benchmark dataset and metadata
- Reproducible evaluation scripts and configurations
- Tables and figures answering RQ1–RQ3
- Ablation and held-out evaluation results
- Lookahead classifier evaluation
- Documented limitations and failed or inconclusive experiments
- Related-work review and appropriately scoped contribution statement
- Final report or paper draft
- Optional execution-selection pipeline and its independent evaluation
- Instructions for reproducing key results

## Completion Criteria

This track is complete when:

1. The main research questions are answered using recorded evidence.
2. Baseline and optimized comparisons use consistent timing definitions.
3. Repeated measurements and variability are reported.
4. The lookahead classifier is evaluated against relevant alternatives.
5. Generalization claims match the tested algorithms, graphs, and
   execution models.
6. Negative results and regressions are not concealed.
7. The results, configurations, code, and environment are sufficiently
   documented for reproduction.
8. The execution-selection pipeline, if completed, is evaluated as a
   separate artifact.

## Final Handoff

Produce a reproducible research package containing the source code,
configuration files, benchmark data, analysis scripts, figures,
environment documentation, and final report.

The final report should make clear which results belong to the core
study and which belong to the optional execution-selection extension.