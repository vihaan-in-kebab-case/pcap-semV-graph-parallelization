# Environment

Records the machines and software behind every reported number. Each run also writes
`results/raw/<RUN_TAG>_environment.txt` (via `scripts/common/collect-environment.sh`), which is the
authoritative record for that run. This file describes the setup in general.

## Machines
| Role | Machine | CPU | Cores | RAM | OS | GPU |
|---|---|---|---|---|---|---|
| Development | Apple Silicon laptop | `<fill>` | 8 physical | `<fill>` | macOS `<fill>` (arm64) | none |
| Reference for final results | `<fill: cloud pod / server>` | `<fill>` | `<fill>` | `<fill>` | Linux `<fill>` | `<fill: model, driver, memory>` |

Development-machine numbers are for correctness and trend checking only. Final numbers come from the
reference machine, with the sequential baseline measured on that same machine.

## Software
| Component | Development machine | Reference machine |
|---|---|---|
| Compiler | AppleClang 17.0.0.17000013 | `<fill: GCC/Clang version>` |
| C++ standard | C++17 | C++17 |
| CMake | `<fill>` (project requires 3.18+) | `<fill>` |
| OpenMP | 5.1 (Homebrew `libomp`) | `<fill>` |
| MPI | Open MPI (Homebrew) `<fill version>` | `<fill>` |
| CUDA toolkit | not available | `<fill>` (11+ for C++17 device code) |
| Python | `<fill>` in `.venv`: pandas, matplotlib, numpy `<fill versions>` | same |

## Build configuration
- Build type `Release` (CMake default optimization flags); warnings `-Wall -Wextra` on C++ code.
- macOS: libomp flags are passed explicitly by `scripts/build.sh`.
- CUDA architecture is read from the installed GPU (`nvidia-smi`), else defaults to 75, 80, 86;
  override with `CUDA_ARCH`.
- MPI and CUDA targets build only if MPI / `nvcc` are found.

## Runtime settings
- OpenMP thread counts: powers of two up to physical cores. On Linux, `OMP_PROC_BIND=close` and
  `OMP_PLACES=cores`. macOS has no thread pinning.
- MPI: single node only, ranks as processes on one machine; `--oversubscribe` is added when ranks
  exceed cores. On macOS `TMPDIR=/tmp` avoids an Open MPI path-length error.
- CUDA: device 0 by default (`--device N`); CUDA context created before timing; kernels timed with
  CUDA events.
- Random inputs: SSSP weights are integers in 1..100 from a hash of the endpoint pair and seed 42;
  the BFS/SSSP source is the highest out-degree vertex (compact vertex id).

## Datasets
| Graph | File | SHA-256 |
|---|---|---|
| ego-Facebook (SNAP) | `datasets/graphs/facebook_combined.txt` | f41c026ed8af3cc3359f1ca5573d0605fb09ae0eefa34544b820fd8c6e2ef296 |
| web-Google (SNAP) | `datasets/graphs/web-Google.txt` | 8c0f453f1eb1e24ad145e36e542b129083237e96e585abae768927bdb70167d1 |

Source and usage details are in `datasets/README.md`.

## Reproducing the setup
    scripts/common/install-deps.sh     # compiler, CMake, OpenMP, MPI, Python
    scripts/fetch-datasets.sh
    scripts/build.sh
    scripts/common/run-tests.sh
Linux and macOS are supported. Windows has not been tested; WSL2 with Ubuntu should work unchanged.

## Configurations not available (reported, not simulated)
- Multi-node MPI: only single-node, shared-memory MPI is available.
- CUDA: `<fill: untested / tested on model X>`; update before reporting any GPU result.
- Hardware counters for Nsight profiling: `<fill: available / restricted on this machine>`.

## Known limitations
- Apple Silicon mixes performance and efficiency cores, and laptops throttle thermally.
- Cloud machines are shared; repeat runs and report IQR.
- Results from different machines must not be pooled.