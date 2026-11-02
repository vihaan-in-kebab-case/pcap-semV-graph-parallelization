#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "algorithm_api.hpp"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error: %s (%s:%d)\n",                         \
                   cudaGetErrorString(err_), __FILE__, __LINE__);              \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace gpu {

// Per-call breakdown (seconds). compute_s = CUDA events; the rest =
// synchronized wall clock.
struct Times {
  double alloc_s = 0, h2d_s = 0, d2h_s = 0, compute_s = 0;
};

inline double wall_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct GpuTimer {
  cudaEvent_t a{}, b{};
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
  }
  ~GpuTimer() {
    cudaEventDestroy(a);
    cudaEventDestroy(b);
  }
  GpuTimer(const GpuTimer &) = delete;
  GpuTimer &operator=(const GpuTimer &) = delete;
  void start() { CUDA_CHECK(cudaEventRecord(a)); }
  double stop() {
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms / 1000.0;
  }
};

template <class T, class H>
void download_ptr(const T *src, std::vector<H> &host, size_t count,
                  double &d2h_s) {
  static_assert(sizeof(T) == sizeof(H), "element size mismatch");
  host.resize(count);
  const double t0 = wall_s();
  CUDA_CHECK(
      cudaMemcpy(host.data(), src, count * sizeof(T), cudaMemcpyDeviceToHost));
  d2h_s += wall_s() - t0;
}

template <class T> struct DevBuf {
  T *p = nullptr;
  size_t n = 0;
  DevBuf() = default;
  DevBuf(const DevBuf &) = delete;
  DevBuf &operator=(const DevBuf &) = delete;
  ~DevBuf() {
    if (p)
      cudaFree(p);
  }
  void alloc(size_t count, double &alloc_s) {
    const double t0 = wall_s();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&p),
                          std::max<size_t>(count, 1) * sizeof(T)));
    n = count;
    alloc_s += wall_s() - t0;
  }
  void upload(const std::vector<T> &h, double &h2d_s) {
    const double t0 = wall_s();
    CUDA_CHECK(
        cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    h2d_s += wall_s() - t0;
  }
  template <class H>
  void download(std::vector<H> &h, size_t count, double &d2h_s) const {
    download_ptr(p, h, count, d2h_s);
  }
};

struct DeviceGraph {
  vid_t n = 0;
  eid_t m = 0;
  DevBuf<eid_t> row_ptr;
  DevBuf<vid_t> col_idx;
  DevBuf<std::uint32_t> weights; // only if uploaded with weights
};

inline void upload_graph(const GraphCSR &g, DeviceGraph &d, Times &t,
                         bool with_weights) {
  d.n = g.n;
  d.m = g.m;
  d.row_ptr.alloc(g.row_ptr.size(), t.alloc_s);
  d.col_idx.alloc(g.col_idx.size(), t.alloc_s);
  d.row_ptr.upload(g.row_ptr, t.h2d_s);
  d.col_idx.upload(g.col_idx, t.h2d_s);
  if (with_weights) {
    d.weights.alloc(g.weights.size(), t.alloc_s);
    d.weights.upload(g.weights, t.h2d_s);
  }
}

constexpr int BLOCK = 256;
inline unsigned grid_for(size_t n) {
  return static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
}

// Vanilla CUDA baselines. Same algorithms and contracts as seq::. Results are
// copied back to the host (counted in Times::d2h_s). gT = in-edge graph (pass
// the same DeviceGraph if symmetric).
BFSResult bfs(const DeviceGraph &g, vid_t source, Times &t);
PageRankResult pagerank(const DeviceGraph &g, const DeviceGraph &gT,
                        const PageRankParams &p, Times &t);
CCResult connected_components(const DeviceGraph &g, Times &t);
SSSPResult sssp(const DeviceGraph &g, vid_t source, Times &t);
TCResult triangle_count(const DeviceGraph &g, Times &t);

} // namespace gpu