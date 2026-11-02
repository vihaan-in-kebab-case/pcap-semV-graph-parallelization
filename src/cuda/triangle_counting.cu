#include "cuda_common.hpp"

namespace {
// One thread per vertex u; each triangle u<v<w is counted once (same merge as
// seq/omp).
__global__ void tc_kernel(const eid_t *rp, const vid_t *ci, vid_t n,
                          unsigned long long *total) {
  const vid_t u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= n)
    return;
  unsigned long long cnt = 0;
  for (eid_t i = rp[u]; i < rp[u + 1]; ++i) {
    const vid_t v = ci[i];
    if (v <= u)
      continue;
    eid_t a = i + 1;
    const eid_t ae = rp[u + 1];
    eid_t lo = rp[v], hi = rp[v + 1];
    const eid_t be = hi;
    while (lo < hi) { // upper_bound(v) in adj(v)
      const eid_t mid = lo + (hi - lo) / 2;
      if (ci[mid] <= v)
        lo = mid + 1;
      else
        hi = mid;
    }
    eid_t b = lo;
    while (a < ae && b < be) {
      const vid_t x = ci[a], y = ci[b];
      if (x < y)
        ++a;
      else if (x > y)
        ++b;
      else {
        ++cnt;
        ++a;
        ++b;
      }
    }
  }
  if (cnt)
    atomicAdd(total, cnt);
}
} // namespace

namespace gpu {
TCResult triangle_count(const DeviceGraph &g, Times &t) {
  DevBuf<unsigned long long> total;
  total.alloc(1, t.alloc_s);
  GpuTimer timer;
  timer.start();
  CUDA_CHECK(cudaMemset(total.p, 0, sizeof(unsigned long long)));
  tc_kernel<<<grid_for(g.n), BLOCK>>>(g.row_ptr.p, g.col_idx.p, g.n, total.p);
  CUDA_CHECK(cudaGetLastError());
  t.compute_s += timer.stop();

  std::vector<unsigned long long> host;
  total.download(host, 1, t.d2h_s);
  TCResult r;
  r.triangles = host[0];
  return r;
}
} // namespace gpu