#include "cuda_common.hpp"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <utility>

namespace {
__global__ void pr_contrib(const eid_t *rp, const double *rank, double *contrib,
                           double *dang, vid_t n) {
  const vid_t v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n)
    return;
  const eid_t deg = rp[v + 1] - rp[v];
  if (deg == 0) {
    contrib[v] = 0.0;
    dang[v] = rank[v];
  } else {
    contrib[v] = rank[v] / static_cast<double>(deg);
    dang[v] = 0.0;
  }
}

// Pull: one thread per destination vertex over the in-edge CSR.
__global__ void pr_pull(const eid_t *rpT, const vid_t *ciT,
                        const double *contrib, const double *rank, double *next,
                        double *diff, vid_t n, double base, double d) {
  const vid_t v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n)
    return;
  double s = 0.0;
  for (eid_t e = rpT[v]; e < rpT[v + 1]; ++e)
    s += contrib[ciT[e]];
  const double nv = base + d * s;
  next[v] = nv;
  diff[v] = fabs(nv - rank[v]);
}

thrust::device_ptr<double> dp(double *p) {
  return thrust::device_pointer_cast(p);
}
} // namespace

namespace gpu {
PageRankResult pagerank(const DeviceGraph &g, const DeviceGraph &gT,
                        const PageRankParams &p, Times &t) {
  const vid_t n = g.n;
  const double inv = 1.0 / n, d = p.damping;
  DevBuf<double> rank_a, rank_b, contrib, dang, diff;
  rank_a.alloc(n, t.alloc_s);
  rank_b.alloc(n, t.alloc_s);
  contrib.alloc(n, t.alloc_s);
  dang.alloc(n, t.alloc_s);
  diff.alloc(n, t.alloc_s);

  PageRankResult res;
  GpuTimer timer;
  timer.start();
  thrust::fill(dp(rank_a.p), dp(rank_a.p) + n, inv);
  double *rk = rank_a.p, *nx = rank_b.p;
  for (int it = 0; it < p.max_iters; ++it) {
    pr_contrib<<<grid_for(n), BLOCK>>>(g.row_ptr.p, rk, contrib.p, dang.p, n);
    const double dangling = thrust::reduce(dp(dang.p), dp(dang.p) + n, 0.0);
    const double base = (1.0 - d) * inv + d * dangling * inv;
    pr_pull<<<grid_for(n), BLOCK>>>(gT.row_ptr.p, gT.col_idx.p, contrib.p, rk,
                                    nx, diff.p, n, base, d);
    const double dsum = thrust::reduce(dp(diff.p), dp(diff.p) + n, 0.0);
    std::swap(rk, nx);
    res.iters = it + 1;
    res.residual = dsum;
    if (dsum < p.tol)
      break;
  }
  CUDA_CHECK(cudaGetLastError());
  t.compute_s += timer.stop();
  download_ptr(rk, res.rank, n, t.d2h_s);
  return res;
}
} // namespace gpu