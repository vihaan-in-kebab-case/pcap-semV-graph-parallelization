#include "cuda_common.hpp"
#include <utility>

namespace {
__global__ void init_sssp(unsigned long long *dist, vid_t *frontier,
                          vid_t src) {
  dist[src] = 0ULL;
  frontier[0] = src;
}

__global__ void sssp_step(const eid_t *rp, const vid_t *ci,
                          const std::uint32_t *w, const vid_t *frontier,
                          unsigned fsize, unsigned long long *dist,
                          unsigned *queued, vid_t *next, unsigned *next_size) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= fsize)
    return;
  const vid_t u = frontier[i];
  const unsigned long long du = *((volatile unsigned long long *)&dist[u]);
  for (eid_t e = rp[u]; e < rp[u + 1]; ++e) {
    const vid_t v = ci[e];
    const unsigned long long nd = du + w[e];
    if (atomicMin(&dist[v], nd) > nd && atomicExch(&queued[v], 1u) == 0u)
      next[atomicAdd(next_size, 1u)] = v;
  }
}

__global__ void clear_queued(unsigned *queued, const vid_t *next,
                             unsigned nsize) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nsize)
    queued[next[i]] = 0u;
}
} // namespace

namespace gpu {
SSSPResult sssp(const DeviceGraph &g, vid_t src, Times &t) {
  DevBuf<unsigned long long> dist;
  DevBuf<unsigned> queued, counter;
  DevBuf<vid_t> fa, fb;
  dist.alloc(g.n, t.alloc_s);
  queued.alloc(g.n, t.alloc_s);
  counter.alloc(1, t.alloc_s);
  fa.alloc(g.n, t.alloc_s);
  fb.alloc(g.n, t.alloc_s);

  GpuTimer timer;
  timer.start();
  CUDA_CHECK(
      cudaMemset(dist.p, 0xFF,
                 sizeof(unsigned long long) * g.n)); // all bits set = INF_DIST
  CUDA_CHECK(cudaMemset(queued.p, 0, sizeof(unsigned) * g.n));
  init_sssp<<<1, 1>>>(dist.p, fa.p, src);
  CUDA_CHECK(cudaGetLastError());
  vid_t *cur = fa.p, *nxt = fb.p;
  unsigned fsize = 1;
  while (fsize > 0) {
    CUDA_CHECK(cudaMemset(counter.p, 0, sizeof(unsigned)));
    sssp_step<<<grid_for(fsize), BLOCK>>>(g.row_ptr.p, g.col_idx.p, g.weights.p,
                                          cur, fsize, dist.p, queued.p, nxt,
                                          counter.p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&fsize, counter.p, sizeof(unsigned),
                          cudaMemcpyDeviceToHost));
    if (fsize > 0)
      clear_queued<<<grid_for(fsize), BLOCK>>>(queued.p, nxt, fsize);
    std::swap(cur, nxt);
  }
  t.compute_s += timer.stop();

  SSSPResult r;
  dist.download(r.dist, g.n, t.d2h_s);
  return r;
}
} // namespace gpu