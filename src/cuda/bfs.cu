#include "cuda_common.hpp"
#include <utility>

namespace {
__global__ void init_bfs(int *level, vid_t *frontier, vid_t src) {
  level[src] = 0;
  frontier[0] = src;
}

// One thread per frontier vertex; vertices are claimed with atomicCAS,
// discoveries appended with atomicAdd.
__global__ void bfs_step(const eid_t *rp, const vid_t *ci,
                         const vid_t *frontier, unsigned fsize, int *level,
                         int depth, vid_t *next, unsigned *next_size) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= fsize)
    return;
  const vid_t u = frontier[i];
  for (eid_t e = rp[u]; e < rp[u + 1]; ++e) {
    const vid_t v = ci[e];
    if (*((volatile int *)&level[v]) < 0 &&
        atomicCAS(&level[v], -1, depth + 1) == -1)
      next[atomicAdd(next_size, 1u)] = v;
  }
}
} // namespace

namespace gpu {
BFSResult bfs(const DeviceGraph &g, vid_t src, Times &t) {
  DevBuf<int> level, counter_dummy;
  DevBuf<vid_t> fa, fb;
  DevBuf<unsigned> counter;
  level.alloc(g.n, t.alloc_s);
  fa.alloc(g.n, t.alloc_s);
  fb.alloc(g.n, t.alloc_s);
  counter.alloc(1, t.alloc_s);

  GpuTimer timer;
  timer.start();
  CUDA_CHECK(cudaMemset(level.p, 0xFF, sizeof(int) * g.n)); // all -1
  init_bfs<<<1, 1>>>(level.p, fa.p, src);
  CUDA_CHECK(cudaGetLastError());
  vid_t *cur = fa.p, *nxt = fb.p;
  unsigned fsize = 1;
  for (int depth = 0; fsize > 0; ++depth) {
    CUDA_CHECK(cudaMemset(counter.p, 0, sizeof(unsigned)));
    bfs_step<<<grid_for(fsize), BLOCK>>>(g.row_ptr.p, g.col_idx.p, cur, fsize,
                                         level.p, depth, nxt, counter.p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&fsize, counter.p, sizeof(unsigned),
                          cudaMemcpyDeviceToHost));
    std::swap(cur, nxt);
  }
  t.compute_s += timer.stop();

  BFSResult r;
  level.download(r.level, g.n, t.d2h_s);
  return r;
}
} // namespace gpu