#include "cuda_common.hpp"

namespace {
__global__ void cc_init(vid_t *parent, vid_t n) {
  const vid_t v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v < n)
    parent[v] = v;
}

__device__ vid_t find_root(vid_t *parent, vid_t x) {
  for (;;) {
    const vid_t p = __ldcg(&parent[x]);
    if (p == x)
      return x;
    const vid_t gp = __ldcg(&parent[p]);
    if (gp != p)
      atomicCAS(&parent[x], p, gp); // path halving; losing the race is harmless
    x = gp;
  }
}

__device__ void unite(vid_t *parent, vid_t a, vid_t b) {
  for (;;) {
    a = find_root(parent, a);
    b = find_root(parent, b);
    if (a == b)
      return;
    if (a < b) {
      const vid_t tmp = a;
      a = b;
      b = tmp;
    } // link the larger root under the smaller
    if (atomicCAS(&parent[a], a, b) == a)
      return;
  }
}

__global__ void cc_hook(const eid_t *rp, const vid_t *ci, vid_t *parent,
                        vid_t n) {
  const vid_t u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= n)
    return;
  for (eid_t e = rp[u]; e < rp[u + 1]; ++e) {
    const vid_t v = ci[e];
    if (v < u)
      continue; // symmetric graph: visit each edge once
    unite(parent, u, v);
  }
}

__global__ void cc_flatten(vid_t *parent, vid_t n) {
  const vid_t v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v < n)
    parent[v] = find_root(parent, v);
}
} // namespace

namespace gpu {
CCResult connected_components(const DeviceGraph &g, Times &t) {
  DevBuf<vid_t> parent;
  parent.alloc(g.n, t.alloc_s);
  GpuTimer timer;
  timer.start();
  cc_init<<<grid_for(g.n), BLOCK>>>(parent.p, g.n);
  cc_hook<<<grid_for(g.n), BLOCK>>>(g.row_ptr.p, g.col_idx.p, parent.p, g.n);
  cc_flatten<<<grid_for(g.n), BLOCK>>>(parent.p, g.n);
  CUDA_CHECK(cudaGetLastError());
  t.compute_s += timer.stop();

  CCResult r;
  parent.download(r.label, g.n, t.d2h_s);
  for (vid_t v = 0; v < g.n; ++v)
    r.num_components += (r.label[v] == v); // host count, untimed
  return r;
}
} // namespace gpu