#include "algorithm_api.hpp"

namespace omp {
CCResult connected_components(const GraphCSR &g) {
  const long N = g.n;
  std::vector<vid_t> label(g.n), next(g.n);
#pragma omp parallel for schedule(static)
  for (long v = 0; v < N; ++v)
    label[v] = static_cast<vid_t>(v);

  int changed = 1;
  while (changed) {
    changed = 0;
#pragma omp parallel for schedule(dynamic, 1024) reduction(| : changed)
    for (long u = 0; u < N; ++u) {
      vid_t best = label[u];
      for (eid_t i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i) {
        const vid_t l = label[g.col_idx[i]];
        if (l < best)
          best = l;
      }
      next[u] = best;
      if (best != label[u])
        changed |= 1;
    }
    label.swap(next);
  }
  CCResult r;
  r.label = std::move(label);
  long roots = 0;
#pragma omp parallel for reduction(+ : roots)
  for (long v = 0; v < N; ++v)
    roots += (r.label[v] == static_cast<vid_t>(v));
  r.num_components = static_cast<vid_t>(roots);
  return r;
}
} // namespace omp