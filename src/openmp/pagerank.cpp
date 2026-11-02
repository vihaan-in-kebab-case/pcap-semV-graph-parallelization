#include "algorithm_api.hpp"
#include <cmath>

namespace omp {
// Pull-based, same formulation as seq::pagerank. Reductions may combine partial
// sums in a different order than sequential, so results match within
// floating-point tolerance.
PageRankResult pagerank(const GraphCSR &g, const GraphCSR &gT,
                        const PageRankParams &p) {
  PageRankResult res;
  const vid_t n = g.n;
  if (n == 0)
    return res;
  const long N = n;
  const double inv = 1.0 / n, d = p.damping;
  std::vector<double> rank(n, inv), next(n), contrib(n);

  for (int it = 0; it < p.max_iters; ++it) {
    double dangling = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : dangling)
    for (long u = 0; u < N; ++u) {
      const eid_t deg = g.out_degree(static_cast<vid_t>(u));
      if (deg == 0) {
        dangling += rank[u];
        contrib[u] = 0.0;
      } else
        contrib[u] = rank[u] / static_cast<double>(deg);
    }
    const double base = (1.0 - d) * inv + d * dangling * inv;
    double diff = 0.0;
#pragma omp parallel for schedule(dynamic, 1024) reduction(+ : diff)
    for (long v = 0; v < N; ++v) {
      double s = 0.0;
      for (eid_t i = gT.row_ptr[v]; i < gT.row_ptr[v + 1]; ++i)
        s += contrib[gT.col_idx[i]];
      next[v] = base + d * s;
      diff += std::fabs(next[v] - rank[v]);
    }
    rank.swap(next);
    res.iters = it + 1;
    res.residual = diff;
    if (diff < p.tol)
      break;
  }
  res.rank = std::move(rank);
  return res;
}
} // namespace omp