#include "algorithm_api.hpp"
#include <cmath>

namespace seq {
PageRankResult pagerank(const GraphCSR &g, const GraphCSR &gT,
                        const PageRankParams &p) {
  PageRankResult res;
  const vid_t n = g.n;
  if (n == 0)
    return res;
  const double inv = 1.0 / n, d = p.damping;
  std::vector<double> rank(n, inv), next(n), contrib(n);

  for (int it = 0; it < p.max_iters; ++it) {
    double dangling = 0.0;
    for (vid_t u = 0; u < n; ++u) {
      eid_t deg = g.out_degree(u);
      if (deg == 0) {
        dangling += rank[u];
        contrib[u] = 0.0;
      } else
        contrib[u] = rank[u] / static_cast<double>(deg);
    }
    const double base = (1.0 - d) * inv + d * dangling * inv;
    double diff = 0.0;
    for (vid_t v = 0; v < n; ++v) {
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
} // namespace seq