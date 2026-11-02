#include "mpi_dist.hpp"
#include <cmath>

namespace mpi {
// g supplies out-degrees of owned vertices; gT (same partition) supplies their
// in-edges.
PageRankResult pagerank(const DistGraph &g, const DistGraph &gT,
                        const PageRankParams &p, Stats &st) {
  PageRankResult res;
  const vid_t ln = g.local_n();
  const double inv = 1.0 / g.n, d = p.damping;
  std::vector<double> rank(ln, inv), next(ln), contrib(ln), contrib_all;

  for (int it = 0; it < p.max_iters; ++it) {
    double dangling_local = 0.0;
    for (vid_t i = 0; i < ln; ++i) {
      const eid_t deg = g.row_ptr[i + 1] - g.row_ptr[i];
      if (deg == 0) {
        dangling_local += rank[i];
        contrib[i] = 0.0;
      } else
        contrib[i] = rank[i] / static_cast<double>(deg);
    }
    const double dangling = sum_f64(dangling_local, st.comm_s);
    allgather_vec(g, contrib, contrib_all, MPI_DOUBLE, st.comm_s);

    const double base = (1.0 - d) * inv + d * dangling * inv;
    double diff_local = 0.0;
    for (vid_t i = 0; i < ln; ++i) {
      double s = 0.0;
      for (eid_t e = gT.row_ptr[i]; e < gT.row_ptr[i + 1]; ++e)
        s += contrib_all[gT.col_idx[e]];
      next[i] = base + d * s;
      diff_local += std::fabs(next[i] - rank[i]);
    }
    const double diff = sum_f64(diff_local, st.comm_s);
    rank.swap(next);
    res.iters = it + 1;
    res.residual = diff;
    if (diff < p.tol)
      break;
  }
  res.rank = std::move(rank);
  return res;
}
} // namespace mpi