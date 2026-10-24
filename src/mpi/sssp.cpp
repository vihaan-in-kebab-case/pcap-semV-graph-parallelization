#include "mpi_dist.hpp"
#include <stdexcept>

namespace mpi {
// Frontier label-correcting; remote relaxations travel as (vertex, distance)
// uint64 pairs.
SSSPResult sssp(const DistGraph &g, vid_t src, Stats &st) {
  if (g.weights.size() != g.col_idx.size())
    throw std::runtime_error("sssp requires weights");
  const vid_t ln = g.local_n();
  SSSPResult r;
  r.dist.assign(ln, INF_DIST);
  std::vector<std::uint8_t> queued(ln, 0);
  std::vector<vid_t> frontier, next;
  if (src >= g.lo && src < g.hi) {
    r.dist[src - g.lo] = 0;
    frontier.push_back(src);
  }
  std::vector<std::vector<std::uint64_t>> out(g.size);

  auto relax = [&](vid_t v, std::uint64_t nd) {
    const vid_t i = v - g.lo;
    if (nd < r.dist[i]) {
      r.dist[i] = nd;
      if (!queued[i]) {
        queued[i] = 1;
        next.push_back(v);
      }
    }
  };

  for (;;) {
    next.clear();
    for (auto &o : out)
      o.clear();
    for (vid_t u : frontier) {
      const vid_t li = u - g.lo;
      const std::uint64_t du = r.dist[li];
      for (eid_t e = g.row_ptr[li]; e < g.row_ptr[li + 1]; ++e) {
        const vid_t v = g.col_idx[e];
        const std::uint64_t nd = du + g.weights[e];
        const int o = g.owner(v);
        if (o == g.rank)
          relax(v, nd);
        else {
          out[o].push_back(v);
          out[o].push_back(nd);
        }
      }
    }
    const std::vector<std::uint64_t> recv =
        alltoallv_vec(g, out, MPI_UINT64_T, st.comm_s);
    for (size_t k = 0; k + 1 < recv.size(); k += 2)
      relax(static_cast<vid_t>(recv[k]), recv[k + 1]);
    for (vid_t v : next)
      queued[v - g.lo] = 0;
    if (sum_u64(next.size(), st.comm_s) == 0)
      break;
    frontier.swap(next);
  }
  return r;
}
} // namespace mpi