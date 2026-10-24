#include "mpi_dist.hpp"

namespace mpi {
BFSResult bfs(const DistGraph &g, vid_t src, Stats &st) {
  BFSResult r;
  r.level.assign(g.local_n(), -1);
  std::vector<vid_t> frontier, next;
  if (src >= g.lo && src < g.hi) {
    r.level[src - g.lo] = 0;
    frontier.push_back(src);
  }
  std::vector<std::vector<vid_t>> out(g.size);

  for (std::int32_t depth = 0;; ++depth) {
    next.clear();
    for (auto &o : out)
      o.clear();
    for (vid_t u : frontier) {
      const vid_t li = u - g.lo;
      for (eid_t e = g.row_ptr[li]; e < g.row_ptr[li + 1]; ++e) {
        const vid_t v = g.col_idx[e];
        const int o = g.owner(v);
        if (o == g.rank) {
          if (r.level[v - g.lo] < 0) {
            r.level[v - g.lo] = depth + 1;
            next.push_back(v);
          }
        } else {
          out[o].push_back(v);
        }
      }
    }
    const std::vector<vid_t> recv =
        alltoallv_vec(g, out, MPI_UINT32_T, st.comm_s);
    for (vid_t v : recv)
      if (r.level[v - g.lo] < 0) {
        r.level[v - g.lo] = depth + 1;
        next.push_back(v);
      }
    if (sum_u64(next.size(), st.comm_s) == 0)
      break;
    frontier.swap(next);
  }
  return r;
}
} // namespace mpi