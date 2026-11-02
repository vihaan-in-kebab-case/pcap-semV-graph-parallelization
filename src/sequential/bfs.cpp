#include "algorithm_api.hpp"

namespace seq {
// Level-synchronous BFS: the single-threaded form of omp::bfs.
BFSResult bfs(const GraphCSR &g, vid_t src) {
  BFSResult r;
  r.level.assign(g.n, -1);
  std::vector<vid_t> frontier{src}, next;
  r.level[src] = 0;
  for (std::int32_t depth = 0; !frontier.empty(); ++depth) {
    next.clear();
    for (vid_t u : frontier)
      for (eid_t e = g.row_ptr[u]; e < g.row_ptr[u + 1]; ++e) {
        const vid_t v = g.col_idx[e];
        if (r.level[v] < 0) {
          r.level[v] = depth + 1;
          next.push_back(v);
        }
      }
    frontier.swap(next);
  }
  return r;
}
} // namespace seq