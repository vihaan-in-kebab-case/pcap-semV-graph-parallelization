#include "algorithm_api.hpp"

namespace omp {
// Level-synchronous BFS. Vertices are claimed with a compare-and-swap on
// level[]; each thread collects discoveries locally, merged once per level.
BFSResult bfs(const GraphCSR &g, vid_t src) {
  BFSResult r;
  r.level.assign(g.n, -1);
  std::int32_t *level = r.level.data();
  std::vector<vid_t> frontier{src}, next;
  level[src] = 0;
  for (std::int32_t depth = 0; !frontier.empty(); ++depth) {
    next.clear();
    const long fs = static_cast<long>(frontier.size());
#pragma omp parallel
    {
      std::vector<vid_t> local;
#pragma omp for schedule(dynamic, 64) nowait
      for (long i = 0; i < fs; ++i) {
        const vid_t u = frontier[i];
        for (eid_t e = g.row_ptr[u]; e < g.row_ptr[u + 1]; ++e) {
          const vid_t v = g.col_idx[e];
          std::int32_t expected = -1;
          if (__atomic_load_n(&level[v], __ATOMIC_RELAXED) < 0 &&
              __atomic_compare_exchange_n(&level[v], &expected, depth + 1,
                                          false, __ATOMIC_RELAXED,
                                          __ATOMIC_RELAXED))
            local.push_back(v);
        }
      }
#pragma omp critical
      next.insert(next.end(), local.begin(), local.end());
    }
    frontier.swap(next);
  }
  return r;
}
} // namespace omp