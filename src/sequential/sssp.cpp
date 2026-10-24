#include "algorithm_api.hpp"
#include <stdexcept>

namespace seq {
// Frontier-based label-correcting SSSP: the single-threaded form of omp::sssp.
SSSPResult sssp(const GraphCSR &g, vid_t src) {
  if (!g.weighted())
    throw std::runtime_error("sssp requires weights");
  SSSPResult r;
  r.dist.assign(g.n, INF_DIST);
  std::vector<std::uint8_t> queued(g.n, 0); // "already in next frontier" flags
  std::vector<vid_t> frontier{src}, next;
  r.dist[src] = 0;
  while (!frontier.empty()) {
    next.clear();
    for (vid_t u : frontier) {
      const std::uint64_t du = r.dist[u];
      for (eid_t e = g.row_ptr[u]; e < g.row_ptr[u + 1]; ++e) {
        const vid_t v = g.col_idx[e];
        const std::uint64_t nd = du + g.weights[e];
        if (nd < r.dist[v]) {
          r.dist[v] = nd;
          if (!queued[v]) {
            queued[v] = 1;
            next.push_back(v);
          }
        }
      }
    }
    for (vid_t v : next)
      queued[v] = 0; // reset after the round
    frontier.swap(next);
  }
  return r;
}
} // namespace seq