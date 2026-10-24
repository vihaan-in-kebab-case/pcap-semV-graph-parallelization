#include "algorithm_api.hpp"
#include <functional>
#include <queue>
#include <stdexcept>

namespace seq {
SSSPResult sssp(const GraphCSR &g, vid_t src) {
  if (!g.weighted())
    throw std::runtime_error("sssp requires weights");
  SSSPResult r;
  r.dist.assign(g.n, INF_DIST);
  using Item = std::pair<std::uint64_t, vid_t>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
  r.dist[src] = 0;
  pq.emplace(0, src);
  while (!pq.empty()) {
    auto [d, u] = pq.top();
    pq.pop();
    if (d != r.dist[u])
      continue; // stale entry
    for (eid_t i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i) {
      vid_t v = g.col_idx[i];
      std::uint64_t nd = d + g.weights[i];
      if (nd < r.dist[v]) {
        r.dist[v] = nd;
        pq.emplace(nd, v);
      }
    }
  }
  return r;
}
} // namespace seq