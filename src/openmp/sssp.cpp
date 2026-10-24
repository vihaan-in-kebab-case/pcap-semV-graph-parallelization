#include "algorithm_api.hpp"
#include <stdexcept>

namespace {
bool atomic_min(std::uint64_t *a, std::uint64_t val) {
  std::uint64_t cur = __atomic_load_n(a, __ATOMIC_RELAXED);
  while (val < cur)
    if (__atomic_compare_exchange_n(a, &cur, val, true, __ATOMIC_RELAXED,
                                    __ATOMIC_RELAXED))
      return true;
  return false;
}
} // namespace

namespace omp {
SSSPResult sssp(const GraphCSR &g, vid_t src) {
  if (!g.weighted())
    throw std::runtime_error("sssp requires weights");
  SSSPResult r;
  r.dist.assign(g.n, INF_DIST);
  std::uint64_t *dist = r.dist.data();
  std::vector<std::uint8_t> queued(g.n, 0); // "already in next frontier" flags
  std::vector<vid_t> frontier{src}, next;
  dist[src] = 0;

  while (!frontier.empty()) {
    next.clear();
    const long fs = static_cast<long>(frontier.size());
#pragma omp parallel
    {
      std::vector<vid_t> local;
#pragma omp for schedule(dynamic, 64) nowait
      for (long i = 0; i < fs; ++i) {
        const vid_t u = frontier[i];
        const std::uint64_t du = __atomic_load_n(&dist[u], __ATOMIC_RELAXED);
        for (eid_t e = g.row_ptr[u]; e < g.row_ptr[u + 1]; ++e) {
          const vid_t v = g.col_idx[e];
          if (atomic_min(&dist[v], du + g.weights[e]) &&
              !__atomic_exchange_n(&queued[v], std::uint8_t{1},
                                   __ATOMIC_RELAXED))
            local.push_back(v);
        }
      }
#pragma omp critical
      next.insert(next.end(), local.begin(), local.end());
    }
    const long ns = static_cast<long>(next.size());
#pragma omp parallel for schedule(static)
    for (long i = 0; i < ns; ++i)
      queued[next[i]] = 0; // reset after the round
    frontier.swap(next);
  }
  return r;
}
} // namespace omp