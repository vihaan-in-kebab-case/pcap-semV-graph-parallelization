#include "algorithm_api.hpp"
#include "correctness.hpp"
#include <iostream>
#include <omp.h>
#include <random>

static int failures = 0;
#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      ++failures;                                                              \
      std::cerr << "FAIL: " << msg << "\n";                                    \
    }                                                                          \
  } while (0)

int main() {
  // Edge cases: single isolated vertex.
  {
    GraphCSR g = build_csr(1, {}, false);
    attach_synthetic_weights(g, 1);
    CHECK(omp::bfs(g, 0).level == std::vector<std::int32_t>{0}, "n=1 bfs");
    CHECK(omp::connected_components(g).num_components == 1, "n=1 cc");
    CHECK(omp::triangle_count(g).triangles == 0, "n=1 tc");
    CHECK(omp::sssp(g, 0).dist == std::vector<std::uint64_t>{0}, "n=1 sssp");
    PageRankParams p;
    auto pr = omp::pagerank(g, g, p);
    CHECK(std::fabs(pr.rank[0] - 1.0) < 1e-12, "n=1 pagerank");
  }

  for (int threads : {1, 2, 3, 4, 8}) {
    omp_set_num_threads(threads);
    for (std::uint64_t s = 1; s <= 15; ++s) {
      std::mt19937_64 rng(1000 + s);
      const vid_t n = 40 + static_cast<vid_t>(rng() % 400);
      std::vector<Edge> e;
      for (vid_t k = 0; k < n * 4; ++k)
        e.emplace_back(rng() % n, rng() % n);
      for (vid_t k = 0; k < n / 2; ++k)
        e.emplace_back(0, rng() % n); // hub => degree skew
      GraphCSR gd = build_csr(n, e, false), gu = build_csr(n, e, true);
      attach_synthetic_weights(gd, s);
      attach_synthetic_weights(gu, s);
      GraphCSR gdT = transposed(gd);
      const vid_t src = static_cast<vid_t>(rng() % n);
      const std::string tag = " (threads " + std::to_string(threads) +
                              ", seed " + std::to_string(s) + ")";

      auto b = omp::bfs(gd, src);
      CHECK(b.level == seq::bfs(gd, src).level, "bfs equals sequential" + tag);
      CHECK(check_bfs(gd, src, b.level).empty(), "bfs validator" + tag);

      auto ss = omp::sssp(gd, src);
      CHECK(ss.dist == seq::sssp(gd, src).dist, "sssp equals sequential" + tag);
      CHECK(check_sssp(gd, src, ss.dist).empty(), "sssp validator" + tag);

      auto cc = omp::connected_components(gu);
      CHECK(check_cc(gu, cc.label).empty(), "cc partition" + tag);
      CHECK(cc.num_components == seq::connected_components(gu).num_components,
            "cc count" + tag);

      auto tc = omp::triangle_count(gu);
      CHECK(tc.triangles == seq::triangle_count(gu).triangles,
            "tc equals sequential" + tag);
      CHECK(check_tc(gu, tc.triangles).empty(), "tc validator" + tag);

      PageRankParams p;
      auto pr = omp::pagerank(gd, gdT, p);
      auto ref = seq::pagerank(gd, gdT, p);
      CHECK(check_pagerank(pr, p).empty(), "pagerank sane" + tag);
      CHECK(pagerank_l1_close(pr.rank, ref.rank, 10 * p.tol),
            "pagerank vs sequential" + tag);
    }
  }
  if (failures == 0)
    std::cout << "all openmp tests passed\n";
  return failures == 0 ? 0 : 1;
}