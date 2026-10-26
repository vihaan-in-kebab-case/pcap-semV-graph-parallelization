#include "algorithm_api.hpp"
#include "correctness.hpp"
#include "cuda_common.hpp"
#include <iostream>
#include <random>

static int failures = 0;
#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      ++failures;                                                              \
      std::cerr << "FAIL: " << msg << "\n";                                    \
    }                                                                          \
  } while (0)

static void run_case(vid_t n, const std::vector<Edge> &e, vid_t src,
                     std::uint64_t seed, const std::string &tag) {
  GraphCSR gd = build_csr(n, e, false), gu = build_csr(n, e, true);
  attach_synthetic_weights(gd, seed);
  GraphCSR gdT = transposed(gd);
  gpu::Times t;
  gpu::DeviceGraph dd, ddT, du, dw;
  gpu::upload_graph(gd, dd, t, false);
  gpu::upload_graph(gdT, ddT, t, false);
  gpu::upload_graph(gu, du, t, false);
  gpu::upload_graph(gd, dw, t, true);

  auto b = gpu::bfs(dd, src, t);
  CHECK(b.level == seq::bfs(gd, src).level &&
            check_bfs(gd, src, b.level).empty(),
        "bfs " + tag);

  PageRankParams p;
  auto pr = gpu::pagerank(dd, ddT, p, t);
  CHECK(check_pagerank(pr, p).empty() &&
            pagerank_l1_close(pr.rank, seq::pagerank(gd, gdT, p).rank,
                              10 * p.tol),
        "pagerank " + tag);

  auto cc = gpu::connected_components(du, t);
  CHECK(check_cc(gu, cc.label).empty() &&
            cc.num_components == seq::connected_components(gu).num_components,
        "cc " + tag);

  auto ss = gpu::sssp(dw, src, t);
  CHECK(ss.dist == seq::sssp(gd, src).dist &&
            check_sssp(gd, src, ss.dist).empty(),
        "sssp " + tag);

  CHECK(gpu::triangle_count(du, t).triangles ==
            seq::triangle_count(gu).triangles,
        "tc " + tag);
}

int main() {
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
    std::cerr << "no CUDA device found\n";
    return 2;
  }
  run_case(1, {}, 0, 1, "n=1");
  run_case(7, {{0, 1}, {0, 2}, {1, 2}, {2, 3}, {3, 4}, {5, 6}}, 0, 2, "tiny");
  for (std::uint64_t s = 1; s <= 15; ++s) {
    std::mt19937_64 rng(3000 + s);
    const vid_t n = 40 + static_cast<vid_t>(rng() % 400);
    std::vector<Edge> e;
    for (vid_t k = 0; k < n * 4; ++k)
      e.emplace_back(rng() % n, rng() % n);
    for (vid_t k = 0; k < n / 2; ++k)
      e.emplace_back(0, rng() % n); // hub => degree skew
    run_case(n, e, static_cast<vid_t>(rng() % n), s,
             "random seed " + std::to_string(s));
  }
  if (failures == 0)
    std::cout << "all cuda tests passed\n";
  return failures == 0 ? 0 : 1;
}