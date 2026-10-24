#include "algorithm_api.hpp"
#include "correctness.hpp"
#include <cmath>
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

static std::vector<std::uint64_t> bellman_ford(const GraphCSR &g, vid_t src) {
  std::vector<std::uint64_t> d(g.n, INF_DIST);
  d[src] = 0;
  for (vid_t it = 0; it < g.n; ++it) {
    bool changed = false;
    for (vid_t u = 0; u < g.n; ++u) {
      if (d[u] == INF_DIST)
        continue;
      for (eid_t i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i)
        if (d[u] + g.weights[i] < d[g.col_idx[i]]) {
          d[g.col_idx[i]] = d[u] + g.weights[i];
          changed = true;
        }
    }
    if (!changed)
      break;
  }
  return d;
}

static std::uint64_t brute_triangles(const GraphCSR &g) {
  std::vector<std::vector<char>> adj(g.n, std::vector<char>(g.n, 0));
  for (vid_t u = 0; u < g.n; ++u)
    for (eid_t i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i)
      adj[u][g.col_idx[i]] = 1;
  std::uint64_t c = 0;
  for (vid_t a = 0; a < g.n; ++a)
    for (vid_t b = a + 1; b < g.n; ++b)
      if (adj[a][b])
        for (vid_t d = b + 1; d < g.n; ++d)
          if (adj[a][d] && adj[b][d])
            ++c;
  return c;
}

static std::vector<double> ref_pagerank(const GraphCSR &g, double d,
                                        int iters) {
  const vid_t n = g.n;
  std::vector<std::vector<vid_t>> in(n);
  for (vid_t u = 0; u < n; ++u)
    for (eid_t i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i)
      in[g.col_idx[i]].push_back(u);
  std::vector<double> r(n, 1.0 / n), nx(n);
  for (int it = 0; it < iters; ++it) {
    double dangling = 0;
    for (vid_t u = 0; u < n; ++u)
      if (g.out_degree(u) == 0)
        dangling += r[u];
    for (vid_t v = 0; v < n; ++v) {
      double s = 0;
      for (vid_t u : in[v])
        s += r[u] / g.out_degree(u);
      nx[v] = (1 - d) / n + d * (dangling / n + s);
    }
    r.swap(nx);
  }
  return r;
}

int main() {
  // Hand-verified tiny graph.
  {
    std::vector<Edge> e = {{0, 1}, {0, 2}, {1, 2}, {2, 3}, {3, 4}, {5, 6}};
    GraphCSR g = build_csr(7, e, false);
    auto b = seq::bfs(g, 0);
    CHECK((b.level == std::vector<std::int32_t>{0, 1, 1, 2, 3, -1, -1}),
          "tiny bfs levels");
    GraphCSR gu = build_csr(7, e, true);
    auto cc = seq::connected_components(gu);
    CHECK(cc.num_components == 2, "tiny cc count");
    CHECK((cc.label == std::vector<vid_t>{0, 0, 0, 0, 0, 5, 5}),
          "tiny cc labels");
    CHECK(seq::triangle_count(gu).triangles == 1, "tiny triangles");
  }
  // K4 has 4 triangles, 1 component.
  {
    std::vector<Edge> e;
    for (vid_t a = 0; a < 4; ++a)
      for (vid_t b = a + 1; b < 4; ++b)
        e.emplace_back(a, b);
    GraphCSR g = build_csr(4, e, true);
    CHECK(seq::triangle_count(g).triangles == 4, "K4 triangles");
    CHECK(seq::connected_components(g).num_components == 1, "K4 components");
  }
  // Random graphs vs independent references.
  for (std::uint64_t s = 1; s <= 20; ++s) {
    std::mt19937_64 rng(s);
    vid_t n = 30 + static_cast<vid_t>(rng() % 30);
    std::vector<Edge> e;
    for (vid_t k = 0; k < n * 3; ++k)
      e.emplace_back(rng() % n, rng() % n);
    GraphCSR gd = build_csr(n, e, false), gu = build_csr(n, e, true);
    attach_synthetic_weights(gd, s);
    attach_synthetic_weights(gu, s);
    vid_t src = static_cast<vid_t>(rng() % n);
    std::string tag = " (seed " + std::to_string(s) + ")";

    CHECK(check_bfs(gd, src, seq::bfs(gd, src).level).empty(),
          "bfs validator" + tag);
    auto ss = seq::sssp(gd, src);
    CHECK(ss.dist == bellman_ford(gd, src), "sssp vs bellman-ford" + tag);
    CHECK(check_sssp(gd, src, ss.dist).empty(), "sssp validator" + tag);
    CHECK(check_cc(gu, seq::connected_components(gu).label).empty(),
          "cc validator" + tag);
    auto tc = seq::triangle_count(gu);
    CHECK(tc.triangles == brute_triangles(gu), "tc vs brute force" + tag);
    CHECK(check_tc(gu, tc.triangles).empty(), "tc validator" + tag);

    PageRankParams p;
    p.tol = 1e-13;
    p.max_iters = 500;
    GraphCSR gdT = transposed(gd);
    CHECK(transposed(gdT).row_ptr == gd.row_ptr &&
              transposed(gdT).col_idx == gd.col_idx,
          "transpose round-trip" + tag);
    auto pr = seq::pagerank(gd, gdT, p);
    CHECK(pagerank_close(pr.rank, ref_pagerank(gd, 0.85, 500), 1e-9),
          "pagerank vs reference" + tag);
  }
  if (failures == 0)
    std::cout << "all sequential tests passed\n";
  return failures == 0 ? 0 : 1;
}