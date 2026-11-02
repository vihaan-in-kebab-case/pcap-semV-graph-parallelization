#include "benchmark_record.hpp"
#include "graph_csr.hpp"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace {
bool ends_with(const std::string &s, const std::string &suf) {
  return s.size() >= suf.size() &&
         s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

bool parse2(const std::string &s, std::uint64_t &a, std::uint64_t &b) {
  const char *p = s.c_str();
  char *e = nullptr;
  a = std::strtoull(p, &e, 10);
  if (e == p)
    return false;
  p = e;
  b = std::strtoull(p, &e, 10);
  return e != p;
}

std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
} // namespace

GraphCSR build_csr(vid_t n, std::vector<Edge> edges, bool symmetrize) {
  if (symmetrize) {
    const size_t k = edges.size();
    edges.reserve(2 * k);
    for (size_t i = 0; i < k; ++i)
      edges.emplace_back(edges[i].second, edges[i].first);
  }
  edges.erase(std::remove_if(edges.begin(), edges.end(),
                             [](const Edge &e) { return e.first == e.second; }),
              edges.end());
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

  GraphCSR g;
  g.n = n;
  g.m = edges.size();
  g.row_ptr.assign(static_cast<size_t>(n) + 1, 0);
  for (const Edge &e : edges)
    ++g.row_ptr[e.first + 1];
  std::partial_sum(g.row_ptr.begin(), g.row_ptr.end(), g.row_ptr.begin());
  g.col_idx.resize(g.m);
  for (size_t i = 0; i < edges.size(); ++i)
    g.col_idx[i] = edges[i].second;
  return g;
}

GraphCSR load_graph(const std::string &path, bool symmetrize, LoadTimings *t) {
  Timer tm;
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open " + path);

  std::vector<std::pair<std::uint64_t, std::uint64_t>> raw;
  std::string line;
  std::uint64_t n64 = 0;
  const bool mtx = ends_with(path, ".mtx");

  if (mtx) {
    std::getline(in, line);
    if (line.rfind("%%MatrixMarket", 0) != 0)
      throw std::runtime_error("bad MatrixMarket header");
    if (line.find("symmetric") != std::string::npos)
      symmetrize = true;
    while (std::getline(in, line) && (line.empty() || line[0] == '%')) {
    }
    std::uint64_t r = 0, c = 0, nz = 0;
    std::istringstream(line) >> r >> c >> nz;
    n64 = std::max(r, c);
    raw.reserve(nz);
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '%')
        continue;
      std::uint64_t a, b;
      if (parse2(line, a, b) && a > 0 && b > 0)
        raw.emplace_back(a - 1, b - 1);
    }
  } else {
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#' || line[0] == '%')
        continue;
      std::uint64_t a, b;
      if (parse2(line, a, b))
        raw.emplace_back(a, b);
    }
    std::vector<std::uint64_t> ids;
    ids.reserve(raw.size() * 2);
    for (auto &e : raw) {
      ids.push_back(e.first);
      ids.push_back(e.second);
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    n64 = ids.size();
    for (auto &e : raw) {
      e.first = std::lower_bound(ids.begin(), ids.end(), e.first) - ids.begin();
      e.second =
          std::lower_bound(ids.begin(), ids.end(), e.second) - ids.begin();
    }
  }
  if (n64 > 0xFFFFFFFEULL)
    throw std::runtime_error("graph too large for 32-bit vertex ids");
  if (t)
    t->read_s = tm.elapsed();

  tm.reset();
  std::vector<Edge> edges;
  edges.reserve(raw.size());
  for (auto &e : raw)
    edges.emplace_back(static_cast<vid_t>(e.first),
                       static_cast<vid_t>(e.second));
  raw.clear();
  raw.shrink_to_fit();
  GraphCSR g = build_csr(static_cast<vid_t>(n64), std::move(edges), symmetrize);
  if (t)
    t->build_s = tm.elapsed();
  return g;
}

GraphCSR symmetrized(const GraphCSR &g) {
  std::vector<Edge> edges;
  edges.reserve(g.m);
  for (vid_t u = 0; u < g.n; ++u)
    for (eid_t i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i)
      edges.emplace_back(u, g.col_idx[i]);
  return build_csr(g.n, std::move(edges), true);
}

bool is_symmetric(const GraphCSR &g) {
  for (vid_t u = 0; u < g.n; ++u)
    for (eid_t i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i) {
      vid_t v = g.col_idx[i];
      auto b = g.col_idx.begin() + g.row_ptr[v];
      auto e = g.col_idx.begin() + g.row_ptr[v + 1];
      if (!std::binary_search(b, e, u))
        return false;
    }
  return true;
}

GraphCSR transposed(const GraphCSR &g) {
  GraphCSR t;
  t.n = g.n;
  t.m = g.m;
  t.row_ptr.assign(static_cast<size_t>(g.n) + 1, 0);
  t.col_idx.resize(g.m);
  for (eid_t i = 0; i < g.m; ++i)
    ++t.row_ptr[g.col_idx[i] + 1]; // in-degree counts
  std::partial_sum(t.row_ptr.begin(), t.row_ptr.end(), t.row_ptr.begin());
  std::vector<eid_t> cursor(t.row_ptr.begin(), t.row_ptr.end() - 1);
  for (vid_t u = 0; u < g.n; ++u) // ascending u => sorted in-lists
    for (eid_t i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i)
      t.col_idx[cursor[g.col_idx[i]]++] = u;
  return t;
}

void attach_synthetic_weights(GraphCSR &g, std::uint64_t seed,
                              std::uint32_t max_w) {
  g.weights.resize(g.m);
  for (vid_t u = 0; u < g.n; ++u)
    for (eid_t i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i) {
      vid_t v = g.col_idx[i];
      std::uint64_t lo = std::min(u, v), hi = std::max(u, v);
      std::uint64_t h = splitmix64(seed ^ splitmix64((lo << 32) | hi));
      g.weights[i] = 1 + static_cast<std::uint32_t>(h % max_w);
    }
}