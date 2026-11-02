#pragma once
#include "algorithm_api.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

inline std::string check_bfs(const GraphCSR &g, vid_t src,
                             const std::vector<std::int32_t> &lv) {
  if (lv.size() != g.n)
    return "bfs: wrong result size";
  if (lv[src] != 0)
    return "bfs: source level != 0";
  std::vector<char> has_parent(g.n, 0);
  for (vid_t u = 0; u < g.n; ++u) {
    if (lv[u] < 0)
      continue;
    for (eid_t i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i) {
      vid_t v = g.col_idx[i];
      if (lv[v] < 0 || lv[v] > lv[u] + 1)
        return "bfs: edge " + std::to_string(u) + "->" + std::to_string(v) +
               " violates level bound";
      if (lv[v] == lv[u] + 1)
        has_parent[v] = 1;
    }
  }
  for (vid_t v = 0; v < g.n; ++v) {
    if (v == src)
      continue;
    if (lv[v] == 0)
      return "bfs: non-source vertex at level 0";
    if (lv[v] > 0 && !has_parent[v])
      return "bfs: vertex " + std::to_string(v) + " has no parent";
  }
  return "";
}

inline std::string check_sssp(const GraphCSR &g, vid_t src,
                              const std::vector<std::uint64_t> &d) {
  if (d.size() != g.n)
    return "sssp: wrong result size";
  if (d[src] != 0)
    return "sssp: source distance != 0";
  std::vector<char> tight(g.n, 0);
  for (vid_t u = 0; u < g.n; ++u) {
    if (d[u] == INF_DIST)
      continue;
    for (eid_t i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i) {
      vid_t v = g.col_idx[i];
      std::uint64_t nd = d[u] + g.weights[i];
      if (d[v] > nd)
        return "sssp: edge " + std::to_string(u) + "->" + std::to_string(v) +
               " not relaxed";
      if (d[v] == nd)
        tight[v] = 1;
    }
  }
  for (vid_t v = 0; v < g.n; ++v)
    if (v != src && d[v] != INF_DIST && !tight[v])
      return "sssp: vertex " + std::to_string(v) + " has no tight predecessor";
  return "";
}

inline std::string check_pagerank(const PageRankResult &r,
                                  const PageRankParams &p) {
  double sum = 0.0;
  for (double x : r.rank) {
    if (!(x >= 0.0))
      return "pagerank: negative or NaN rank";
    sum += x;
  }
  if (std::fabs(sum - 1.0) > 1e-6)
    return "pagerank: ranks do not sum to 1";
  if (r.iters >= p.max_iters && r.residual >= p.tol)
    return "pagerank: did not converge";
  return "";
}

inline std::string check_cc(const GraphCSR &g,
                            const std::vector<vid_t> &label) {
  if (label.size() != g.n)
    return "cc: wrong result size";
  const vid_t NONE = static_cast<vid_t>(-1);
  std::vector<vid_t> ref(g.n, NONE), stack;
  for (vid_t s = 0; s < g.n; ++s) { // independent flood fill
    if (ref[s] != NONE)
      continue;
    ref[s] = s;
    stack.push_back(s);
    while (!stack.empty()) {
      vid_t u = stack.back();
      stack.pop_back();
      for (eid_t i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i) {
        vid_t v = g.col_idx[i];
        if (ref[v] == NONE) {
          ref[v] = s;
          stack.push_back(v);
        }
      }
    }
  }
  std::vector<vid_t> fwd(g.n, NONE),
      bwd(g.n, NONE); // reference <-> reported label bijection
  for (vid_t v = 0; v < g.n; ++v) {
    if (label[v] >= g.n)
      return "cc: label out of range";
    const vid_t r = ref[v], l = label[v];
    if (fwd[r] == NONE)
      fwd[r] = l;
    else if (fwd[r] != l)
      return "cc: component split";
    if (bwd[l] == NONE)
      bwd[l] = r;
    else if (bwd[l] != r)
      return "cc: components merged";
  }
  return "";
}

inline std::string check_tc(const GraphCSR &g, std::uint64_t count) {
  std::uint64_t total = 0;
  for (vid_t u = 0; u < g.n; ++u)
    for (eid_t i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i) {
      vid_t v = g.col_idx[i];
      if (v <= u)
        continue;
      eid_t a = g.row_ptr[u], ae = g.row_ptr[u + 1];
      eid_t b = g.row_ptr[v], be = g.row_ptr[v + 1];
      while (a < ae && b < be) {
        if (g.col_idx[a] < g.col_idx[b])
          ++a;
        else if (g.col_idx[a] > g.col_idx[b])
          ++b;
        else {
          ++total;
          ++a;
          ++b;
        }
      }
    }
  if (total % 3 != 0 || total / 3 != count)
    return "tc: expected " + std::to_string(total / 3) + " got " +
           std::to_string(count);
  return "";
}

inline bool pagerank_close(const std::vector<double> &a,
                           const std::vector<double> &b, double tol) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::fabs(a[i] - b[i]) > tol)
      return false;
  return true;
}

// L1 distance between two rank vectors (ranks sum to 1, so this is
// scale-aware).
inline bool pagerank_l1_close(const std::vector<double> &a,
                              const std::vector<double> &b, double tol) {
  if (a.size() != b.size())
    return false;
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    s += std::fabs(a[i] - b[i]);
  return s <= tol;
}