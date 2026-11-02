#include "algorithm_api.hpp"
#include <numeric>

namespace {
vid_t find(std::vector<vid_t> &p, vid_t x) {
  while (p[x] != x) {
    p[x] = p[p[x]];
    x = p[x];
  }
  return x;
}
} // namespace

namespace seq {
CCResult connected_components(const GraphCSR &g) {
  std::vector<vid_t> parent(g.n);
  std::iota(parent.begin(), parent.end(), 0);
  for (vid_t u = 0; u < g.n; ++u)
    for (eid_t i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i) {
      if (g.col_idx[i] < u)
        continue; // symmetric graph: visit each edge once
      vid_t a = find(parent, u), b = find(parent, g.col_idx[i]);
      if (a == b)
        continue;
      if (a < b)
        parent[b] = a;
      else
        parent[a] = b;
    }
  CCResult r;
  r.label.resize(g.n);
  for (vid_t v = 0; v < g.n; ++v) {
    r.label[v] = find(parent, v);
    if (r.label[v] == v)
      ++r.num_components;
  }
  return r;
}
} // namespace seq