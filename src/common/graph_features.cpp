#include "graph_features.hpp"
#include <algorithm>
#include <cmath>

// Only O(1) / O(n) work: row_ptr is enough, no pass over the edges.
GraphFeatures extract_features(const GraphCSR &g, bool declared_symmetric) {
  GraphFeatures f;
  f.n = g.n;
  f.m = g.m;
  f.symmetric = declared_symmetric;
  if (g.n == 0)
    return f;
  const double n = static_cast<double>(g.n);
  const double mean = static_cast<double>(g.m) / n;
  f.mean_degree = mean;
  double s2 = 0.0;
  eid_t dmax = 0, dangling = 0;
  for (vid_t v = 0; v < g.n; ++v) {
    const eid_t d = g.out_degree(v);
    dmax = std::max(dmax, d);
    if (d == 0)
      ++dangling;
    const double x = static_cast<double>(d) - mean;
    s2 += x * x;
  }
  f.degree_skew = mean > 0 ? static_cast<double>(dmax) / mean : 0.0;
  f.out_cv = mean > 0 ? std::sqrt(s2 / n) / mean : 0.0;
  f.dangling_frac = static_cast<double>(dangling) / n;
  return f;
}