#include "algorithm_api.hpp"
#include <algorithm>

namespace omp {
TCResult triangle_count(const GraphCSR &g) {
  std::uint64_t total = 0;
  const vid_t *col = g.col_idx.data();
  const long N = g.n;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : total)
  for (long ul = 0; ul < N; ++ul) {
    const vid_t u = static_cast<vid_t>(ul);
    for (eid_t i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i) {
      const vid_t v = col[i];
      if (v <= u)
        continue;
      eid_t a = i + 1, ae = g.row_ptr[u + 1];
      eid_t b =
          std::upper_bound(col + g.row_ptr[v], col + g.row_ptr[v + 1], v) - col;
      const eid_t be = g.row_ptr[v + 1];
      while (a < ae && b < be) {
        if (col[a] < col[b])
          ++a;
        else if (col[a] > col[b])
          ++b;
        else {
          ++total;
          ++a;
          ++b;
        }
      }
    }
  }
  TCResult r;
  r.triangles = total;
  return r;
}
} // namespace omp