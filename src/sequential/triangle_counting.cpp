#include "algorithm_api.hpp"
#include <algorithm>

namespace seq {
TCResult triangle_count(const GraphCSR &g) {
  TCResult r;
  const vid_t *col = g.col_idx.data();
  for (vid_t u = 0; u < g.n; ++u) {
    for (eid_t i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i) {
      vid_t v = col[i];
      if (v <= u)
        continue;
      eid_t a = i + 1, ae = g.row_ptr[u + 1]; // neighbours of u greater than v
      eid_t b =
          std::upper_bound(col + g.row_ptr[v], col + g.row_ptr[v + 1], v) - col;
      eid_t be = g.row_ptr[v + 1];
      while (a < ae && b < be) {
        if (col[a] < col[b])
          ++a;
        else if (col[a] > col[b])
          ++b;
        else {
          ++r.triangles;
          ++a;
          ++b;
        }
      }
    }
  }
  return r;
}
} // namespace seq