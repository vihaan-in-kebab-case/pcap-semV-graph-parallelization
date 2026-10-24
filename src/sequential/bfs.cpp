#include "algorithm_api.hpp"

namespace seq {
BFSResult bfs(const GraphCSR &g, vid_t src) {
  BFSResult r;
  r.level.assign(g.n, -1);
  std::vector<vid_t> q;
  q.reserve(g.n);
  r.level[src] = 0;
  q.push_back(src);
  for (size_t head = 0; head < q.size(); ++head) {
    vid_t u = q[head];
    for (eid_t i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i) {
      vid_t v = g.col_idx[i];
      if (r.level[v] < 0) {
        r.level[v] = r.level[u] + 1;
        q.push_back(v);
      }
    }
  }
  return r;
}
} // namespace seq