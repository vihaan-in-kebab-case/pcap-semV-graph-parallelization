#include "algorithm_api.hpp"
#include <utility>

namespace {
vid_t find_root(vid_t *parent, vid_t x) {
  for (;;) {
    vid_t p = __atomic_load_n(&parent[x], __ATOMIC_RELAXED);
    if (p == x)
      return x;
    const vid_t gp = __atomic_load_n(&parent[p], __ATOMIC_RELAXED);
    if (gp != p) // path halving; losing this race is harmless
      __atomic_compare_exchange_n(&parent[x], &p, gp, true, __ATOMIC_RELAXED,
                                  __ATOMIC_RELAXED);
    x = gp;
  }
}

void unite(vid_t *parent, vid_t a, vid_t b) {
  for (;;) {
    a = find_root(parent, a);
    b = find_root(parent, b);
    if (a == b)
      return;
    if (a < b)
      std::swap(a, b); // a is now the larger root: link it under b
    vid_t expected = a;
    if (__atomic_compare_exchange_n(&parent[a], &expected, b, false,
                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED))
      return;
    // a stopped being a root while we worked; retry from the new roots
  }
}
} // namespace

namespace omp {
CCResult connected_components(const GraphCSR &g) {
  const long N = g.n;
  std::vector<vid_t> parent(g.n);
#pragma omp parallel for schedule(static)
  for (long v = 0; v < N; ++v)
    parent[v] = static_cast<vid_t>(v);

  vid_t *par = parent.data();
#pragma omp parallel for schedule(dynamic, 1024)
  for (long ul = 0; ul < N; ++ul) {
    const vid_t u = static_cast<vid_t>(ul);
    for (eid_t i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i) {
      const vid_t v = g.col_idx[i];
      if (v < u)
        continue; // symmetric graph: visit each edge once
      unite(par, u, v);
    }
  }

  CCResult r;
  r.label.resize(g.n);
  long roots = 0;
#pragma omp parallel for schedule(static) reduction(+ : roots)
  for (long v = 0; v < N; ++v) { // all unions are done (implicit barrier above)
    r.label[v] = find_root(par, static_cast<vid_t>(v));
    roots += (r.label[v] == static_cast<vid_t>(v));
  }
  r.num_components = static_cast<vid_t>(roots);
  return r;
}
} // namespace omp