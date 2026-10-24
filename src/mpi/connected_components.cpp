#include "mpi_dist.hpp"

namespace mpi {
// Distributed min-label propagation (Jacobi): every round allgathers the
// labels.
CCResult connected_components(const DistGraph &g, Stats &st) {
  const vid_t ln = g.local_n();
  std::vector<vid_t> label(ln), next(ln), all;
  for (vid_t i = 0; i < ln; ++i)
    label[i] = g.lo + i;

  for (;;) {
    allgather_vec(g, label, all, MPI_UINT32_T, st.comm_s);
    std::uint64_t changed = 0;
    for (vid_t i = 0; i < ln; ++i) {
      vid_t best = label[i];
      for (eid_t e = g.row_ptr[i]; e < g.row_ptr[i + 1]; ++e)
        best = std::min(best, all[g.col_idx[e]]);
      next[i] = best;
      if (best != label[i])
        changed = 1;
    }
    label.swap(next);
    if (sum_u64(changed, st.comm_s) == 0)
      break;
  }
  std::uint64_t roots = 0;
  for (vid_t i = 0; i < ln; ++i)
    roots += (label[i] == g.lo + i);
  CCResult r;
  r.label = std::move(label);
  r.num_components = static_cast<vid_t>(sum_u64(roots, st.comm_s));
  return r;
}
} // namespace mpi