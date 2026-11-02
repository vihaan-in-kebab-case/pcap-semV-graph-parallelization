#include "mpi_dist.hpp"
#include <algorithm>

namespace mpi {
// Replicates the CSR on every rank (Allgatherv), then each rank counts
// triangles for its owned u. Same merge-intersection as seq/omp; replication
// cost is counted in comm_s.
TCResult triangle_count(const DistGraph &g, Stats &st) {
  const int P = g.size;
  const vid_t ln = g.local_n();
  std::uint64_t my_m = g.col_idx.size();
  std::vector<std::uint64_t> mcount(P);
  {
    CommTimer t(st.comm_s);
    MPI_Allgather(&my_m, 1, MPI_UINT64_T, mcount.data(), 1, MPI_UINT64_T,
                  MPI_COMM_WORLD);
  }
  std::vector<int> ecnt(P), edisp(P), vcnt(P), vdisp(P);
  std::uint64_t total = 0;
  for (int r = 0; r < P; ++r) {
    ecnt[r] = static_cast<int>(mcount[r]);
    edisp[r] = static_cast<int>(total);
    total += mcount[r];
    vcnt[r] = static_cast<int>(g.starts[r + 1] - g.starts[r]);
    vdisp[r] = static_cast<int>(g.starts[r]);
  }
  std::vector<eid_t> rp_local(ln);
  for (vid_t i = 0; i < ln; ++i)
    rp_local[i] = g.row_ptr[i] + static_cast<eid_t>(edisp[g.rank]);
  std::vector<eid_t> rp(static_cast<size_t>(g.n) + 1);
  std::vector<vid_t> col(total);
  {
    CommTimer t(st.comm_s);
    MPI_Allgatherv(rp_local.data(), static_cast<int>(ln), MPI_UINT64_T,
                   rp.data(), vcnt.data(), vdisp.data(), MPI_UINT64_T,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(g.col_idx.data(), static_cast<int>(my_m), MPI_UINT32_T,
                   col.data(), ecnt.data(), edisp.data(), MPI_UINT32_T,
                   MPI_COMM_WORLD);
  }
  rp[g.n] = total;

  std::uint64_t local = 0;
  for (vid_t u = g.lo; u < g.hi; ++u) {
    for (eid_t i = rp[u]; i < rp[u + 1]; ++i) {
      const vid_t v = col[i];
      if (v <= u)
        continue;
      eid_t a = i + 1, ae = rp[u + 1];
      eid_t b =
          std::upper_bound(col.begin() + rp[v], col.begin() + rp[v + 1], v) -
          col.begin();
      const eid_t be = rp[v + 1];
      while (a < ae && b < be) {
        if (col[a] < col[b])
          ++a;
        else if (col[a] > col[b])
          ++b;
        else {
          ++local;
          ++a;
          ++b;
        }
      }
    }
  }
  TCResult r;
  r.triangles = sum_u64(local, st.comm_s);
  return r;
}
} // namespace mpi