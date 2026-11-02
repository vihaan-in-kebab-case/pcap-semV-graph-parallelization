#include "mpi_dist.hpp"
#include <climits>
#include <cstdio>

DistGraph distribute(const GraphCSR *full, bool with_weights) {
  DistGraph d;
  MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &d.size);
  const bool root = (d.rank == 0);

  std::uint64_t n64 = root ? full->n : 0;
  MPI_Bcast(&n64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  d.n = static_cast<vid_t>(n64);
  d.starts.resize(d.size + 1);
  for (int r = 0; r <= d.size; ++r)
    d.starts[r] =
        static_cast<vid_t>(static_cast<std::uint64_t>(d.n) * r / d.size);
  d.lo = d.starts[d.rank];
  d.hi = d.starts[d.rank + 1];

  std::vector<int> rc(d.size, 0), rd(d.size, 0), ec(d.size, 0), ed(d.size, 0);
  if (root) {
    if (full->m > static_cast<eid_t>(INT_MAX) ||
        (with_weights && full->weights.size() != full->m)) {
      std::fprintf(stderr, "distribute: graph too large for int-count scatter, "
                           "or weights missing\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int r = 0; r < d.size; ++r) {
      rc[r] = static_cast<int>(d.starts[r + 1] - d.starts[r] +
                               1); // overlapping slices of row_ptr
      rd[r] = static_cast<int>(d.starts[r]);
      const eid_t eb = full->row_ptr[d.starts[r]],
                  ee = full->row_ptr[d.starts[r + 1]];
      ec[r] = static_cast<int>(ee - eb);
      ed[r] = static_cast<int>(eb);
    }
  }

  d.row_ptr.resize(d.local_n() + 1);
  MPI_Scatterv(root ? full->row_ptr.data() : nullptr, rc.data(), rd.data(),
               MPI_UINT64_T, d.row_ptr.data(),
               static_cast<int>(d.row_ptr.size()), MPI_UINT64_T, 0,
               MPI_COMM_WORLD);
  const eid_t base = d.row_ptr.front();
  const eid_t cnt = d.row_ptr.back() - base;
  for (eid_t &x : d.row_ptr)
    x -= base;

  d.col_idx.resize(cnt);
  MPI_Scatterv(root ? full->col_idx.data() : nullptr, ec.data(), ed.data(),
               MPI_UINT32_T, d.col_idx.data(), static_cast<int>(cnt),
               MPI_UINT32_T, 0, MPI_COMM_WORLD);
  if (with_weights) {
    d.weights.resize(cnt);
    MPI_Scatterv(root ? full->weights.data() : nullptr, ec.data(), ed.data(),
                 MPI_UINT32_T, d.weights.data(), static_cast<int>(cnt),
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);
  }
  return d;
}