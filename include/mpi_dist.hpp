#pragma once
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "algorithm_api.hpp"

// 1D block partition: rank r owns vertices [starts[r], starts[r+1]) and their
// out-edges.
struct DistGraph {
  vid_t n = 0;          // global vertex count
  vid_t lo = 0, hi = 0; // owned range
  int rank = 0, size = 1;
  std::vector<vid_t> starts;  // size+1 block boundaries
  std::vector<eid_t> row_ptr; // local offsets, size local_n()+1, starts at 0
  std::vector<vid_t> col_idx; // GLOBAL neighbour ids
  std::vector<std::uint32_t> weights; // optional, parallel to col_idx

  vid_t local_n() const { return hi - lo; }
  int owner(vid_t v) const {
    return static_cast<int>(std::upper_bound(starts.begin(), starts.end(), v) -
                            starts.begin()) -
           1;
  }
};

// Accumulates wall time spent inside MPI calls.
struct CommTimer {
  double &acc;
  double t0;
  explicit CommTimer(double &a) : acc(a), t0(MPI_Wtime()) {}
  ~CommTimer() { acc += MPI_Wtime() - t0; }
};

// Collective. Rank 0 passes the full graph (others pass nullptr) and everyone
// gets its slice.
DistGraph distribute(const GraphCSR *full, bool with_weights);

inline std::uint64_t sum_u64(std::uint64_t x, double &comm_s) {
  std::uint64_t r = 0;
  CommTimer t(comm_s);
  MPI_Allreduce(&x, &r, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  return r;
}
inline double sum_f64(double x, double &comm_s) {
  double r = 0;
  CommTimer t(comm_s);
  MPI_Allreduce(&x, &r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return r;
}

// Gathers every rank's slice into one length-n vector on rank 0 (empty
// elsewhere). Not timed.
template <class T>
std::vector<T> gather_to_root(const DistGraph &d, const std::vector<T> &local,
                              MPI_Datatype ty) {
  std::vector<int> cnt(d.size), disp(d.size);
  for (int r = 0; r < d.size; ++r) {
    cnt[r] = static_cast<int>(d.starts[r + 1] - d.starts[r]);
    disp[r] = static_cast<int>(d.starts[r]);
  }
  std::vector<T> all;
  if (d.rank == 0)
    all.resize(d.n);
  MPI_Gatherv(local.data(), static_cast<int>(local.size()), ty, all.data(),
              cnt.data(), disp.data(), ty, 0, MPI_COMM_WORLD);
  return all;
}

// Every rank ends up with the full length-n vector.
template <class T>
void allgather_vec(const DistGraph &d, const std::vector<T> &local,
                   std::vector<T> &global, MPI_Datatype ty, double &comm_s) {
  std::vector<int> cnt(d.size), disp(d.size);
  for (int r = 0; r < d.size; ++r) {
    cnt[r] = static_cast<int>(d.starts[r + 1] - d.starts[r]);
    disp[r] = static_cast<int>(d.starts[r]);
  }
  global.resize(d.n);
  CommTimer t(comm_s);
  MPI_Allgatherv(local.data(), static_cast<int>(local.size()), ty,
                 global.data(), cnt.data(), disp.data(), ty, MPI_COMM_WORLD);
}

// out[r] = items for rank r; returns everything addressed to this rank. Packing
// is not timed.
template <class T>
std::vector<T> alltoallv_vec(const DistGraph &d,
                             const std::vector<std::vector<T>> &out,
                             MPI_Datatype ty, double &comm_s) {
  const int P = d.size;
  std::vector<int> sc(P), sd(P), rc(P), rd(P);
  std::vector<T> sendbuf;
  int tot = 0;
  for (int r = 0; r < P; ++r) {
    sc[r] = static_cast<int>(out[r].size());
    sd[r] = tot;
    sendbuf.insert(sendbuf.end(), out[r].begin(), out[r].end());
    tot += sc[r];
  }
  {
    CommTimer t(comm_s);
    MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, MPI_COMM_WORLD);
  }
  int rt = 0;
  for (int r = 0; r < P; ++r) {
    rd[r] = rt;
    rt += rc[r];
  }
  std::vector<T> recv(rt);
  {
    CommTimer t(comm_s);
    MPI_Alltoallv(sendbuf.data(), sc.data(), sd.data(), ty, recv.data(),
                  rc.data(), rd.data(), ty, MPI_COMM_WORLD);
  }
  return recv;
}

// Distributed-memory implementations. Results hold THIS RANK'S slice [lo,hi),
// except CCResult::num_components and TCResult::triangles, which are global on
// every rank.
namespace mpi {
struct Stats {
  double comm_s = 0.0;
}; // wall time this rank spent inside MPI calls

BFSResult bfs(const DistGraph &g, vid_t source, Stats &st);
PageRankResult pagerank(const DistGraph &g, const DistGraph &gT,
                        const PageRankParams &p, Stats &st);
CCResult connected_components(const DistGraph &g, Stats &st);
SSSPResult sssp(const DistGraph &g, vid_t source, Stats &st);
TCResult triangle_count(const DistGraph &g, Stats &st);
} // namespace mpi