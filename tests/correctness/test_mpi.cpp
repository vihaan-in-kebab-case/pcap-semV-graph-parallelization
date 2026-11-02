#include <mpi.h>

#include "algorithm_api.hpp"
#include "correctness.hpp"
#include "mpi_dist.hpp"
#include <iostream>
#include <random>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int failures = 0;
  auto check = [&](bool ok, const std::string &msg) {
    if (!ok) {
      ++failures;
      if (rank == 0)
        std::cerr << "FAIL: " << msg << "\n";
    }
  };

  auto run_case = [&](vid_t n, const std::vector<Edge> &e, vid_t src,
                      std::uint64_t seed, const std::string &tag) {
    GraphCSR gd = build_csr(n, e, false), gu = build_csr(n, e, true);
    attach_synthetic_weights(gd, seed);
    GraphCSR gdT = transposed(gd);
    const bool root = (rank == 0);
    mpi::Stats st;
    { // BFS
      DistGraph dg = distribute(root ? &gd : nullptr, false);
      auto lv = gather_to_root(dg, mpi::bfs(dg, src, st).level, MPI_INT32_T);
      if (root)
        check(lv == seq::bfs(gd, src).level && check_bfs(gd, src, lv).empty(),
              "bfs " + tag);
    }
    { // PageRank
      PageRankParams p;
      DistGraph dg = distribute(root ? &gd : nullptr, false),
                dgT = distribute(root ? &gdT : nullptr, false);
      auto r = mpi::pagerank(dg, dgT, p, st);
      auto all = gather_to_root(dg, r.rank, MPI_DOUBLE);
      if (root)
        check(
            pagerank_l1_close(all, seq::pagerank(gd, gdT, p).rank, 10 * p.tol),
            "pagerank " + tag);
    }
    { // CC
      DistGraph dg = distribute(root ? &gu : nullptr, false);
      auto r = mpi::connected_components(dg, st);
      auto lab = gather_to_root(dg, r.label, MPI_UINT32_T);
      if (root)
        check(check_cc(gu, lab).empty() &&
                  r.num_components ==
                      seq::connected_components(gu).num_components,
              "cc " + tag);
    }
    { // SSSP
      DistGraph dg = distribute(root ? &gd : nullptr, true);
      auto d = gather_to_root(dg, mpi::sssp(dg, src, st).dist, MPI_UINT64_T);
      if (root)
        check(d == seq::sssp(gd, src).dist && check_sssp(gd, src, d).empty(),
              "sssp " + tag);
    }
    { // TC
      DistGraph dg = distribute(root ? &gu : nullptr, false);
      auto t = mpi::triangle_count(dg, st);
      if (root)
        check(t.triangles == seq::triangle_count(gu).triangles, "tc " + tag);
    }
  };

  run_case(2, {{0, 1}}, 0, 1, "n=2 (empty blocks at high rank counts)");
  run_case(7, {{0, 1}, {0, 2}, {1, 2}, {2, 3}, {3, 4}, {5, 6}}, 0, 2, "tiny");
  for (std::uint64_t s = 1; s <= 10; ++s) {
    std::mt19937_64 rng(2000 + s);
    const vid_t n = 20 + static_cast<vid_t>(rng() % 200);
    std::vector<Edge> e;
    for (vid_t k = 0; k < n * 4; ++k)
      e.emplace_back(rng() % n, rng() % n);
    for (vid_t k = 0; k < n / 2; ++k)
      e.emplace_back(0, rng() % n); // hub => skew
    run_case(n, e, static_cast<vid_t>(rng() % n), s,
             "random seed " + std::to_string(s));
  }
  int total = 0;
  MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0 && total == 0)
    std::cout << "all mpi tests passed\n";
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}