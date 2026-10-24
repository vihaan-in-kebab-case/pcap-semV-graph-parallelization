#pragma once
#include "graph_csr.hpp"
#include <cstdint>
#include <limits>
#include <vector>

constexpr std::uint64_t INF_DIST = std::numeric_limits<std::uint64_t>::max();

struct BFSResult {
  std::vector<std::int32_t> level;
};
struct PageRankParams {
  double damping = 0.85;
  double tol = 1e-8;
  int max_iters = 100;
};
struct PageRankResult {
  std::vector<double> rank;
  int iters = 0;
  double residual = 0.0;
};
struct CCResult {
  std::vector<vid_t> label;
  vid_t num_components = 0;
};
struct SSSPResult {
  std::vector<std::uint64_t> dist;
};
struct TCResult {
  std::uint64_t triangles = 0;
};

namespace seq {
BFSResult bfs(const GraphCSR &g, vid_t source);
PageRankResult pagerank(const GraphCSR &g, const GraphCSR &gT,
                        const PageRankParams &p);
CCResult connected_components(const GraphCSR &g);
SSSPResult sssp(const GraphCSR &g, vid_t source);
TCResult triangle_count(const GraphCSR &g);
} // namespace seq

namespace omp {
BFSResult bfs(const GraphCSR &g, vid_t source);
PageRankResult pagerank(const GraphCSR &g, const GraphCSR &gT,
                        const PageRankParams &p);
CCResult connected_components(const GraphCSR &g);
SSSPResult sssp(const GraphCSR &g, vid_t source);
TCResult triangle_count(const GraphCSR &g);
} // namespace omp