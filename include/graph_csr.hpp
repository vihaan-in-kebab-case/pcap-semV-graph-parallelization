#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using vid_t = std::uint32_t;
using eid_t = std::uint64_t;
using Edge = std::pair<vid_t, vid_t>;

struct GraphCSR {
  vid_t n = 0;
  eid_t m = 0;
  std::vector<eid_t> row_ptr;
  std::vector<vid_t> col_idx;
  std::vector<std::uint32_t> weights;

  eid_t out_degree(vid_t v) const { return row_ptr[v + 1] - row_ptr[v]; }
  bool weighted() const { return weights.size() == col_idx.size(); }
};

struct LoadTimings {
  double read_s = 0.0;
  double build_s = 0.0;
};

GraphCSR build_csr(vid_t n, std::vector<Edge> edges, bool symmetrize);
GraphCSR load_graph(const std::string &path, bool symmetrize,
                    LoadTimings *t = nullptr);
GraphCSR symmetrized(const GraphCSR &g);
bool is_symmetric(const GraphCSR &g);
// In-edge CSR (row v lists every u with an edge u->v). Weights are not carried
// over.
GraphCSR transposed(const GraphCSR &g);
void attach_synthetic_weights(GraphCSR &g, std::uint64_t seed,
                              std::uint32_t max_w = 100);