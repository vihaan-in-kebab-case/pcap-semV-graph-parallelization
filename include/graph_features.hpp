#pragma once
#include "graph_csr.hpp"
#include <cstdint>

// Cheap, machine-independent graph properties used by the selector.
struct GraphFeatures {
  std::uint64_t n = 0, m = 0;
  double mean_degree = 0.0;   // m / n
  double degree_skew = 0.0;   // max out-degree / mean degree
  double out_cv = 0.0;        // std / mean of out-degree
  double dangling_frac = 0.0; // share of vertices with out-degree 0
  bool symmetric = false;
};

GraphFeatures extract_features(const GraphCSR &g, bool declared_symmetric);