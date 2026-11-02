#pragma once
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

struct Timer {
  using clock = std::chrono::steady_clock;
  clock::time_point t0 = clock::now();
  void reset() { t0 = clock::now(); }
  double elapsed() const {
    return std::chrono::duration<double>(clock::now() - t0).count();
  }
};

struct BenchmarkRecord {
  std::string algorithm, impl, dataset, note;
  std::uint64_t n = 0, m = 0;
  int trial = 0;
  unsigned threads = 1;
  double load_s = -1, preprocess_s = -1, alloc_s = -1, h2d_s = -1, d2h_s = -1,
         feature_s = -1, classifier_s = -1, setup_s = -1, compute_s = -1,
         comm_s = -1, e2e_s = -1;
  bool correct = true;
};

namespace detail {
inline std::string num(double v) {
  if (v < 0)
    return "";
  char b[40];
  std::snprintf(b, sizeof b, "%.9g", v);
  return b;
}
inline std::string clean(std::string s) {
  for (char &c : s)
    if (c == ',' || c == '\n')
      c = ';';
  return s;
}
} // namespace detail

inline const char *record_header() {
  return "algorithm,impl,dataset,n,m,trial,threads,load_s,preprocess_s,alloc_s,"
         "h2d_s,d2h_s,feature_s,classifier_s,setup_s,compute_s,comm_s,e2e_s,"
         "correct,note";
}

inline void append_record(const std::string &path, const BenchmarkRecord &r) {
  bool need_header;
  {
    std::ifstream f(path);
    need_header = !(f.good() && f.peek() != std::ifstream::traits_type::eof());
  }
  std::ofstream out(path, std::ios::app);
  if (need_header)
    out << record_header() << "\n";
  using detail::num;
  out << r.algorithm << ',' << r.impl << ',' << r.dataset << ',' << r.n << ','
      << r.m << ',' << r.trial << ',' << r.threads << ',' << num(r.load_s)
      << ',' << num(r.preprocess_s) << ',' << num(r.alloc_s) << ','
      << num(r.h2d_s) << ',' << num(r.d2h_s) << ',' << num(r.feature_s) << ','
      << num(r.classifier_s) << ',' << num(r.setup_s) << ',' << num(r.compute_s)
      << ',' << num(r.comm_s) << ',' << num(r.e2e_s) << ','
      << (r.correct ? 1 : 0) << ',' << detail::clean(r.note) << "\n";
}