#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "algorithm_api.hpp"
#include "benchmark_record.hpp"
#include "correctness.hpp"

namespace {
struct Options {
  std::string graph, algo = "all", out;
  std::string impl = "seq"; // seq | omp
  bool undirected = false;
  int trials = 5;
  int threads = 1;  // omp only
  int warmup = 1;   // untimed runs before the timed trials
  long source = -1; // -1 => highest out-degree vertex (compact id)
  std::uint64_t seed = 42;
};

[[noreturn]] void usage() {
  std::cerr
      << "usage: gps_bench --graph FILE [--algo bfs|pagerank|cc|sssp|tc|all]\n"
         "                 [--impl seq|omp] [--threads N] [--warmup N] "
         "[--trials N]\n"
         "                 [--undirected] [--source V] [--seed S] [--out "
         "FILE.csv]\n";
  std::exit(2);
}

Options parse(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc)
        usage();
      return argv[++i];
    };
    if (a == "--graph")
      o.graph = next();
    else if (a == "--algo")
      o.algo = next();
    else if (a == "--out")
      o.out = next();
    else if (a == "--impl")
      o.impl = next();
    else if (a == "--trials")
      o.trials = std::stoi(next());
    else if (a == "--threads")
      o.threads = std::stoi(next());
    else if (a == "--warmup")
      o.warmup = std::stoi(next());
    else if (a == "--source")
      o.source = std::stol(next());
    else if (a == "--seed")
      o.seed = std::stoull(next());
    else if (a == "--undirected")
      o.undirected = true;
    else
      usage();
  }
  if (o.graph.empty())
    usage();
  return o;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

std::string base_name(const std::string &p) {
  auto pos = p.find_last_of("/\\");
  return pos == std::string::npos ? p : p.substr(pos + 1);
}
} // namespace

int main(int argc, char **argv) {
  const Options o = parse(argc, argv);
  if (o.impl != "seq" && o.impl != "omp")
    usage();
  if (o.trials < 1 || o.threads < 1 || o.warmup < 0)
    usage();
  const bool use_omp = (o.impl == "omp");
  if (use_omp)
    omp_set_num_threads(o.threads);

  GraphCSR g;
  LoadTimings lt;
  try {
    g = load_graph(o.graph, o.undirected, &lt);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  const double load_s = lt.read_s + lt.build_s;
  const std::string dataset = base_name(o.graph);
  std::cout << "loaded " << dataset << ": n=" << g.n << " m=" << g.m
            << " (directed edges after cleanup) load=" << load_s << "s"
            << "  impl=" << o.impl;
  if (use_omp)
    std::cout << " threads=" << o.threads;
  std::cout << " warmup=" << o.warmup << " trials=" << o.trials << "\n";

  const std::vector<std::string> known = {"bfs", "pagerank", "cc", "sssp",
                                          "tc"};
  if (o.algo != "all" &&
      std::find(known.begin(), known.end(), o.algo) == known.end())
    usage();
  auto want = [&](const std::string &a) {
    return o.algo == "all" || o.algo == a;
  };

  if (g.n == 0) {
    std::cerr << "error: empty graph\n";
    return 1;
  }
  vid_t src = 0;
  if (o.source >= 0)
    src = static_cast<vid_t>(o.source);
  else
    for (vid_t v = 0; v < g.n; ++v)
      if (g.out_degree(v) > g.out_degree(src))
        src = v;
  if (src >= g.n) {
    std::cerr << "error: source out of range\n";
    return 1;
  }

  auto run = [&](const std::string &name, const GraphCSR &G, double pre_s,
                 auto compute, auto validate, auto summarize) {
    std::vector<double> times;
    std::string err, summary;
    for (int w = 0; w < o.warmup; ++w)
      (void)compute(); // untimed warm-up, discarded
    for (int t = 0; t < o.trials; ++t) {
      Timer tm;
      auto res = compute();
      const double c = tm.elapsed();
      if (t == 0) {
        err = validate(res);
        summary = summarize(res);
      }
      BenchmarkRecord r;
      r.algorithm = name;
      r.impl = o.impl;
      r.dataset = dataset;
      r.n = G.n;
      r.m = G.m;
      r.trial = t;
      r.threads = use_omp ? o.threads : 1;
      r.load_s = load_s;
      r.preprocess_s = pre_s;
      r.compute_s = c;
      r.e2e_s = load_s + pre_s + c;
      r.correct = err.empty();
      r.note = err;
      if (!o.out.empty())
        append_record(o.out, r);
      times.push_back(c);
    }
    std::cout << "[" << name << "] n=" << G.n << " m=" << G.m
              << " preprocess=" << pre_s << "s median_compute=" << median(times)
              << "s correct=" << (err.empty() ? "yes" : "NO (" + err + ")")
              << "\n      " << summary << "\n";
  };

  if (want("bfs")) {
    run(
        "bfs", g, 0.0,
        [&] { return use_omp ? omp::bfs(g, src) : seq::bfs(g, src); },
        [&](const BFSResult &r) { return check_bfs(g, src, r.level); },
        [&](const BFSResult &r) {
          size_t reached = 0;
          int mx = 0;
          for (int l : r.level)
            if (l >= 0) {
              ++reached;
              mx = std::max(mx, l);
            }
          std::ostringstream s;
          s << "source=" << src << " reached=" << reached
            << " max_level=" << mx;
          return s.str();
        });
  }
  if (want("pagerank")) {
    PageRankParams p;
    Timer tt;
    GraphCSR gT_storage;
    const GraphCSR *gT = &g; // symmetric: transpose is the graph itself
    if (!is_symmetric(g)) {
      gT_storage = transposed(g);
      gT = &gT_storage;
    }
    const double transpose_s = tt.elapsed(); // reported as preprocess_s
    run(
        "pagerank", g, transpose_s,
        [&] {
          return use_omp ? omp::pagerank(g, *gT, p) : seq::pagerank(g, *gT, p);
        },
        [&](const PageRankResult &r) -> std::string {
          std::string e = check_pagerank(r, p);
          if (!e.empty() || !use_omp)
            return e;
          auto ref = seq::pagerank(g, *gT, p); // untimed, trial 0 only
          return pagerank_l1_close(r.rank, ref.rank, 10 * p.tol)
                     ? ""
                     : "pagerank: differs from sequential reference";
        },
        [&](const PageRankResult &r) {
          size_t top =
              std::max_element(r.rank.begin(), r.rank.end()) - r.rank.begin();
          std::ostringstream s;
          s << "iters=" << r.iters << " residual=" << r.residual
            << " top_vertex=" << top << " top_rank=" << r.rank[top];
          return s.str();
        });
  }

  const GraphCSR *gs = &g;
  GraphCSR gsym;
  double sym_s = 0.0;
  if (want("cc") || want("tc")) {
    Timer t;
    if (!is_symmetric(g)) {
      gsym = symmetrized(g);
      gs = &gsym;
    }
    sym_s = t.elapsed();
  }
  if (want("cc")) {
    run(
        "cc", *gs, sym_s,
        [&] {
          return use_omp ? omp::connected_components(*gs)
                         : seq::connected_components(*gs);
        },
        [&](const CCResult &r) { return check_cc(*gs, r.label); },
        [&](const CCResult &r) {
          return "components=" + std::to_string(r.num_components);
        });
  }
  if (want("sssp")) {
    Timer t;
    attach_synthetic_weights(g, o.seed);
    const double w_s = t.elapsed();
    run(
        "sssp", g, w_s,
        [&] { return use_omp ? omp::sssp(g, src) : seq::sssp(g, src); },
        [&](const SSSPResult &r) { return check_sssp(g, src, r.dist); },
        [&](const SSSPResult &r) {
          size_t reached = 0;
          std::uint64_t mx = 0;
          for (auto d : r.dist)
            if (d != INF_DIST) {
              ++reached;
              mx = std::max(mx, d);
            }
          std::ostringstream s;
          s << "source=" << src << " weight_seed=" << o.seed
            << " reached=" << reached << " max_dist=" << mx;
          return s.str();
        });
  }
  if (want("tc")) {
    run(
        "tc", *gs, sym_s,
        [&] {
          return use_omp ? omp::triangle_count(*gs) : seq::triangle_count(*gs);
        },
        [&](const TCResult &r) { return check_tc(*gs, r.triangles); },
        [&](const TCResult &r) {
          return "triangles=" + std::to_string(r.triangles);
        });
  }
  return 0;
}