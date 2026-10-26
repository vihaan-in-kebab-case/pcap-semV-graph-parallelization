#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "algorithm_api.hpp"
#include "benchmark_record.hpp"
#include "correctness.hpp"
#include "cuda_common.hpp"

namespace {
struct Options {
  std::string graph, algo = "all", out;
  bool undirected = false;
  int trials = 5, warmup = 1, device = 0;
  long source = -1;
  std::uint64_t seed = 42;
};

[[noreturn]] void usage() {
  std::cerr << "usage: gps_bench_cuda --graph FILE [--algo "
               "bfs|pagerank|cc|sssp|tc|all] [--device N]\n"
               "       [--undirected] [--trials N] [--warmup N] [--source V] "
               "[--seed S] [--out FILE.csv]\n";
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
    else if (a == "--trials")
      o.trials = std::stoi(next());
    else if (a == "--warmup")
      o.warmup = std::stoi(next());
    else if (a == "--device")
      o.device = std::stoi(next());
    else if (a == "--source")
      o.source = std::stol(next());
    else if (a == "--seed")
      o.seed = std::stoull(next());
    else if (a == "--undirected")
      o.undirected = true;
    else
      usage();
  }
  if (o.graph.empty() || o.trials < 1 || o.warmup < 0)
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
  const std::vector<std::string> known = {"bfs", "pagerank", "cc", "sssp",
                                          "tc"};
  if (o.algo != "all" &&
      std::find(known.begin(), known.end(), o.algo) == known.end())
    usage();
  auto want = [&](const std::string &a) {
    return o.algo == "all" || o.algo == a;
  };

  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
    std::cerr << "error: no CUDA device found\n";
    return 1;
  }
  CUDA_CHECK(cudaSetDevice(o.device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, o.device));
  CUDA_CHECK(cudaFree(nullptr)); // create the context before any timing

  GraphCSR g;
  LoadTimings lt;
  try {
    g = load_graph(o.graph, o.undirected, &lt);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  if (g.n == 0) {
    std::cerr << "error: empty graph\n";
    return 1;
  }
  const double load_s = lt.read_s + lt.build_s;
  const std::string dataset = base_name(o.graph);
  std::cout << "loaded " << dataset << ": n=" << g.n << " m=" << g.m
            << " load=" << load_s << "s  impl=cuda gpu=\"" << prop.name
            << "\" sm_" << prop.major << prop.minor << " warmup=" << o.warmup
            << " trials=" << o.trials << "\n";

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

  // up = graph upload cost (constant per run); compute(Times&) runs one trial.
  auto run = [&](const std::string &name, const GraphCSR &G, double pre_s,
                 const gpu::Times &up, auto compute, auto validate,
                 auto summarize) {
    std::vector<double> kernel_times, wall_times;
    std::string err, summary;
    for (int w = 0; w < o.warmup; ++w) {
      gpu::Times tmp;
      (void)compute(tmp);
    }
    for (int t = 0; t < o.trials; ++t) {
      gpu::Times tm;
      CUDA_CHECK(cudaDeviceSynchronize());
      const double w0 = gpu::wall_s();
      auto res = compute(tm);
      CUDA_CHECK(cudaDeviceSynchronize());
      const double wall = gpu::wall_s() - w0;
      if (t == 0) {
        err = validate(res);
        summary = summarize(res);
      }
      BenchmarkRecord r;
      r.algorithm = name;
      r.impl = "cuda";
      r.dataset = dataset;
      r.n = G.n;
      r.m = G.m;
      r.trial = t;
      r.threads = 1;
      r.load_s = load_s;
      r.preprocess_s = pre_s;
      r.alloc_s = up.alloc_s + tm.alloc_s;
      r.h2d_s = up.h2d_s;
      r.d2h_s = tm.d2h_s;
      r.compute_s = tm.compute_s;
      r.e2e_s = load_s + pre_s + up.alloc_s + up.h2d_s + wall;
      r.correct = err.empty();
      r.note = err;
      if (!o.out.empty())
        append_record(o.out, r);
      kernel_times.push_back(tm.compute_s);
      wall_times.push_back(wall);
    }
    std::cout << "[" << name << "] n=" << G.n << " m=" << G.m
              << " preprocess=" << pre_s << "s graph_h2d=" << up.h2d_s
              << "s median_kernel=" << median(kernel_times)
              << "s median_trial_wall=" << median(wall_times)
              << "s correct=" << (err.empty() ? "yes" : "NO (" + err + ")")
              << "\n      " << summary << "\n";
  };

  if (want("bfs")) {
    gpu::DeviceGraph dg;
    gpu::Times up;
    gpu::upload_graph(g, dg, up, false);
    run(
        "bfs", g, 0.0, up, [&](gpu::Times &t) { return gpu::bfs(dg, src, t); },
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
    const GraphCSR *gT = &g;
    if (!is_symmetric(g)) {
      gT_storage = transposed(g);
      gT = &gT_storage;
    }
    const double transpose_s = tt.elapsed();
    gpu::DeviceGraph dg, dgT_storage;
    gpu::Times up;
    gpu::upload_graph(g, dg, up, false);
    const gpu::DeviceGraph *dgT = &dg;
    if (gT != &g) {
      gpu::upload_graph(*gT, dgT_storage, up, false);
      dgT = &dgT_storage;
    }
    run(
        "pagerank", g, transpose_s, up,
        [&](gpu::Times &t) { return gpu::pagerank(dg, *dgT, p, t); },
        [&](const PageRankResult &r) -> std::string {
          std::string e = check_pagerank(r, p);
          if (!e.empty())
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
    gpu::DeviceGraph dg;
    gpu::Times up;
    gpu::upload_graph(*gs, dg, up, false);
    run(
        "cc", *gs, sym_s, up,
        [&](gpu::Times &t) { return gpu::connected_components(dg, t); },
        [&](const CCResult &r) { return check_cc(*gs, r.label); },
        [&](const CCResult &r) {
          return "components=" + std::to_string(r.num_components);
        });
  }
  if (want("sssp")) {
    Timer t;
    attach_synthetic_weights(g, o.seed);
    const double w_s = t.elapsed();
    gpu::DeviceGraph dg;
    gpu::Times up;
    gpu::upload_graph(g, dg, up, true);
    run(
        "sssp", g, w_s, up,
        [&](gpu::Times &tt) { return gpu::sssp(dg, src, tt); },
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
    gpu::DeviceGraph dg;
    gpu::Times up;
    gpu::upload_graph(*gs, dg, up, false);
    run(
        "tc", *gs, sym_s, up,
        [&](gpu::Times &t) { return gpu::triangle_count(dg, t); },
        [&](const TCResult &r) { return check_tc(*gs, r.triangles); },
        [&](const TCResult &r) {
          return "triangles=" + std::to_string(r.triangles);
        });
  }
  return 0;
}