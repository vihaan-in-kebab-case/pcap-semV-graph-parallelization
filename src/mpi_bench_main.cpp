#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "algorithm_api.hpp"
#include "benchmark_record.hpp"
#include "correctness.hpp"
#include "mpi_dist.hpp"

namespace {
struct Options {
  std::string graph, algo = "all", out;
  bool undirected = false;
  int trials = 5, warmup = 1;
  long source = -1;
  std::uint64_t seed = 42;
};

[[noreturn]] void usage(int rank) {
  if (rank == 0)
    std::cerr << "usage: mpirun -np P gps_bench_mpi --graph FILE [--algo "
                 "bfs|pagerank|cc|sssp|tc|all]\n"
                 "       [--undirected] [--trials N] [--warmup N] [--source V] "
                 "[--seed S] [--out FILE.csv]\n";
  MPI_Abort(MPI_COMM_WORLD, 2);
  std::exit(2);
}

Options parse(int argc, char **argv, int rank) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc)
        usage(rank);
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
    else if (a == "--source")
      o.source = std::stol(next());
    else if (a == "--seed")
      o.seed = std::stoull(next());
    else if (a == "--undirected")
      o.undirected = true;
    else
      usage(rank);
  }
  if (o.graph.empty() || o.trials < 1 || o.warmup < 0)
    usage(rank);
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
double max_over_ranks(double x) {
  double r = 0.0;
  MPI_Allreduce(&x, &r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return r;
}
} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const bool root = (rank == 0);
  const Options o = parse(argc, argv, rank);
  const std::vector<std::string> known = {"bfs", "pagerank", "cc", "sssp",
                                          "tc"};
  if (o.algo != "all" &&
      std::find(known.begin(), known.end(), o.algo) == known.end())
    usage(rank);
  auto want = [&](const std::string &a) {
    return o.algo == "all" || o.algo == a;
  };

  // Rank 0 loads and preprocesses; ranks receive slices via distribute().
  GraphCSR g;
  double load_s = 0.0;
  std::uint64_t n64 = 0;
  if (root) {
    try {
      LoadTimings lt;
      g = load_graph(o.graph, o.undirected, &lt);
      load_s = lt.read_s + lt.build_s;
      n64 = g.n;
    } catch (const std::exception &e) {
      std::cerr << "error: " << e.what() << "\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  MPI_Bcast(&load_s, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  if (n64 == 0) {
    if (root)
      std::cerr << "error: empty graph\n";
    MPI_Finalize();
    return 1;
  }

  vid_t src = 0;
  if (root) {
    if (o.source >= 0)
      src = static_cast<vid_t>(o.source);
    else
      for (vid_t v = 0; v < g.n; ++v)
        if (g.out_degree(v) > g.out_degree(src))
          src = v;
    if (src >= g.n) {
      std::cerr << "error: source out of range\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  MPI_Bcast(&src, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

  const std::string dataset = base_name(o.graph);
  if (root)
    std::cout << "loaded " << dataset << ": n=" << n64 << " m=" << g.m
              << " load=" << load_s << "s  impl=mpi ranks=" << size
              << " warmup=" << o.warmup << " trials=" << o.trials << "\n";

  auto timed_setup = [&](auto &&fn) {
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    fn();
    return max_over_ranks(MPI_Wtime() - t0);
  };
  auto bcast_d = [&](double x) {
    MPI_Bcast(&x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return x;
  };

  // compute(Stats&) -> local result; inspect(result) is COLLECTIVE and returns
  // {error, summary} (valid on rank 0).
  auto run = [&](const std::string &name, const GraphCSR *Gv, double pre_s,
                 double setup_s, auto compute, auto inspect) {
    std::vector<double> times, comms;
    std::string err, summary;
    for (int w = 0; w < o.warmup; ++w) {
      mpi::Stats st;
      MPI_Barrier(MPI_COMM_WORLD);
      (void)compute(st);
    }
    for (int t = 0; t < o.trials; ++t) {
      mpi::Stats st;
      MPI_Barrier(MPI_COMM_WORLD);
      const double t0 = MPI_Wtime();
      auto res = compute(st);
      const double c = max_over_ranks(MPI_Wtime() - t0);
      const double cm = max_over_ranks(st.comm_s);
      if (t == 0) {
        auto p = inspect(res);
        err = p.first;
        summary = p.second;
      }
      if (root) {
        BenchmarkRecord r;
        r.algorithm = name;
        r.impl = "mpi";
        r.dataset = dataset;
        r.n = Gv->n;
        r.m = Gv->m;
        r.trial = t;
        r.threads = static_cast<unsigned>(size); // = ranks
        r.load_s = load_s;
        r.preprocess_s = pre_s;
        r.setup_s = setup_s;
        r.compute_s = c;
        r.comm_s = cm;
        r.e2e_s = load_s + pre_s + setup_s + c;
        r.correct = err.empty();
        r.note = err;
        if (!o.out.empty())
          append_record(o.out, r);
        times.push_back(c);
        comms.push_back(cm);
      }
    }
    if (root)
      std::cout << "[" << name << "] n=" << Gv->n << " m=" << Gv->m
                << " preprocess=" << pre_s << "s setup=" << setup_s
                << "s median_compute=" << median(times)
                << "s median_comm=" << median(comms)
                << "s correct=" << (err.empty() ? "yes" : "NO (" + err + ")")
                << "\n      " << summary << "\n";
  };
  using Pair = std::pair<std::string, std::string>;

  if (want("bfs")) {
    DistGraph dg;
    const double setup =
        timed_setup([&] { dg = distribute(root ? &g : nullptr, false); });
    run(
        "bfs", &g, 0.0, setup,
        [&](mpi::Stats &st) { return mpi::bfs(dg, src, st); },
        [&](const BFSResult &r) -> Pair {
          auto lv = gather_to_root(dg, r.level, MPI_INT32_T);
          if (!root)
            return {};
          size_t reached = 0;
          int mx = 0;
          for (int l : lv)
            if (l >= 0) {
              ++reached;
              mx = std::max(mx, l);
            }
          std::ostringstream s;
          s << "source=" << src << " reached=" << reached
            << " max_level=" << mx;
          return {check_bfs(g, src, lv), s.str()};
        });
  }
  if (want("pagerank")) {
    PageRankParams p;
    GraphCSR gT_storage;
    const GraphCSR *gT = &g;
    double pre = 0.0;
    if (root) {
      Timer tt;
      if (!is_symmetric(g)) {
        gT_storage = transposed(g);
        gT = &gT_storage;
      }
      pre = tt.elapsed();
    }
    pre = bcast_d(pre);
    DistGraph dg, dgT;
    const double setup = timed_setup([&] {
      dg = distribute(root ? &g : nullptr, false);
      dgT = distribute(root ? gT : nullptr, false);
    });
    run(
        "pagerank", &g, pre, setup,
        [&](mpi::Stats &st) { return mpi::pagerank(dg, dgT, p, st); },
        [&](const PageRankResult &r) -> Pair {
          auto all = gather_to_root(dg, r.rank, MPI_DOUBLE);
          if (!root)
            return {};
          PageRankResult full;
          full.rank = std::move(all);
          full.iters = r.iters;
          full.residual = r.residual;
          std::string e = check_pagerank(full, p);
          if (e.empty()) {
            auto ref = seq::pagerank(g, *gT, p); // untimed, trial 0 only
            if (!pagerank_l1_close(full.rank, ref.rank, 10 * p.tol))
              e = "pagerank: differs from sequential reference";
          }
          size_t top = std::max_element(full.rank.begin(), full.rank.end()) -
                       full.rank.begin();
          std::ostringstream s;
          s << "iters=" << full.iters << " residual=" << full.residual
            << " top_vertex=" << top << " top_rank=" << full.rank[top];
          return {e, s.str()};
        });
  }

  const GraphCSR *gs = &g;
  GraphCSR gsym;
  double sym_s = 0.0;
  if (want("cc") || want("tc")) {
    if (root) {
      Timer t;
      if (!is_symmetric(g)) {
        gsym = symmetrized(g);
        gs = &gsym;
      }
      sym_s = t.elapsed();
    }
    sym_s = bcast_d(sym_s);
  }
  if (want("cc")) {
    DistGraph dg;
    const double setup =
        timed_setup([&] { dg = distribute(root ? gs : nullptr, false); });
    run(
        "cc", gs, sym_s, setup,
        [&](mpi::Stats &st) { return mpi::connected_components(dg, st); },
        [&](const CCResult &r) -> Pair {
          auto lab = gather_to_root(dg, r.label, MPI_UINT32_T);
          if (!root)
            return {};
          return {check_cc(*gs, lab),
                  "components=" + std::to_string(r.num_components) +
                      " (label propagation)"};
        });
  }
  if (want("sssp")) {
    double w_s = 0.0;
    if (root) {
      Timer t;
      attach_synthetic_weights(g, o.seed);
      w_s = t.elapsed();
    }
    w_s = bcast_d(w_s);
    DistGraph dg;
    const double setup =
        timed_setup([&] { dg = distribute(root ? &g : nullptr, true); });
    run(
        "sssp", &g, w_s, setup,
        [&](mpi::Stats &st) { return mpi::sssp(dg, src, st); },
        [&](const SSSPResult &r) -> Pair {
          auto d = gather_to_root(dg, r.dist, MPI_UINT64_T);
          if (!root)
            return {};
          size_t reached = 0;
          std::uint64_t mx = 0;
          for (auto x : d)
            if (x != INF_DIST) {
              ++reached;
              mx = std::max(mx, x);
            }
          std::ostringstream s;
          s << "source=" << src << " weight_seed=" << o.seed
            << " reached=" << reached << " max_dist=" << mx;
          return {check_sssp(g, src, d), s.str()};
        });
  }
  if (want("tc")) {
    DistGraph dg;
    const double setup =
        timed_setup([&] { dg = distribute(root ? gs : nullptr, false); });
    run(
        "tc", gs, sym_s, setup,
        [&](mpi::Stats &st) { return mpi::triangle_count(dg, st); },
        [&](const TCResult &r) -> Pair {
          if (!root)
            return {};
          return {check_tc(*gs, r.triangles),
                  "triangles=" + std::to_string(r.triangles)};
        });
  }
  MPI_Finalize();
  return 0;
}