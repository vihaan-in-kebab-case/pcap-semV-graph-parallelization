#!/usr/bin/env python3
"""Replay the selector against recorded sweep data (no GPU, no benchmark runs).

usage: replay.py --train results/processed/sweep_5090_dataset.csv --list
       replay.py --train results/processed/sweep_5090_dataset.csv --graph rmat_n18_d8 --algo bfs
       replay.py --train sweep_5090_dataset.csv --queries sweep_5090_demo_dataset.csv --graph demo_rmat_n18_d8 --algo pagerank
The decision comes from a model that has NOT seen the query graph (held out when it is in the training
data). Every time shown is a recorded measurement from the sweep, not a live run.
"""
import argparse
import csv
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policies  # noqa: E402

COLS = ("load_s", "preprocess_s", "setup_s", "alloc_s", "h2d_s", "d2h_s", "compute_s", "comm_s", "e2e_s")


def fmt(x):
    return "-" if x is None else f"{x * 1e6:.0f} us" if x < 1e-3 else f"{x * 1e3:.2f} ms" if x < 1 else f"{x:.3f} s"


def tag_of(path):
    return os.path.basename(path).replace("_dataset.csv", "")


def recorded(raw_dir, tag, graph, algo, cfg):
    p = os.path.join(raw_dir, f"{tag}_{graph}.csv")
    if not os.path.exists(p):
        return None
    impl, w = cfg
    with open(p) as f:
        rows = [r for r in csv.DictReader(f)
                if r["algorithm"] == algo and r["impl"] == impl and int(float(r["threads"])) == w]
    if not rows:
        return None
    out = {}
    for c in COLS:
        v = [float(r[c]) for r in rows if r[c] != ""]
        out[c] = statistics.median(v) if v else None
    out["correct"] = all(r["correct"] == "1" for r in rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--queries")
    ap.add_argument("--graph")
    ap.add_argument("--algo")
    ap.add_argument("--policy", default="tree", choices=["tree", "knn", "rule", "fixed", "seq"])
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-gain", type=float, default=1.05)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-gpu", action="store_true", help="simulate a machine without a GPU")
    a = ap.parse_args()

    train = policies.load_train(a.train)
    queries = policies.load_train(a.queries) if a.queries else train
    if a.list:
        seen = {}
        for (g, algo), e in queries.items():
            seen.setdefault(g, [e["family"], round(10 ** e["vec"][0]), []])[2].append(algo)
        for g, (fam, n, algos) in sorted(seen.items(), key=lambda kv: kv[1][1]):
            print(f"{g:24s} {fam:11s} n={n:<9d} {','.join(sorted(algos))}")
        return
    key = (a.graph, a.algo)
    if key not in queries:
        sys.exit(f"no recorded data for {key}; use --list to see what exists")
    inst = queries[key]
    qtag, ttag = tag_of(a.queries or a.train), tag_of(a.train)
    cands = [c for c in inst["scores"]
             if c == policies.SEQ or (c[0] == "cuda" and not a.no_gpu) or (c[0] != "cuda" and c[1] > 1)]
    exclude = {a.graph} if (not a.queries or a.queries == a.train) else set()

    print("=== REPLAY: recorded measurements; nothing is executing on a GPU now ===")
    print(f"recorded sweep: {qtag}   model trained on: {ttag}"
          + ("   (query graph held out)" if exclude else "   (query graph never seen by the model)"))
    v = inst["vec"]
    print(f"graph: {a.graph} [{inst['family']}]   algorithm: {a.algo}")
    print(f"[features] n~{round(10 ** v[0]):,} mean_degree={10 ** v[1] - 1:.3g} degree_skew={10 ** v[2] - 1:.3g} "
          f"out_cv={v[3]:.3g} dangling={v[4]:.3g} symmetric={bool(v[5])}")
    print("[options]  " + ", ".join(policies.cfg_label(c) for c in cands))
    if a.no_gpu:
        print("[gpu]      GPU absent (simulated): CUDA excluded from the options; policy trained on CPU options only")

    kw = dict(k=a.k, min_gain=a.min_gain)
    policies.decide(a.policy, train, a.algo, v, cands, exclude, **kw)       # fit and cache (done offline)
    t0 = time.perf_counter()
    choice, scores, reason, info = policies.decide(a.policy, train, a.algo, v, cands, exclude, **kw)
    infer = time.perf_counter() - t0
    print(f"[decision] policy={info['policy_used']}  ->  {policies.cfg_label(choice)}   ({reason})")
    for line in info["explain"]:
        print("           " + line)
    if scores:
        print("           predicted speedup: " + ", ".join(
            f"{policies.cfg_label(c)} {math.exp(s):.2f}x"
            for c, s in sorted(scores.items(), key=lambda kv: -kv[1])[:4]))

    D = {c: inst["D"][c] for c in cands}
    ranked = sorted(D, key=D.get)
    seq = D.get(policies.SEQ)
    print("\n[measured] every option, recorded decision cost (setup + compute, excludes loading)")
    for i, c in enumerate(ranked, 1):
        tag = "  <- chosen" if c == choice else ("  <- best" if i == 1 else "")
        print(f"  {i:2d}. {policies.cfg_label(c):12s} {fmt(D[c]):>10s}   speedup vs sequential "
              f"{(seq / D[c]) if seq else float('nan'):.2f}x{tag}")
    if a.no_gpu:
        for c in inst["scores"]:
            if c[0] == "cuda":
                print(f"   -  {policies.cfg_label(c):12s} excluded (no GPU); recorded on the GPU machine: "
                      f"{fmt(inst['D'][c])}, {(seq / inst['D'][c]) if seq else float('nan'):.2f}x vs sequential")
    regret = D[choice] / D[ranked[0]] - 1
    print(f"regret vs best option: {100 * regret:.1f}%   (0% = picked the best)")

    r = recorded(a.raw, qtag, a.graph, a.algo, choice)
    overhead = inst["extract_s"] + infer
    print("\n[metrics] chosen option, recorded medians")
    if r:
        print(f"  correct                  {'yes' if r['correct'] else 'NO'}")
        print(f"  raw compute runtime      {fmt(r['compute_s'])}"
              + (f"   (MPI communication {fmt(r['comm_s'])})" if r.get("comm_s") else ""))
        print(f"  run end-to-end           {fmt(r['e2e_s'])}   = load {fmt(r['load_s'])} + preprocess "
              f"{fmt(r['preprocess_s'])} + setup/transfers + compute")
        print(f"  selection overhead       {fmt(overhead)}   (recorded feature extraction + measured inference)")
        print(f"  one-shot total           {fmt(r['e2e_s'] + overhead)}")
    else:
        print(f"  (raw rows not found under {a.raw}; showing decision cost only: {fmt(D[choice])})")
        print(f"  selection overhead       {fmt(overhead)}")


if __name__ == "__main__":
    main()