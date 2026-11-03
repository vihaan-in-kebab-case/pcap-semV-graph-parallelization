#!/usr/bin/env python3
"""Builds the selector training table from sweep results.

usage: build_dataset.py --tag sweep_mac [--raw results/raw] [--features results/features]
                        [--manifest datasets/manifest.csv] [--out results/processed]
Decision cost D = e2e_s - load_s - preprocess_s (median over trials) per (graph, algo, impl, workers).
log_speedup = ln(D_seq / D). Configs with any incorrect trial are dropped.
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics
from collections import Counter, defaultdict

FEATS = ["n", "m", "mean_degree", "degree_skew", "out_cv", "dangling_frac", "symmetric", "extract_s"]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--features", default="results/features")
    ap.add_argument("--manifest", default="datasets/manifest.csv")
    ap.add_argument("--out", default="results/processed")
    ap.add_argument("--drop-impl", action="append", default=[],
                    help="ignore rows of this implementation (e.g. mpi when its sweep is incomplete)")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = list(csv.DictReader(f))
    by_file = {r["file"]: r for r in manifest}

    feats = {}
    with open(os.path.join(args.features, f"{args.tag}_features.jsonl")) as f:
        for line in f:
            if line.strip():
                j = json.loads(line)
                feats[j["graph"]] = j

    files = sorted(glob.glob(os.path.join(args.raw, f"{args.tag}_*.csv")))
    if not files:
        raise SystemExit(f"no files match {args.raw}/{args.tag}_*.csv")
    cells = defaultdict(list)
    for p in files:
        with open(p) as f:
            for r in csv.DictReader(f):
                if r["impl"] in args.drop_impl:
                    continue
                m = by_file.get(r["dataset"])
                if m is None:
                    continue
                d = num(r["e2e_s"]) - num(r["load_s"]) - num(r["preprocess_s"])
                cells[(m["name"], r["algorithm"], r["impl"], int(num(r["threads"])))].append((d, r["correct"] == "1"))

    D, dropped = {}, 0
    for key, vals in cells.items():
        if not all(ok for _, ok in vals):
            dropped += 1
            continue
        D[key] = max(statistics.median(d for d, _ in vals), 1e-9)

    family = {r["name"]: r["family"] for r in manifest}
    rows = []
    for (g, algo, impl, w), d in sorted(D.items()):
        base = D.get((g, algo, "seq", 1))
        if base is None or g not in feats:
            continue
        row = {"graph": g, "family": family[g], "algo": algo, "impl": impl, "workers": w,
               "D": d, "log_speedup": math.log(base / d)}
        row.update({k: feats[g][k] for k in FEATS})
        rows.append(row)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.tag}_dataset.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    best = {}
    for r in rows:
        k = (r["graph"], r["algo"])
        if k not in best or r["D"] < best[k]["D"]:
            best[k] = r
    print(f"wrote {path}: {len(rows)} rows, {len({r['graph'] for r in rows})} graphs, "
          f"{len(best)} instances, {dropped} configs dropped (incorrect)")
    print("\noracle choice (best configuration) by algorithm:")
    for algo in ["bfs", "pagerank", "cc", "sssp", "tc"]:
        c = Counter(r["impl"] for (g, a), r in best.items() if a == algo)
        print(f"  {algo:9s} {dict(c)}")


if __name__ == "__main__":
    main()