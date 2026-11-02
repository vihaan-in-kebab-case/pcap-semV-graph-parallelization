#!/usr/bin/env python3
"""Leave-one-graph-out (LOGO) and leave-one-family-out (LOFO) evaluation of the selection policies.

usage: evaluate.py --dataset results/processed/sweep_mac_dataset.csv [--k 3] [--min-gain 1.05] [--show-tree]
Policy cost = decision cost of the chosen config (+ feature extraction + inference for rule/knn/tree).
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

ORDER = ["seq", "fixed", "rule", "knn", "tree", "oracle"]
LABEL = {"seq": "always sequential", "fixed": "best single config", "rule": "size-threshold rule",
         "knn": "kNN", "tree": "decision tree", "oracle": "oracle"}
ALGOS = ["bfs", "pagerank", "cc", "sssp", "tc"]
FLOOR = 1e-4


def run_split(train, split, args):
    family = {g: e["family"] for (g, _), e in train.items()}
    rows = []
    for (g, algo), inst in sorted(train.items()):
        if policies.SEQ not in inst["D"]:
            continue
        exclude = {g} if split == "graph" else {h for h, f in family.items() if f == family[g]}
        cands = [c for c in inst["scores"] if c == policies.SEQ or c[0] == "cuda" or c[1] > 1]
        best = max(cands, key=inst["scores"].get)
        d_seq, d_best = inst["D"][policies.SEQ], inst["D"][best]
        for name in ORDER:
            overhead = 0.0
            if name == "oracle":
                choice = best
            else:
                kw = dict(k=args.k, min_gain=args.min_gain)
                policies.decide(name, train, algo, inst["vec"], cands, exclude, **kw)   # fit + cache (offline)
                t0 = time.perf_counter()
                choice, _, _, _ = policies.decide(name, train, algo, inst["vec"], cands, exclude, **kw)
                infer = time.perf_counter() - t0
                if name in ("rule", "knn", "tree") and not args.no_overhead:
                    overhead = inst["extract_s"] + infer
            rows.append({"split": split, "graph": g, "family": family[g], "algo": algo, "policy": name,
                         "choice": policies.cfg_label(choice), "D_chosen": inst["D"][choice],
                         "overhead_s": overhead, "t_policy": inst["D"][choice] + overhead,
                         "t_oracle": d_best, "t_seq": d_seq})
    return rows


def summarize(rows):
    out = []
    for name in ORDER:
        rs = [r for r in rows if r["policy"] == name]
        if not rs:
            continue
        slow = sorted(max(r["t_policy"], FLOOR) / max(r["t_oracle"], FLOOR) for r in rs)
        sp = [max(r["t_seq"], FLOOR) / max(r["t_policy"], FLOOR) for r in rs]
        out.append({
            "policy": name, "n": len(rs),
            "geomean_speedup": math.exp(statistics.fmean(math.log(x) for x in sp)),
            "median_regret_pct": 100 * (statistics.median(slow) - 1),
            "p90_regret_pct": 100 * (slow[int(0.9 * (len(slow) - 1))] - 1),
            "max_regret_pct": 100 * (slow[-1] - 1),
            "total_ratio": sum(r["t_policy"] for r in rs) / sum(r["t_oracle"] for r in rs),
            "regression_rate_pct": 100 * sum(1 for r in rs if r["t_policy"] > r["t_seq"] + FLOOR) / len(rs),
            "top1_pct": 100 * sum(1 for r in rs if r["D_chosen"] <= max(1.05 * r["t_oracle"], r["t_oracle"] + FLOOR)) / len(rs),
            "overhead_pct": 100 * statistics.fmean(r["overhead_s"] / r["t_policy"] for r in rs)})
    return out


def print_table(title, summ):
    print(f"\n{title}")
    print(f"{'policy':22s}{'geomean speedup':>16s}{'median regret':>15s}{'p90 regret':>12s}"
          f"{'max regret':>12s}{'total time':>12s}{'regression':>12s}{'top-1':>8s}{'overhead':>10s}")
    for s in summ:
        print(f"{LABEL[s['policy']]:22s}{s['geomean_speedup']:>15.2f}x{s['median_regret_pct']:>14.1f}%"
              f"{s['p90_regret_pct']:>11.1f}%{s['max_regret_pct']:>11.1f}%{s['total_ratio']:>11.2f}x"
              f"{s['regression_rate_pct']:>11.1f}%{s['top1_pct']:>7.1f}%{s['overhead_pct']:>9.2f}%")


def figures(by_split, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.38
    for i, (split, label) in enumerate([("graph", "leave-one-graph-out"), ("family", "leave-one-family-out")]):
        summ = {s["policy"]: s for s in summarize(by_split[split])}
        ax.bar(np.arange(len(ORDER)) + (i - 0.5) * w, [summ[n]["geomean_speedup"] for n in ORDER], w, label=label)
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LABEL[n] for n in ORDER], rotation=20, ha="right")
    ax.set_ylabel("geometric-mean speedup vs sequential\n(selection overhead included)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(out_dir, "1_speedup_by_policy.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    names = ORDER[:-1]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    data = [[r["t_policy"] / r["t_oracle"] for r in by_split["graph"] if r["policy"] == n] for n in names]
    ax.boxplot(data)
    ax.set_xticklabels([LABEL[n] for n in names], rotation=20, ha="right")
    ax.set_yscale("log")
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_ylabel("time vs oracle (1.0 = perfect), leave-one-graph-out")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(out_dir, "2_slowdown_vs_oracle.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    pol = ["seq", "fixed", "rule", "knn", "tree"]
    w = 0.16
    for i, p in enumerate(pol):
        vals = []
        for a in ALGOS:
            xs = [r["t_policy"] / r["t_oracle"] for r in by_split["graph"] if r["policy"] == p and r["algo"] == a]
            vals.append(math.exp(statistics.fmean(math.log(x) for x in xs)) if xs else float("nan"))
        ax.bar(np.arange(len(ALGOS)) + (i - 2) * w, vals, w, label=LABEL[p])
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xticks(range(len(ALGOS)))
    ax.set_xticklabels(ALGOS)
    ax.set_yscale("log")
    ax.set_ylabel("geometric-mean time vs oracle (lower is better)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(out_dir, "3_slowdown_by_algorithm.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures to {out_dir}")


def show_trees(train):
    for algo in ALGOS:
        pool = [(g, e) for (g, a), e in train.items() if a == algo]
        if len(pool) < 5:
            continue
        model, cfgs = policies.fit_tree(pool)
        print(f"\n--- decision tree for {algo} (fit on all {len(pool)} graphs) ---")
        print(policies.tree_to_text(model, cfgs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-gain", type=float, default=1.05)
    ap.add_argument("--out", default="results/processed")
    ap.add_argument("--figs", default="results/figures")
    ap.add_argument("--show-tree", action="store_true")
    ap.add_argument("--no-overhead", action="store_true")
    args = ap.parse_args()

    train = policies.load_train(args.dataset)
    tag = os.path.basename(args.dataset).replace("_dataset.csv", "")
    by_split = {s: run_split(train, s, args) for s in ("graph", "family")}
    for s, title in (("graph", "Leave-one-graph-out"), ("family", "Leave-one-family-out")):
        n_inst = len({(r["graph"], r["algo"]) for r in by_split[s]})
        print_table(f"{title} ({n_inst} instances)", summarize(by_split[s]))
    print("\nregret = time / oracle time - 1; regression = slower than sequential; "
          "top-1 = within 5% of the best option; overhead = selection time / total.")

    os.makedirs(args.out, exist_ok=True)
    allrows = by_split["graph"] + by_split["family"]
    path = os.path.join(args.out, f"{tag}_selector_eval.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys()))
        w.writeheader()
        w.writerows(allrows)
    print(f"wrote {path}")
    figures(by_split, os.path.join(args.figs, f"selector_{tag}"))
    if args.show_tree:
        show_trees(train)


if __name__ == "__main__":
    main()