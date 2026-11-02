#!/usr/bin/env python3
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

ALGOS = ["bfs", "pagerank", "cc", "sssp", "tc"]
TIME_COLS = ["load_s", "preprocess_s", "alloc_s", "h2d_s", "d2h_s", "setup_s", "compute_s", "comm_s", "e2e_s"]
COLORS = {"seq": "#555555", "omp": "#1f77b4", "mpi": "#2ca02c", "cuda": "#d62728"}
LABEL = {"seq": "sequential", "omp": "OpenMP", "mpi": "MPI", "cuda": "CUDA"}
FOOT = ("compute = algorithm only (CUDA: kernel events, excludes transfers); "
        "end-to-end = load + preprocess + setup/transfers + compute.")


def nz(x):
    return 0.0 if pd.isna(x) else float(x)


def load(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    for c in TIME_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df.rename(columns={"threads": "workers"})


def summarize(df):
    keys = ["dataset", "algorithm", "impl", "workers"]
    agg = {c: "median" for c in TIME_COLS}
    agg["correct"] = "min"
    agg["trial"] = "count"
    med = df.groupby(keys).agg(agg).reset_index().rename(columns={"trial": "trials"})
    q = df.groupby(keys)["compute_s"].quantile([0.25, 0.75]).unstack().reset_index()
    q.columns = keys + ["compute_q1", "compute_q3"]
    med = med.merge(q, on=keys)
    med["compute_iqr"] = med.compute_q3 - med.compute_q1
    med["transfer_s"] = med.alloc_s.fillna(0) + med.h2d_s.fillna(0) + med.d2h_s.fillna(0)
    med["compute_incl_transfer_s"] = med.compute_s + med.transfer_s   # meaningful for CUDA only
    s = med[med.impl == "seq"][["dataset", "algorithm", "compute_s", "e2e_s"]].rename(
        columns={"compute_s": "seq_compute_s", "e2e_s": "seq_e2e_s"})
    out = med.merge(s, on=["dataset", "algorithm"], how="left")
    out["speedup_compute"] = out.seq_compute_s / out.compute_s
    out["speedup_e2e"] = out.seq_e2e_s / out.e2e_s
    out["speedup_incl_transfer"] = np.where(out.impl == "cuda", out.seq_compute_s / out.compute_incl_transfer_s, np.nan)
    out["efficiency"] = np.where(out.impl.isin(["omp", "mpi"]), out.speedup_compute / out.workers, np.nan)
    return out


def best_rows(d, impl):
    sub = d[(d.impl == impl) & (d.correct == 1)].dropna(subset=["compute_s"])
    if sub.empty:
        return {}
    idx = sub.groupby("algorithm")["compute_s"].idxmin()
    return {sub.loc[i, "algorithm"]: sub.loc[i] for i in idx}


def pick(d, best, kind, a):
    if kind in ("seq", "cuda"):
        r = d[(d.impl == kind) & (d.algorithm == a)]
        return r.iloc[0] if len(r) else None
    return best[kind].get(a)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig_speedup(d, ds, out):
    fig, axes = plt.subplots(1, len(ALGOS), figsize=(4 * len(ALGOS), 3.8))
    for ax, a in zip(axes, ALGOS):
        da = d[d.algorithm == a]
        for impl in ("omp", "mpi"):
            s = da[da.impl == impl].sort_values("workers")
            if len(s):
                ax.plot(s.workers, s.speedup_compute, marker="o", color=COLORS[impl], label=LABEL[impl])
        xs = sorted(da[da.impl.isin(["omp", "mpi"])].workers.unique())
        if xs:
            ax.plot(xs, xs, "--", color="#999999", label="ideal")
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs)
            ax.xaxis.set_major_formatter(ScalarFormatter())
        c = da[da.impl == "cuda"]
        if len(c):
            ax.axhline(c.speedup_compute.iloc[0], color=COLORS["cuda"], ls=":", label="CUDA kernel")
            if c.speedup_incl_transfer.notna().any():
                ax.axhline(c.speedup_incl_transfer.iloc[0], color=COLORS["cuda"], ls="-.", label="CUDA + transfers")
        ax.axhline(1.0, color=COLORS["seq"], lw=0.8)
        ax.set_yscale("log")
        ax.set_title(a)
        ax.set_xlabel("threads / ranks")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("speedup over sequential (compute)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{ds}: speedup vs sequential (1.0 = no gain)", y=1.03)
    fig.text(0.01, -0.04, FOOT, fontsize=7)
    save(fig, out, f"{ds}_1_speedup.png")


def fig_compare(d, ds, out):
    best = {i: best_rows(d, i) for i in ("omp", "mpi")}
    kinds = [("sequential", "seq", COLORS["seq"], "compute_s"),
             ("OpenMP (best)", "omp", COLORS["omp"], "compute_s"),
             ("MPI (best)", "mpi", COLORS["mpi"], "compute_s"),
             ("CUDA kernel", "cuda", COLORS["cuda"], "compute_s"),
             ("CUDA + transfers", "cuda", "#f4a6a6", "compute_incl_transfer_s")]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w, x = 0.16, np.arange(len(ALGOS))
    for k, (lab, kind, col, field) in enumerate(kinds):
        ys, notes = [], []
        for a in ALGOS:
            r = pick(d, best, kind, a)
            ys.append(np.nan if r is None else float(r[field]))
            notes.append("" if r is None or kind not in ("omp", "mpi") else f"@{int(r.workers)}")
        pos = x + (k - 2) * w
        ax.bar(pos, ys, w, label=lab, color=col)
        for px, y, nt in zip(pos, ys, notes):
            if nt and not np.isnan(y):
                ax.text(px, y, nt, ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ALGOS)
    ax.set_ylabel("median compute time (s, log scale)")
    ax.set_title(f"{ds}: compute time by implementation (best thread/rank count shown as @N)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.text(0.01, -0.03, FOOT, fontsize=7)
    save(fig, out, f"{ds}_2_compute_compare.png")


def fig_breakdown(d, ds, out):
    best = {i: best_rows(d, i) for i in ("omp", "mpi")}
    labels, parts = [], []
    for a in ALGOS:
        for kind in ("seq", "omp", "mpi", "cuda"):
            r = pick(d, best, kind, a)
            if r is None:
                continue
            load, pre = nz(r.load_s), nz(r.preprocess_s)
            move = nz(r.setup_s) + nz(r.transfer_s)
            comp = nz(r.compute_s)
            other = max(nz(r.e2e_s) - (load + pre + move + comp), 0.0)
            labels.append(f"{a}\n{kind}" + (f"@{int(r.workers)}" if kind in ("omp", "mpi") else ""))
            parts.append([load, pre, move, comp, other])
    if not parts:
        return
    P = np.array(parts)
    tot = P.sum(axis=1)
    tot[tot == 0] = 1.0
    names = ["load", "preprocess", "setup / transfers", "compute", "other"]
    cols = ["#9e9e9e", "#ff9f40", "#9467bd", "#1f77b4", "#d9d9d9"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(10, 0.5 * len(labels)), 8), sharex=True)
    xs = np.arange(len(labels))
    b1, b2 = np.zeros(len(labels)), np.zeros(len(labels))
    for j, (n, c) in enumerate(zip(names, cols)):
        a1.bar(xs, P[:, j], bottom=b1, color=c, label=n)
        a2.bar(xs, P[:, j] / tot * 100, bottom=b2, color=c)
        b1 += P[:, j]
        b2 += P[:, j] / tot * 100
    a1.set_ylabel("end-to-end time (s)")
    a2.set_ylabel("share of end-to-end (%)")
    a1.legend(fontsize=8, ncol=5)
    a2.set_xticks(xs)
    a2.set_xticklabels(labels, fontsize=7)
    a1.set_title(f"{ds}: where the end-to-end time goes (does compute even matter?)")
    fig.text(0.01, 0.0, FOOT, fontsize=7)
    save(fig, out, f"{ds}_3_e2e_breakdown.png")


def fig_efficiency(d, ds, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    drew = False
    for ax, impl in zip(axes, ("omp", "mpi")):
        for a in ALGOS:
            s = d[(d.impl == impl) & (d.algorithm == a)].sort_values("workers")
            if len(s):
                ax.plot(s.workers, s.efficiency, marker="o", label=a)
                drew = True
        ax.axhline(1.0, color="#999999", ls="--")
        ax.set_title(LABEL[impl])
        ax.set_xlabel("threads / ranks")
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("parallel efficiency (speedup / workers)")
    axes[0].legend(fontsize=7)
    if drew:
        save(fig, out, f"{ds}_4_efficiency.png")
    else:
        plt.close(fig)


def fig_mpi_comm(d, ds, out):
    m = d[d.impl == "mpi"]
    if m.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for a in ALGOS:
        s = m[m.algorithm == a].sort_values("workers")
        if len(s):
            ax.plot(s.workers, s.comm_s / s.compute_s * 100, marker="o", label=a)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("% of compute time spent in MPI calls")
    ax.set_title(f"{ds}: MPI communication share")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    save(fig, out, f"{ds}_5_mpi_comm.png")


def print_summary(d, ds):
    best = {i: best_rows(d, i) for i in ("omp", "mpi")}
    rows = []
    for a in ALGOS:
        for kind in ("omp", "mpi", "cuda"):
            r = pick(d, best, kind, a)
            if r is None:
                continue
            rows.append(dict(algorithm=a, impl=kind, workers=int(r.workers),
                             compute_s=r.compute_s, speedup_compute=r.speedup_compute,
                             speedup_incl_transfer=r.speedup_incl_transfer, speedup_e2e=r.speedup_e2e))
    print(f"\n=== {ds}: best configuration per algorithm ===")
    if rows:
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3g}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--processed", default="results/processed")
    args = ap.parse_args()

    df = load(args.csv)
    bad = df[df.correct == 0]
    if len(bad):
        print("WARNING: incorrect runs present:\n", bad[["algorithm", "impl", "dataset", "workers", "note"]].to_string(index=False))
    med = summarize(df)
    for ds in sorted(med.dataset.unique()):
        d = med[med.dataset == ds]
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", ds)
        print_summary(d, ds)
        fig_speedup(d, name, args.out)
        fig_compare(d, name, args.out)
        fig_breakdown(d, name, args.out)
        fig_efficiency(d, name, args.out)
        fig_mpi_comm(d, name, args.out)
    os.makedirs(args.processed, exist_ok=True)
    path = os.path.join(args.processed, os.path.basename(os.path.normpath(args.out)) + "_summary.csv")
    med.to_csv(path, index=False)
    print("wrote", path)


if __name__ == "__main__":
    main()