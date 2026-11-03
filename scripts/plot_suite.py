#!/usr/bin/env python3
"""Sweep-level figure suite for one scenario (one machine, optionally some implementations dropped).

usage: plot_suite.py --tag sweep_5090 --out results/figures/5090_gpu/suite [--drop-impl mpi] [--title LABEL]
Reads results/raw/<tag>_<graph>.csv, results/features/<tag>_features.jsonl and datasets/manifest.csv.
Compute time = algorithm only (CUDA: kernel events, no transfers). Decision cost D = e2e - load - preprocess.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

ALGOS = ["bfs", "pagerank", "cc", "sssp", "tc"]
ALGO_NAME = {"bfs": "BFS", "pagerank": "PageRank", "cc": "Connected components", "sssp": "SSSP",
             "tc": "Triangle counting"}
FW = ["seq", "omp", "mpi", "cuda"]
FWNAME = {"seq": "sequential", "omp": "OpenMP", "mpi": "MPI", "cuda": "CUDA"}
COL = {"seq": "#6b6b6b", "omp": "#1f77b4", "mpi": "#2ca02c", "cuda": "#d62728"}
FAMS = ["uniform", "rmat", "smallworld", "grid"]
MARK = {"uniform": "o", "rmat": "^", "smallworld": "s", "grid": "D", "social": "P", "web": "X"}
TIMEC = ["load_s", "preprocess_s", "alloc_s", "h2d_s", "d2h_s", "setup_s", "compute_s", "comm_s", "e2e_s"]
PARTS = [("load", "#9e9e9e"), ("preprocess", "#ff9f40"), ("alloc / setup", "#bcbd22"),
         ("host to device", "#9467bd"), ("compute", "#1f77b4"), ("MPI communication", "#2ca02c"),
         ("device to host", "#8c564b"), ("other", "#d9d9d9")]
NOTE = ("Compute = algorithm only (CUDA: kernel events, no transfers). Decision cost = end-to-end minus "
        "load and preprocess. Medians over trials.")


# ------------------------------------------------------------------ data
def load(a):
    manifest = pd.read_csv(a.manifest)
    by_file = dict(zip(manifest.file, manifest.name))
    fam = dict(zip(manifest.name, manifest.family))
    dmap = dict(zip(manifest.name, manifest.d))
    feats = {}
    with open(os.path.join(a.features, f"{a.tag}_features.jsonl")) as f:
        for line in f:
            if line.strip():
                j = json.loads(line)
                feats[j["graph"]] = j
    files = sorted(glob.glob(os.path.join(a.raw, f"{a.tag}_*.csv")))
    if not files:
        raise SystemExit(f"no files match {a.raw}/{a.tag}_*.csv")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df = df[df.dataset.isin(by_file)].copy()
    df["graph"] = df.dataset.map(by_file)
    df = df.rename(columns={"threads": "workers"})
    df = df[~df.impl.isin(a.drop_impl)]
    bad = int((df.correct != 1).sum())
    if bad:
        print(f"note: excluding {bad} incorrect rows")
    df = df[df.correct == 1]

    keys = ["graph", "algorithm", "impl", "workers"]
    g = df.groupby(keys)
    med = g[TIMEC].median().reset_index()
    q = g["compute_s"].quantile([0.25, 0.75]).unstack().reset_index()
    q.columns = keys + ["c_q1", "c_q3"]
    med = med.merge(q, on=keys).rename(columns={"algorithm": "algo"})
    for k in ("n", "m", "mean_degree", "degree_skew", "out_cv"):
        med[k] = med.graph.map(lambda x, k=k: feats.get(x, {}).get(k, np.nan))
    med["family"] = med.graph.map(fam)
    med["d"] = med.graph.map(dmap)
    for c in ("alloc_s", "h2d_s", "d2h_s", "setup_s", "comm_s"):
        med[c] = med[c].fillna(0.0)
    med["D"] = med.e2e_s - med.load_s - med.preprocess_s
    seq = med[med.impl == "seq"].set_index(["graph", "algo"])
    med["seq_compute"] = [seq.compute_s.get((g_, a_), np.nan) for g_, a_ in zip(med.graph, med.algo)]
    med["seq_D"] = [seq.D.get((g_, a_), np.nan) for g_, a_ in zip(med.graph, med.algo)]
    med["speedup"] = med.seq_compute / med.compute_s
    med["speedup_D"] = med.seq_D / med.D
    med["offered"] = med.impl.isin(["seq", "cuda"]) | (med.workers > 1)
    return med


def best(med, by):
    d = med[med.offered]
    idx = d.groupby(["graph", "algo", "impl"])[by].idxmin()
    return d.loc[idx.values].reset_index(drop=True)


def binned(s, x, y):
    t = s.dropna(subset=[x, y])
    t = t[(t[x] > 0) & (t[y] > 0)]
    if t.empty:
        return t
    return (t.assign(_b=np.round(np.log2(t[x]))).groupby("_b")
            .agg(xm=(x, "median"), ym=(y, "median")).sort_values("xm"))


def save(fig, out, name, title, note=NOTE):
    os.makedirs(out, exist_ok=True)
    fig.suptitle(title, y=1.0, fontsize=13)
    fig.text(0.01, -0.01, note, fontsize=7, color="#555")
    path = os.path.join(out, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def five(h=4.2):
    fig, axes = plt.subplots(1, 5, figsize=(21, h))
    return fig, axes


# ------------------------------------------------------------------ figures
def fig1(bc, out, title):
    """Compute time versus graph size, per algorithm, one line per framework."""
    fig, axes = five()
    for ax, a in zip(axes, ALGOS):
        for fw in FW:
            s = bc[(bc.algo == a) & (bc.impl == fw)].dropna(subset=["m", "compute_s"])
            if s.empty:
                continue
            ax.scatter(s.m, s.compute_s, s=12, alpha=0.3, color=COL[fw])
            b = binned(s, "m", "compute_s")
            if len(b):
                ax.plot(b.xm, b.ym, color=COL[fw], lw=1.8, label=FWNAME[fw])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3)
        ax.set_title(ALGO_NAME[a]); ax.set_xlabel("edges m")
    axes[0].set_ylabel("compute time (s), best config per framework")
    axes[0].legend(fontsize=8)
    save(fig, out, "01_runtime_vs_size.png", title)


def fig2(bc, out, title):
    """Framework performance for each algorithm within each graph family (main d=8 series)."""
    main = bc[(bc.family == "grid") | (bc.d == 8)]
    fig, axes = plt.subplots(5, 4, figsize=(16, 15), sharex="col")
    for i, a in enumerate(ALGOS):
        for j, f in enumerate(FAMS):
            ax = axes[i, j]
            for fw in FW:
                s = main[(main.algo == a) & (main.family == f) & (main.impl == fw)].sort_values("n")
                if len(s):
                    ax.plot(s.n, s.compute_s, marker="o", ms=3, color=COL[fw], label=FWNAME[fw])
            ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f)
            if j == 0:
                ax.set_ylabel(ALGO_NAME[a] + "\ncompute (s)")
            if i == 4:
                ax.set_xlabel("vertices n")
    axes[0, 0].legend(fontsize=7)
    save(fig, out, "02_framework_by_family.png", title)


def fig3(bc, out, title):
    """Speedup over sequential versus size."""
    fig, axes = five()
    for ax, a in zip(axes, ALGOS):
        for fw in ("omp", "mpi", "cuda"):
            s = bc[(bc.algo == a) & (bc.impl == fw)]
            b = binned(s, "m", "speedup")
            if len(b):
                ax.scatter(s.m, s.speedup, s=12, alpha=0.3, color=COL[fw])
                ax.plot(b.xm, b.ym, color=COL[fw], lw=1.8,
                        label=FWNAME[fw] + (" (kernel only)" if fw == "cuda" else ""))
            if fw == "cuda":
                b2 = binned(s, "m", "speedup_D")
                if len(b2):
                    ax.plot(b2.xm, b2.ym, color=COL[fw], lw=1.6, ls="--", label="CUDA incl. transfers")
        ax.axhline(1.0, color="#999", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3)
        ax.set_title(ALGO_NAME[a]); ax.set_xlabel("edges m")
    axes[0].set_ylabel("speedup over sequential")
    axes[0].legend(fontsize=8)
    save(fig, out, "03_speedup_vs_size.png", title)


def fig4(win, out, title):
    """Which option was fastest for every graph and algorithm (decision cost)."""
    rank = {"social": 0, "web": 1, "uniform": 2, "rmat": 3, "smallworld": 4, "grid": 5}
    gm = (win.drop_duplicates("graph").assign(_r=lambda d: d.family.map(rank).fillna(9))
          .sort_values(["_r", "n"]))
    graphs = gm.graph.tolist()
    code = {"seq": 0, "omp": 1, "mpi": 2, "cuda": 3}
    M = -np.ones((len(graphs), len(ALGOS)))
    T = [[""] * len(ALGOS) for _ in graphs]
    for r in win.itertuples():
        i, j = graphs.index(r.graph), ALGOS.index(r.algo)
        M[i, j] = code[r.impl]
        T[i][j] = r.impl if r.impl in ("seq", "cuda") else f"{r.impl}{int(r.workers)}"
    cmap = ListedColormap([COL[f] for f in FW])
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(graphs) + 2))
    ax.imshow(np.ma.masked_equal(M, -1), cmap=cmap, vmin=0, vmax=3, aspect="auto")
    for i in range(len(graphs)):
        for j in range(len(ALGOS)):
            if T[i][j]:
                ax.text(j, i, T[i][j], ha="center", va="center", fontsize=7, color="white")
    ax.set_xticks(range(len(ALGOS)))
    ax.set_xticklabels([ALGO_NAME[a] for a in ALGOS], rotation=20, ha="right")
    ax.set_yticks(range(len(graphs)))
    ax.set_yticklabels(graphs, fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[f]) for f in FW if (win.impl == f).any()]
    ax.legend(handles, [FWNAME[f] for f in FW if (win.impl == f).any()], loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=8)
    save(fig, out, "04_winner_map.png", title,
         "Fastest option by decision cost among options a selector can choose (OpenMP/MPI at 2+ workers).")


def fig5(med, out, title):
    """Scaling with workers for the largest graphs."""
    impls = [i for i in ("omp", "mpi") if (med.impl == i).any()]
    if not impls:
        return
    top = med.drop_duplicates("graph").nlargest(5, "m").graph.tolist()
    fig, axes = plt.subplots(len(impls), 5, figsize=(21, 3.8 * len(impls)), squeeze=False)
    for r, impl in enumerate(impls):
        ws = sorted(med[med.impl == impl].workers.unique())
        for ax, a in zip(axes[r], ALGOS):
            for g in top:
                s = med[(med.graph == g) & (med.algo == a) & (med.impl == impl)].sort_values("workers")
                if len(s):
                    ax.plot(s.workers, s.speedup, marker="o", ms=3, label=g)
            ax.plot(ws, ws, "--", color="#999", label="ideal")
            ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
            ax.set_xticks(ws); ax.set_xticklabels([str(int(w)) for w in ws]); ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(ALGO_NAME[a])
            ax.set_xlabel(("threads" if impl == "omp" else "ranks"))
        axes[r][0].set_ylabel(f"{FWNAME[impl]} speedup over sequential")
    axes[0][0].legend(fontsize=6)
    save(fig, out, "05_scaling_largest_graphs.png", title)


def parts(r):
    comp = r.compute_s - (r.comm_s if r.impl == "mpi" else 0.0)
    comm = r.comm_s if r.impl == "mpi" else 0.0
    v = [r.load_s, r.preprocess_s, r.alloc_s + r.setup_s, r.h2d_s, comp, comm, r.d2h_s]
    return v + [max(r.e2e_s - sum(v), 0.0)]


def fig6(med, graph, out, title):
    """Timing-region split for one graph: absolute and share of end-to-end, per framework."""
    sub = med[(med.graph == graph) & med.offered]
    if sub.empty:
        return
    bd = best(sub, "D")
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for j, a in enumerate(ALGOS):
        s = bd[bd.algo == a].copy()
        s["_o"] = s.impl.map({f: i for i, f in enumerate(FW)})
        s = s.sort_values("_o")
        if s.empty:
            continue
        labels = [r.impl if r.impl in ("seq", "cuda") else f"{r.impl}x{int(r.workers)}" for r in s.itertuples()]
        P = np.array([parts(r) for r in s.itertuples()])
        tot = P.sum(axis=1)
        tot[tot == 0] = 1.0
        b0, b1 = np.zeros(len(s)), np.zeros(len(s))
        for k, (_, color) in enumerate(PARTS):
            axes[0, j].bar(labels, P[:, k], bottom=b0, color=color)
            axes[1, j].bar(labels, 100 * P[:, k] / tot, bottom=b1, color=color)
            b0 += P[:, k]
            b1 += 100 * P[:, k] / tot
        axes[0, j].set_title(ALGO_NAME[a])
    axes[0, 0].set_ylabel("seconds")
    axes[1, 0].set_ylabel("% of end-to-end")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in PARTS]
    fig.legend(handles, [n for n, _ in PARTS], loc="lower center", ncol=8, fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    row = sub.iloc[0]
    save(fig, out, f"06_time_split_{graph}.png", f"{title}: {graph} (n={int(row.n)}, m={int(row.m)})",
         "Best option per framework by decision cost. MPI compute is split into compute and communication; "
         "'other' = frees, syncs and untimed work.")


def fig7(bc, out, title):
    """Share of end-to-end time that is compute, versus size."""
    fig, axes = five()
    d = bc.copy()
    d["share"] = d.compute_s / d.e2e_s
    for ax, a in zip(axes, ALGOS):
        for fw in FW:
            s = d[(d.algo == a) & (d.impl == fw)]
            b = binned(s, "m", "share")
            if len(b):
                ax.scatter(s.m, s.share, s=12, alpha=0.3, color=COL[fw])
                ax.plot(b.xm, b.ym, color=COL[fw], lw=1.8, label=FWNAME[fw])
        ax.set_xscale("log"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
        ax.set_title(ALGO_NAME[a]); ax.set_xlabel("edges m")
    axes[0].set_ylabel("compute / end-to-end")
    axes[0].legend(fontsize=8)
    save(fig, out, "07_compute_share_of_end_to_end.png", title)


def fig8(med, out, title):
    """Cost of moving data: CUDA alloc+transfers versus kernel, MPI distribution versus compute."""
    c = med[med.impl == "cuda"].copy()
    c["ratio"] = (c.alloc_s + c.h2d_s + c.d2h_s) / c.compute_s
    m_ = med[(med.impl == "mpi") & (med.workers > 1)]
    if len(m_):
        m_ = m_[m_.workers == m_.workers.max()].copy()
        m_["ratio"] = m_.setup_s / m_.compute_s
    panels = [("CUDA: (alloc + host-to-device + device-to-host) / kernel", c),
              (f"MPI at {int(m_.workers.max()) if len(m_) else 0} ranks: distribution / compute", m_)]
    panels = [p for p in panels if len(p[1])]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 4.4), squeeze=False)
    for ax, (name, d) in zip(axes[0], panels):
        for i, a in enumerate(ALGOS):
            b = binned(d[d.algo == a], "m", "ratio")
            if len(b):
                ax.plot(b.xm, b.ym, marker="o", ms=3, color=plt.cm.tab10.colors[i], label=ALGO_NAME[a])
        ax.axhline(1.0, color="#999", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3)
        ax.set_title(name, fontsize=10); ax.set_xlabel("edges m")
    axes[0][0].set_ylabel("overhead / compute (above 1 = data movement dominates)")
    axes[0][0].legend(fontsize=7)
    save(fig, out, "08_transfer_and_setup_cost.png", title)


def fig9(med, out, title):
    """Parallel efficiency versus workers; solid = large graphs (m >= 1M), dotted = small (m < 100k)."""
    impls = [i for i in ("omp", "mpi") if (med.impl == i).any()]
    if not impls:
        return
    fig, axes = plt.subplots(1, len(impls), figsize=(6.5 * len(impls), 4.4), squeeze=False)
    for ax, impl in zip(axes[0], impls):
        s = med[med.impl == impl].dropna(subset=["m", "speedup"]).copy()
        s["eff"] = s.speedup / s.workers
        for i, a in enumerate(ALGOS):
            for mask, ls in ((s.m >= 1e6, "-"), (s.m < 1e5, ":")):
                t = s[(s.algo == a) & mask].groupby("workers").eff.median()
                if len(t):
                    ax.plot(t.index, t.values, ls=ls, marker="o", ms=3, color=plt.cm.tab10.colors[i],
                            label=ALGO_NAME[a] if ls == "-" else None)
        ax.axhline(1.0, color="#999", ls="--", lw=0.8)
        ax.set_xscale("log", base=2); ax.grid(alpha=0.3)
        ax.set_title(FWNAME[impl]); ax.set_xlabel("threads" if impl == "omp" else "ranks")
    axes[0][0].set_ylabel("parallel efficiency (median over graphs)")
    axes[0][0].legend(fontsize=7)
    save(fig, out, "09_parallel_efficiency.png", title)


def fig10(bc, out, title):
    """Spread of speedups across graphs per framework."""
    fig, axes = five(4.6)
    for ax, a in zip(axes, ALGOS):
        data, names = [], []
        for fw in ("omp", "mpi", "cuda"):
            s = bc[(bc.algo == a) & (bc.impl == fw)].speedup.dropna()
            if len(s):
                data.append(s.values); names.append(FWNAME[fw] + ("\n(kernel)" if fw == "cuda" else ""))
            if fw == "cuda":
                s2 = bc[(bc.algo == a) & (bc.impl == "cuda")].speedup_D.dropna()
                if len(s2):
                    data.append(s2.values); names.append("CUDA\n+transfers")
        if not data:
            continue
        ax.boxplot(data)
        ax.set_xticklabels(names, fontsize=8)
        ax.axhline(1.0, color="#999", lw=0.8)
        ax.set_yscale("log"); ax.grid(axis="y", alpha=0.3); ax.set_title(ALGO_NAME[a])
    axes[0].set_ylabel("speedup over sequential, across graphs")
    save(fig, out, "10_speedup_distribution.png", title)


def fig11(win, out, title):
    """Graph size and skew, coloured by the fastest option: the decision regions a selector must learn."""
    fig, axes = five(4.6)
    for ax, a in zip(axes, ALGOS):
        s = win[win.algo == a]
        for (fw, fam_), g in s.groupby(["impl", "family"]):
            ax.scatter(g.n, g.degree_skew, c=COL[fw], marker=MARK.get(fam_, "o"), s=38,
                       edgecolors="k", linewidths=0.3)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3)
        ax.set_title(ALGO_NAME[a]); ax.set_xlabel("vertices n")
    axes[0].set_ylabel("degree skew (max / mean degree)")
    h1 = [plt.Line2D([], [], marker="o", ls="", color=COL[f]) for f in FW if (win.impl == f).any()]
    l1 = [FWNAME[f] for f in FW if (win.impl == f).any()]
    h2 = [plt.Line2D([], [], marker=MARK[f], ls="", color="#444") for f in MARK if (win.family == f).any()]
    l2 = [f for f in MARK if (win.family == f).any()]
    axes[-1].legend(h1 + h2, l1 + l2, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(fig, out, "11_winner_by_size_and_skew.png", title, "Colour = fastest option, marker = graph family.")


def fig12(med, out, title):
    """CUDA speedup versus degree skew (filled = kernel only, hollow = including transfers)."""
    c = med[med.impl == "cuda"].dropna(subset=["degree_skew"])
    if c.empty:
        return
    fams = sorted(c.family.dropna().unique())
    fc = dict(zip(fams, plt.cm.tab10.colors))
    fig, axes = five(4.6)
    for ax, a in zip(axes, ALGOS):
        for f, g in c[c.algo == a].groupby("family"):
            ax.scatter(g.degree_skew, g.speedup, color=fc[f], s=30, label=f)
            ax.scatter(g.degree_skew, g.speedup_D, facecolors="none", edgecolors=fc[f], s=30)
        ax.axhline(1.0, color="#999", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3)
        ax.set_title(ALGO_NAME[a]); ax.set_xlabel("degree skew")
    axes[0].set_ylabel("CUDA speedup over sequential")
    axes[0].legend(fontsize=7)
    save(fig, out, "12_cuda_speedup_vs_skew.png", title, "Filled = kernel only, hollow = including alloc and transfers.")


def fig13(med, out, title):
    """Which worker count wins, per algorithm."""
    impls = [i for i in ("omp", "mpi") if (med.impl == i).any()]
    if not impls:
        return
    fig, axes = plt.subplots(len(impls), 5, figsize=(21, 3.4 * len(impls)), squeeze=False)
    for r, impl in enumerate(impls):
        sub = med[(med.impl == impl) & (med.workers > 1)]
        b = best(sub, "D")
        for ax, a in zip(axes[r], ALGOS):
            cnt = b[b.algo == a].workers.value_counts().sort_index()
            ax.bar([str(int(w)) for w in cnt.index], cnt.values, color=COL[impl])
            if r == 0:
                ax.set_title(ALGO_NAME[a])
            ax.set_xlabel("threads" if impl == "omp" else "ranks")
        axes[r][0].set_ylabel(f"{FWNAME[impl]}: graphs where\nthis count was fastest")
    save(fig, out, "13_best_worker_counts.png", title,
         "Counts the best worker count among that framework's own options, by decision cost.")


def summary(bc, win, out, title):
    lines = [title, ""]
    w = win.groupby(["algo", "impl"]).size().unstack(fill_value=0).reindex(ALGOS)
    lines += ["Fastest option count by algorithm (decision cost, selector-offered options):", w.to_string(), ""]
    g = (bc[bc.impl != "seq"].dropna(subset=["speedup"]).groupby(["algo", "impl"]).speedup
         .apply(lambda x: float(np.exp(np.log(x).mean()))).unstack().reindex(ALGOS))
    lines += ["Geometric-mean compute speedup over sequential (best config per framework):", g.round(2).to_string(), ""]
    c = bc[bc.impl == "cuda"].dropna(subset=["speedup_D"])
    if len(c):
        gc = c.groupby("algo").speedup_D.apply(lambda x: float(np.exp(np.log(x).mean()))).reindex(ALGOS)
        lines += ["CUDA geometric-mean speedup including alloc and transfers (decision-cost basis):",
                  gc.round(2).to_string(), ""]
    f = (bc[bc.impl != "seq"].dropna(subset=["speedup"]).groupby(["algo", "impl"]).speedup
         .apply(lambda x: float((x > 1).mean())).unstack().reindex(ALGOS))
    lines += ["Share of graphs where the framework beat sequential (compute time):", f.round(2).to_string()]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def safe(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except Exception as e:  # keep going: one broken figure must not stop the rest
        print(f"[skip] {fn.__name__}: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--features", default="results/features")
    ap.add_argument("--manifest", default="datasets/manifest.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title")
    ap.add_argument("--drop-impl", action="append", default=[])
    ap.add_argument("--split-graphs", default="", help="comma list for the timing-split figures")
    a = ap.parse_args()

    med = load(a)
    title = a.title or a.tag
    if a.drop_impl:
        title += " [dropped: " + ", ".join(a.drop_impl) + "]"
    bc, bd = best(med, "compute_s"), best(med, "D")
    win = bd.loc[bd.groupby(["graph", "algo"])["D"].idxmin().values]
    print(f"{title}: {med.graph.nunique()} graphs, implementations: {sorted(med.impl.unique())}")

    for fn, args in [(fig1, (bc,)), (fig2, (bc,)), (fig3, (bc,)), (fig4, (win,)), (fig5, (med,)),
                     (fig7, (bc,)), (fig8, (med,)), (fig9, (med,)), (fig10, (bc,)), (fig11, (win,)),
                     (fig12, (med,)), (fig13, (med,))]:
        safe(fn, *args, out=a.out, title=title)

    names = [g for g in a.split_graphs.split(",") if g] or \
            [g for g in ("uniform_n12_d8", "uniform_n16_d8", "uniform_n20_d8") if g in set(med.graph)]
    if len(names) < 3:
        gm = med.drop_duplicates("graph").sort_values("m").dropna(subset=["m"])
        names = [gm.iloc[int(q * (len(gm) - 1))].graph for q in (0.2, 0.55, 0.95)]
    for g in names:
        safe(fig6, med, g, out=a.out, title=title)
    safe(summary, bc, win, a.out, title)


if __name__ == "__main__":
    main()