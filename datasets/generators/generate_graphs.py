#!/usr/bin/env python3
"""usage: generate_graphs.py [--preset demo|small] [--out datasets/graphs]
                          [--manifest datasets/manifest.csv] [--seed 12345] [--force]

Families: uniform (random directed), rmat (power-law directed), smallworld (undirected ring + rewiring),
grid (undirected 2D lattice, high diameter). 'd' = target mean degree of the stored CSR.
Writes SNAP-style edge lists with a '# Directed/Undirected graph' header, and a manifest
(name,family,source,directed,file,n,d,seed) used by the benchmark scripts and the model.
"""
import argparse
import csv
import os
import zlib

import numpy as np


def gen_uniform(n, d, rng):
    m = n * d
    return np.stack([rng.integers(0, n, m), rng.integers(0, n, m)], axis=1), True


def gen_rmat(n, d, rng, a=0.57, b=0.19, c=0.19):
    scale = int(np.log2(n))
    assert 2 ** scale == n, "rmat needs n a power of two"
    m = n * d
    u = np.zeros(m, dtype=np.int64)
    v = np.zeros(m, dtype=np.int64)
    for _ in range(scale):
        r = rng.random(m)
        ub = (r >= a + b).astype(np.int64)                                # lower half
        vb = (((r >= a) & (r < a + b)) | (r >= a + b + c)).astype(np.int64)  # right half
        u = (u << 1) | ub
        v = (v << 1) | vb
    perm = rng.permutation(n)                                             # remove id/degree correlation
    return np.stack([perm[u], perm[v]], axis=1), True


def gen_smallworld(n, d, rng, p=0.1):
    k = max(1, d // 2)
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = (src + np.tile(np.arange(1, k + 1, dtype=np.int64), n)) % n
    rew = rng.random(src.size) < p
    dst[rew] = rng.integers(0, n, int(rew.sum()))
    return np.stack([src, dst], axis=1), False


def gen_grid(n, rng):
    side = int(np.sqrt(n))
    idx = np.arange(side * side, dtype=np.int64).reshape(side, side)
    right = np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1)
    down = np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1)
    return np.concatenate([right, down]), False


def preset_specs(name):
    if name == "small":
        sizes, extra = [10, 12, 14], []
    elif name == "demo":
        sizes = [10, 12, 14, 16, 18, 20]
        extra = [(f, s, d) for f in ("uniform", "rmat") for s in (14, 16) for d in (2, 32)]
    else:
        raise SystemExit(f"unknown preset {name}")
    specs = [(f, s, 8) for f in ("uniform", "rmat", "smallworld") for s in sizes]
    specs += [("grid", s, 0) for s in sizes]
    specs += extra
    return sorted(specs, key=lambda t: (t[1], t[0], t[2]))   # small graphs first


def make(family, log2n, d, seed):
    n = 2 ** log2n
    name = f"{family}_n{log2n}" + (f"_d{d}" if family != "grid" else "")
    rng = np.random.default_rng(zlib.crc32(name.encode()) ^ seed)
    if family == "uniform":
        edges, directed = gen_uniform(n, d, rng)
    elif family == "rmat":
        edges, directed = gen_rmat(n, d, rng)
    elif family == "smallworld":
        edges, directed = gen_smallworld(n, d, rng)
    else:
        edges, directed = gen_grid(n, rng)
    return name, n, edges, directed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="demo")
    ap.add_argument("--out", default="datasets/graphs")
    ap.add_argument("--manifest", default="datasets/manifest.csv")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--spec", action="append", help="family:log2n:d, repeatable; overrides --preset")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--no-real", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = [] if args.no_real else [
        ["facebook", "social", "snap", 0, "facebook_combined.txt", 4039, "", ""],
        ["web-google", "web", "snap", 1, "web-Google.txt", 875713, "", ""],
    ]
    specs = ([(s.split(":")[0], int(s.split(":")[1]), int(s.split(":")[2])) for s in args.spec]
             if args.spec else preset_specs(args.preset))
    for family, log2n, d in specs:
        name, n, edges, directed = make(family, log2n, d, args.seed)
        name = args.prefix + name
        fname = f"{name}.txt"
        path = os.path.join(args.out, fname)
        rows.append([name, family, "synthetic", 1 if directed else 0, fname, n, d if d else "", args.seed])
        if os.path.exists(path) and not args.force:
            print(f"[skip] {fname}")
            continue
        with open(path, "w") as f:
            f.write(f"# {'Directed' if directed else 'Undirected'} graph (synthetic {family} n={n} d={d} seed={args.seed})\n")
            np.savetxt(f, edges, fmt="%d")
        print(f"[ok  ] {fname}: n={n} stored_edges={len(edges)}")

    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    with open(args.manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "family", "source", "directed", "file", "n", "d", "seed"])
        w.writerows(rows)
    print(f"wrote {args.manifest} ({len(rows)} graphs)")


if __name__ == "__main__":
    main()