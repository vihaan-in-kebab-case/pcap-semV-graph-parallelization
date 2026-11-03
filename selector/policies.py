#!/usr/bin/env python3
"""Selection policies shared by gps.py and evaluate.py.

Training entries: (graph, algo) -> {"vec", "family", "scores": {cfg: ln(speedup vs seq)},
"D": {cfg: decision cost in seconds}, "extract_s"}.  cfg = (impl, workers).
"""
import csv
import math
import statistics

FEATURE_NAMES = ["log10_n", "log10_1p_mean_degree", "log10_1p_degree_skew", "out_cv",
                 "dangling_frac", "symmetric"]
SEQ = ("seq", 1)
_CACHE = {}

def cfg_label(cfg):
    impl, w = cfg
    return {"seq": "sequential", "cuda": "CUDA"}.get(impl, f"{'OpenMP' if impl == 'omp' else 'MPI'} x{w}")

def fvec(f):
    return [math.log10(max(float(f["n"]), 1.0)), math.log10(1 + float(f["mean_degree"])),
            math.log10(1 + float(f["degree_skew"])), float(f["out_cv"]),
            float(f["dangling_frac"]), 1.0 if float(f["symmetric"]) >= 0.5 else 0.0]

def load_train(path):
    train = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            e = train.setdefault((r["graph"], r["algo"]),
                                 {"vec": fvec(r), "family": r["family"], "scores": {}, "D": {},
                                  "extract_s": float(r["extract_s"])})
            cfg = (r["impl"], int(r["workers"]))
            e["scores"][cfg] = float(r["log_speedup"])
            e["D"][cfg] = float(r["D"])
    return train

def _pool(train, algo, exclude):
    return [(g, e) for (g, a), e in train.items() if a == algo and g not in exclude]

def _raw(j, x):   # feature index -> (readable name, raw value)
    if j == 0:
        return "n", 10 ** x
    if j == 1:
        return "mean_degree", 10 ** x - 1
    if j == 2:
        return "degree_skew", 10 ** x - 1
    return FEATURE_NAMES[j], x
    
def restrict_pool(pool, cands):
    """Keep only the options in `cands` in every training entry, so policies train on what exists."""
    allowed = set(cands)
    return [(g, {**e, "scores": {c: s for c, s in e["scores"].items() if c in allowed},
                 "D": {c: d for c, d in e["D"].items() if c in allowed}}) for g, e in pool]

def _sc(e, cfg):   # missing configuration -> pessimistic (worst measured for that graph)
    s = e["scores"]
    return s[cfg] if cfg in s else min(s.values())

def _means(pool, cfgs):
    out = {}
    for c in cfgs:
        vals = [e["scores"][c] for _, e in pool if c in e["scores"]]
        if vals:
            out[c] = statistics.fmean(vals)
    return out

# ---------------------------------------------------------------- rule
def rule_default(vec, cands):
    n = 10 ** vec[0]
    m = n * (10 ** vec[1] - 1)
    if m < 100_000:
        return SEQ, f"m ~ {m:,.0f} < 100,000 (uncalibrated default)"
    if ("cuda", 1) in cands and m >= 1_000_000:
        return ("cuda", 1), f"m ~ {m:,.0f} >= 1,000,000 and a GPU option exists (uncalibrated default)"
    omp = [c for c in cands if c[0] == "omp"]
    if omp:
        return max(omp, key=lambda c: c[1]), "large enough to parallelize: OpenMP on all cores (uncalibrated default)"
    return SEQ, "no parallel option available"

def _fit_threshold(pool):
    cfgs = sorted({c for _, e in pool for c in e["scores"] if c != SEQ})
    best = None
    for T in sorted({e["vec"][0] for _, e in pool}) + [float("inf")]:
        above = [(g, e) for g, e in pool if e["vec"][0] >= T]
        means = _means(above, cfgs)
        par = max(means, key=means.get) if means else None
        loss = 0.0
        for g, e in pool:
            chosen = SEQ if (e["vec"][0] < T or par is None) else par
            loss += max(e["scores"].values()) - _sc(e, chosen)
        if best is None or (loss, -T) < best[0]:
            best = ((loss, -T), T)
    return best[1]

def rule_calibrated(pool, vec, cands, key=None):
    """Size stump: n < T -> sequential, else the parallel config with the best mean speedup above T."""
    T = _CACHE.get(key) if key is not None else None
    if T is None:
        T = _fit_threshold(pool)
        if key is not None:
            _CACHE[key] = T
    cfgs = sorted({c for _, e in pool for c in e["scores"] if c != SEQ})
    thr = "inf" if T == float("inf") else f"{10 ** T:,.0f}"
    means = _means([(g, e) for g, e in pool if e["vec"][0] >= T], [c for c in cfgs if c in cands])
    expl = [f"calibrated on {len(pool)} graphs: n < {thr} -> sequential, otherwise the best parallel option"]
    if vec[0] < T or not means:
        return SEQ, f"n={10 ** vec[0]:,.0f} is below the size threshold {thr}", expl
    c = max(means, key=means.get)
    if means[c] <= 0:
        return SEQ, "no parallel option beat sequential on average", expl
    return c, f"n={10 ** vec[0]:,.0f} >= {thr}: best parallel option on average ({math.exp(means[c]):.2f}x)", expl

# ---------------------------------------------------------------- kNN
def knn_scores(pool, vec, cands, k):
    cols = list(zip(*[e["vec"] for _, e in pool]))
    mu = [statistics.fmean(c) for c in cols]
    sd = [statistics.pstdev(c) or 1.0 for c in cols]
    def z(v):
        return [(a - m) / s for a, m, s in zip(v, mu, sd)]

    q = z(vec)
    near = sorted(((math.dist(q, z(e["vec"])), g, e) for g, e in pool), key=lambda t: t[0])[:k]
    scores = {}
    for cfg in cands:
        if cfg == SEQ:
            scores[cfg] = 0.0
            continue
        num = den = 0.0
        for d, g, e in near:
            if cfg in e["scores"]:
                w = 1.0 / (d + 1e-6)
                num, den = num + w * e["scores"][cfg], den + w
        if den > 0:
            scores[cfg] = num / den
    expl = []
    for d, g, e in near:
        b = max(e["scores"], key=e["scores"].get)
        expl.append(f"neighbour {g} [{e['family']}] dist={d:.2f}: best was {cfg_label(b)} "
                    f"({math.exp(e['scores'][b]):.2f}x)")
    return scores, expl

# ---------------------------------------------------------------- decision tree (CART, multi-output)
def fit_tree(pool, depth=3, leaf=3):
    from sklearn.tree import DecisionTreeRegressor
    cfgs = sorted({c for _, e in pool for c in e["scores"]})
    X = [e["vec"] for _, e in pool]
    Y = [[_sc(e, c) for c in cfgs] for _, e in pool]
    model = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=max(1, min(leaf, len(pool) // 4)),
                                  random_state=0)
    model.fit(X, Y)
    return model, cfgs

def export_tree(model, cfgs):
    t = model.tree_
    return {"cfgs": [list(c) for c in cfgs], "left": t.children_left.tolist(),
            "right": t.children_right.tolist(), "feature": t.feature.tolist(),
            "threshold": t.threshold.tolist(), "value": [v.ravel().tolist() for v in t.value],
            "n_samples": t.n_node_samples.tolist()}


def tree_scores(pool, vec, cands, key=None):
    exported = _CACHE.get(key) if key is not None else None
    if exported is None:
        model, cfgs = fit_tree(pool)
        exported = export_tree(model, cfgs)
        if key is not None:
            _CACHE[key] = exported
    return tree_json_scores(exported, vec, cands)

def tree_to_text(model, cfgs):
    t = model.tree_
    lines = []
    def rec(nid, depth):
        pad = "    " * depth
        if t.children_left[nid] == t.children_right[nid]:
            vals = t.value[nid].ravel()
            j = max(range(len(cfgs)), key=lambda i: vals[i])
            lines.append(f"{pad}-> {cfg_label(cfgs[j])} (predicted {math.exp(vals[j]):.2f}x, "
                         f"{int(t.n_node_samples[nid])} graphs)")
            return
        name, thr = _raw(int(t.feature[nid]), float(t.threshold[nid]))
        lines.append(f"{pad}if {name} <= {thr:.4g}:")
        rec(t.children_left[nid], depth + 1)
        lines.append(f"{pad}else:")
        rec(t.children_right[nid], depth + 1)

    rec(0, 0)
    return "\n".join(lines)

def tree_json_scores(model, vec, cands):
    cfgs = [tuple(c) for c in model["cfgs"]]
    nid, path = 0, []
    while model["left"][nid] != model["right"][nid]:
        j, thr = model["feature"][nid], model["threshold"][nid]
        name, thr_raw = _raw(j, thr)
        _, val = _raw(j, vec[j])
        left = vec[j] <= thr
        path.append(f"rule: {name} = {val:.4g} {'<=' if left else '>'} {thr_raw:.4g}")
        nid = model["left"][nid] if left else model["right"][nid]
    scores = {c: v for c, v in zip(cfgs, model["value"][nid]) if c in cands}
    return scores, path

# ---------------------------------------------------------------- entry point
def _pick(scores, min_gain, info):
    if not scores:
        return SEQ, None, "no scored options", info
    best = max(scores, key=scores.get)
    reason = f"highest predicted speedup {math.exp(scores[best]):.2f}x among {len(scores)} options"
    if scores[best] < math.log(min_gain):
        best, reason = SEQ, f"no option predicted above {min_gain}x, so stay sequential"
    return best, scores, reason, info


def decide(name, train, algo, vec, cands, exclude, k=3, min_gain=1.05, model=None):
    """Policies only see, and choose among, the options in `cands`. With no CUDA option (no GPU) they are
    trained on CPU options alone. Returns (choice, scores or None, reason, info)."""
    info = {"policy_used": name, "explain": [], "gpu_absent": not any(c[0] == "cuda" for c in cands)}
    if name == "seq":
        return SEQ, None, "always sequential", info
    if name == "tree" and model is not None:
        m = model["models"].get(algo)
        if m is not None and "cfgs" not in m:
            m = m["cpu"] if info["gpu_absent"] else m["all"]
        if m is None:
            choice, reason = rule_default(vec, cands)
            info["policy_used"] = "rule (fallback: no trained tree for this algorithm)"
            return choice, None, reason, info
        scores, info["explain"] = tree_json_scores(m, vec, cands)
        return _pick(scores, min_gain, info)
    pool = restrict_pool(_pool(train, algo, exclude), cands)
    key = (name, algo, frozenset(exclude), frozenset(cands), id(train))
    if len(pool) < 5:
        choice, reason = rule_default(vec, cands)
        info["policy_used"] = "rule (fallback: fewer than 5 training graphs)"
        return choice, None, reason, info
    if name == "fixed":
        means = _means(pool, list(cands))
        c = max(means, key=means.get) if means else SEQ
        return c, means, f"best average speedup over {len(pool)} training graphs ({math.exp(means.get(c, 0.0)):.2f}x)", info
    if name == "rule":
        choice, reason, expl = rule_calibrated(pool, vec, cands, key)
        info["explain"] = expl
        return choice, None, reason, info
    if name == "knn":
        scores, info["explain"] = knn_scores(pool, vec, cands, k)
    elif name == "tree":
        scores, info["explain"] = tree_scores(pool, vec, cands, key)
    else:
        raise ValueError(f"unknown policy {name}")
    return _pick(scores, min_gain, info)