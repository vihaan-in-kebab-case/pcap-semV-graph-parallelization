#!/usr/bin/env python3
"""Exports decision trees per algorithm as plain JSON: one over all offered options ("all") and one over
CPU options only ("cpu", used when no GPU is available).
usage: train.py --dataset results/processed/sweep_mac_dataset.csv [--out results/models/<tag>_tree.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policies  # noqa: E402

ALGOS = ["bfs", "pagerank", "cc", "sssp", "tc"]


def export(model, cfgs):
    t = model.tree_
    return {"cfgs": [list(c) for c in cfgs], "left": t.children_left.tolist(),
            "right": t.children_right.tolist(), "feature": t.feature.tolist(),
            "threshold": t.threshold.tolist(), "value": [v.ravel().tolist() for v in t.value],
            "n_samples": t.n_node_samples.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    tag = os.path.basename(args.dataset).replace("_dataset.csv", "")
    out = args.out or os.path.join("results", "models", f"{tag}_tree.json")
    train = policies.load_train(args.dataset)
    models, has_cuda = {}, False
    for algo in ALGOS:
        pool = [(g, e) for (g, a), e in train.items() if a == algo]
        if len(pool) < 5:
            continue
        offered = {c for _, e in pool for c in e["scores"] if c == policies.SEQ or c[0] == "cuda" or c[1] > 1}
        has_cuda = has_cuda or any(c[0] == "cuda" for c in offered)
        variants = {"all": offered, "cpu": {c for c in offered if c[0] != "cuda"}}
        models[algo] = {}
        for label, allowed in variants.items():
            model, cfgs = policies.fit_tree(policies.restrict_pool(pool, allowed))
            models[algo][label] = export(model, cfgs)
            print(f"\n--- {algo} [{label}]: tree fit on {len(pool)} graphs ---")
            print(policies.tree_to_text(model, cfgs))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"dataset": args.dataset, "features": policies.FEATURE_NAMES, "has_cuda": has_cuda,
                   "models": models}, f)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()