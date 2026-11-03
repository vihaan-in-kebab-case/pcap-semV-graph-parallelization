#!/usr/bin/env python3
"""One comparison table across scenarios, from evaluate.py outputs.
usage: summary_table.py LABEL=results/processed/X_selector_eval.csv [LABEL=...]
Use CSVs from plain runs (with selection overhead), so overhead is counted.
"""
import sys

import pandas as pd

POL = ["seq", "fixed", "rule", "tree", "oracle"]
NAMES = {"seq": "always seq", "fixed": "best fixed", "rule": "size rule", "tree": "tree", "oracle": "oracle"}
FLOOR = 1e-4

rows = []
for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    d = pd.read_csv(path)
    for split in ("graph", "family"):
        s = d[d.split == split]
        orc = s[s.policy == "oracle"].t_policy.sum()
        seq = s[s.policy == "seq"].t_policy.sum()
        for p in POL:
            x = s[s.policy == p]
            rows.append({"scenario": label, "split": "unseen " + split, "policy": NAMES[p],
                         "vs_oracle": x.t_policy.sum() / orc, "vs_sequential": x.t_policy.sum() / seq,
                         "slower_than_seq_%": 100 * (x.t_policy > x.t_seq + FLOOR).mean()})
df = pd.DataFrame(rows)
order = [NAMES[p] for p in POL]
for col, title in (("vs_oracle", "total time / oracle total (1.00 = perfect, within a scenario)"),
                   ("vs_sequential", "total time / always-sequential total (lower is better, comparable across scenarios)"),
                   ("slower_than_seq_%", "instances slower than sequential (%)")):
    print("\n" + title)
    print(df.pivot_table(index=["scenario", "split"], columns="policy", values=col)[order].round(2).to_string())