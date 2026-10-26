import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
bad = df[df.correct == 0]
if len(bad):
    print("INCORRECT RUNS:\n", bad[["algorithm", "impl", "dataset", "trial", "note"]], "\n")

def iqr(s):
    return s.quantile(0.75) - s.quantile(0.25)

g = df.groupby(["dataset", "algorithm", "impl", "threads"])
out = g.agg(trials=("trial", "count"),
            compute_med=("compute_s", "median"), compute_iqr=("compute_s", iqr),
            e2e_med=("e2e_s", "median"), e2e_iqr=("e2e_s", iqr)).reset_index()
print(out.to_string(index=False))
seq = out[out.impl == "seq"].set_index(["dataset", "algorithm"])[["compute_med", "e2e_med"]]
par = out[out.impl != "seq"].join(seq, on=["dataset", "algorithm"], rsuffix="_seq")
par["speedup_compute"] = par.compute_med_seq / par.compute_med
par["speedup_e2e"] = par.e2e_med_seq / par.e2e_med
par["efficiency"] = par.speedup_compute / par.threads
par.loc[par.impl == "cuda", "efficiency"] = float("nan")
print("\n", par[["dataset", "algorithm", "impl", "threads", "speedup_compute",
                 "speedup_e2e", "efficiency"]].to_string(index=False))