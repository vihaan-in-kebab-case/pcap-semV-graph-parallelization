#!/usr/bin/env python3
"""Graph analytics selector: ingest -> machine profile -> features -> decide -> run -> trace.

usage: gps.py --graph FILE --algo bfs|pagerank|cc|sssp|tc [--policy knn|rule] [--train DATASET.csv]
              [--verify] [--directed|--undirected] [--source V] [--seed S] [--trials N] [--warmup N]
              [--k 3] [--min-gain 1.05] [--keep-self] [--device 0]
By default the query graph is excluded from the training data (hold-out); use --keep-self to disable.
"""
import argparse
import csv
import glob
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import policies

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("BUILD_DIR", ROOT / "build"))
MANIFEST = ROOT / "datasets" / "manifest.csv"
ALGOS = ["bfs", "pagerank", "cc", "sssp", "tc"]

def say(s=""):
    print(s, flush=True)


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def fmt(x):
    return "-" if x is None else f"{x:.4g} s"


def cfg_name(cfg):
    impl, w = cfg
    return {"seq": "sequential", "cuda": "CUDA"}.get(impl, f"{'OpenMP' if impl == 'omp' else 'MPI'} x{w}")


# ---------------------------------------------------------------- machine

def sweep_cores():
    if sys.platform == "darwin":
        return int(sh(["sysctl", "-n", "hw.physicalcpu"]))
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def worker_counts(maxc):
    ws, c = [], 1
    while c <= maxc:
        ws.append(c)
        c *= 2
    return ws


def detect_machine():
    cpu = platform.processor() or platform.machine()
    try:
        if sys.platform == "darwin":
            cpu = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        else:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    bins = {n: (BUILD / n).exists() for n in ("gps_bench", "gps_bench_mpi", "gps_bench_cuda")}
    gpu = None
    if shutil.which("nvidia-smi"):
        try:
            line = sh(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
                       "--format=csv,noheader,nounits"]).splitlines()[0]
            name, mem, cc, drv = [x.strip() for x in line.split(",")]
            gpu = {"name": name, "mem_mib": int(float(mem)), "compute_cap": cc, "driver": drv}
        except Exception:
            gpu = None
    return {"os": f"{platform.system()} {platform.release()}", "arch": platform.machine(), "cpu": cpu,
            "cores": sweep_cores(), "logical": os.cpu_count(), "gpu": gpu, "bins": bins,
            "mpi": bool(shutil.which("mpirun")) and bins["gps_bench_mpi"],
            "gpu_usable": gpu is not None and bins["gps_bench_cuda"]}


# ---------------------------------------------------------------- ingest

def read_manifest():
    if not MANIFEST.exists():
        return {}
    with open(MANIFEST) as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def sniff(path, force_directed):
    p = Path(path)
    info = {"path": str(p), "format": "edge-list", "directed": True,
            "directed_source": "default (assumed directed)", "converted_path": None}
    with open(p, "r", errors="replace") as f:
        head = [f.readline() for _ in range(60)]
    if head and head[0].startswith("%%MatrixMarket"):
        info["format"] = "matrix-market"
        info["directed"] = "symmetric" not in head[0].lower()
        info["directed_source"] = "MatrixMarket header"
    else:
        text = " ".join(l for l in head if l.startswith(("#", "%")))
        if re.search(r"undirected", text, re.I):
            info["directed"], info["directed_source"] = False, "file header"
        elif re.search(r"directed", text, re.I):
            info["directed"], info["directed_source"] = True, "file header"
    entry = read_manifest().get(p.name)
    if entry is not None:
        info["directed"], info["directed_source"] = entry["directed"] == "1", "datasets/manifest.csv"
        info["name"] = entry["name"]
    if force_directed is not None:
        info["directed"], info["directed_source"] = force_directed, "command-line flag"
    data_line = next((l for l in head if l.strip() and not l.startswith(("#", "%"))), "")
    if info["format"] == "edge-list" and re.search(r"[,;]", data_line):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        with open(p, errors="replace") as src, tmp:
            for line in src:
                tmp.write(line if line.startswith(("#", "%")) else re.sub(r"[,;]", " ", line))
        info["converted_path"] = tmp.name
        info["format"] += " (delimiters normalized)"
    return info


def get_features(gpath, flags):
    t0 = time.perf_counter()
    r = subprocess.run([str(BUILD / "gps_bench"), "--graph", gpath, *flags, "--features"],
                       capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        sys.exit(f"feature extraction failed:\n{r.stderr or r.stdout}")
    for line in r.stdout.splitlines():
        if line.startswith("FEATURES_JSON "):
            return json.loads(line[len("FEATURES_JSON "):]), wall
    sys.exit("no features returned (is gps_bench rebuilt with --features support?)")


# ---------------------------------------------------------------- candidates and policies

def gpu_fits(f, algo, gpu):
    n, m = f["n"], f["m"]
    need = 8 * (n + 1) + 4 * m
    if algo == "pagerank":
        need *= 2
    if algo == "sssp":
        need += 4 * m
    need += 64 * n
    return need < 0.8 * gpu["mem_mib"] * 1024 * 1024, need


def candidates(machine, algo, f):
    cands, excluded = [("seq", 1)], []
    ws = [w for w in worker_counts(machine["cores"]) if w >= 2]
    if machine["bins"]["gps_bench"]:
        cands += [("omp", w) for w in ws]
    if machine["mpi"]:
        cands += [("mpi", w) for w in ws]
    else:
        excluded.append("MPI: mpirun or gps_bench_mpi not available")
    if machine["gpu_usable"]:
        ok, need = gpu_fits(f, algo, machine["gpu"])
        if ok:
            cands.append(("cuda", 1))
        else:
            excluded.append(f"CUDA: estimated {need / 2**20:.0f} MiB exceeds 80% of GPU memory")
    else:
        excluded.append("CUDA: no usable GPU or gps_bench_cuda not built")
    return cands, excluded

# ---------------------------------------------------------------- execution

def mpi_extra(np_):
    r = subprocess.run(["mpirun", "--version"], capture_output=True, text=True)
    flags = []
    if re.search(r"open.?mpi", (r.stdout + r.stderr).lower()):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            flags.append("--allow-run-as-root")
        if np_ > sweep_cores():
            flags.append("--oversubscribe")
    return flags


def run_config(cfg, algo, gpath, flags, a, tmp_dir):
    impl, w = cfg
    csv_path = os.path.join(tmp_dir, f"{impl}{w}.csv")
    common = ["--graph", gpath, *flags, "--algo", algo, "--trials", str(a.trials),
              "--warmup", str(a.warmup), "--out", csv_path, "--seed", str(a.seed)]
    if a.source is not None:
        common += ["--source", str(a.source)]
    env = os.environ.copy()
    if sys.platform == "darwin":
        env["TMPDIR"] = "/tmp"
    if impl == "seq":
        cmd = [str(BUILD / "gps_bench"), *common, "--impl", "seq"]
    elif impl == "omp":
        cmd = [str(BUILD / "gps_bench"), *common, "--impl", "omp", "--threads", str(w)]
        if sys.platform.startswith("linux"):
            env.update(OMP_PROC_BIND="close", OMP_PLACES="cores")
    elif impl == "mpi":
        cmd = ["mpirun", "-np", str(w), *mpi_extra(w), str(BUILD / "gps_bench_mpi"), *common]
    else:
        cmd = [str(BUILD / "gps_bench_cuda"), *common, "--device", str(a.device)]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t0
    if r.returncode != 0 or not os.path.exists(csv_path):
        return {"ok": False, "error": (r.stderr or r.stdout).strip()[-500:], "process_wall_s": wall}
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    def med(col):
        vals = [float(x[col]) for x in rows if x[col] != ""]
        return statistics.median(vals) if vals else None

    d = statistics.median(float(x["e2e_s"]) - float(x["load_s"]) - float(x["preprocess_s"]) for x in rows)
    summary = [l for l in r.stdout.splitlines() if l.startswith("[") or l.startswith("      ")]
    return {"ok": True, "correct": all(x["correct"] == "1" for x in rows), "decision_cost_s": d,
            "process_wall_s": wall, "driver_summary": summary,
            **{c: med(c) for c in ("load_s", "preprocess_s", "setup_s", "alloc_s", "h2d_s", "d2h_s",
                                   "compute_s", "comm_s", "e2e_s")}}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--algo", required=True, choices=ALGOS)
    ap.add_argument("--policy", default="tree", choices=["tree", "knn", "rule", "fixed", "seq"])
    ap.add_argument("--train")
    ap.add_argument("--model")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-gain", type=float, default=1.05)
    ap.add_argument("--keep-self", action="store_true")
    ap.add_argument("--verify", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--directed", action="store_true")
    g.add_argument("--undirected", action="store_true")
    ap.add_argument("--source", type=int)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    t_start = time.perf_counter()
    trace = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "algo": a.algo}

    say(f"=== gps: {a.algo} on {a.graph} ===")
    if not (BUILD / "gps_bench").exists():
        sys.exit(f"{BUILD}/gps_bench not found: run scripts/build.sh first")

    # 1. ingest and convert
    t0 = time.perf_counter()
    force = True if a.directed else False if a.undirected else None
    info = sniff(a.graph, force)
    gpath = info["converted_path"] or info["path"]
    flags = [] if info["directed"] else ["--undirected"]
    t_ingest = time.perf_counter() - t0
    say(f"[1] Ingest    format={info['format']}  {'directed' if info['directed'] else 'undirected'} "
        f"(from {info['directed_source']})  {fmt(t_ingest)}")
    say("              the C++ loader builds CSR (compacts ids, removes self-loops and duplicates)")

    # 2. machine profile
    machine = detect_machine()
    gpu = machine["gpu"]
    say(f"[2] Machine   {machine['cpu']} | {machine['cores']} cores | "
        f"GPU: {gpu['name'] + ' ' + str(gpu['mem_mib']) + ' MiB' if gpu else 'none'} | "
        f"MPI: {'yes' if machine['mpi'] else 'no'}")

    # 3. features
    f, t_feat = get_features(gpath, flags)
    say(f"[3] Features  n={f['n']} m={f['m']} mean_deg={f['mean_degree']:.3g} skew={f['degree_skew']:.3g} "
        f"out_cv={f['out_cv']:.3g} dangling={f['dangling_frac']:.3g} "
        f"symmetric={bool(f['symmetric'])}")
    say(f"              extraction {fmt(f['extract_s'])} (feature process wall {fmt(t_feat)}, includes loading the graph)")

    # 4. candidates
    cands, excluded = candidates(machine, a.algo, f)
    say(f"[4] Options   {', '.join(cfg_name(c) for c in cands)}")
    for e in excluded:
        say(f"              not available: {e}")

    # 5. decision
    t0 = time.perf_counter()
    model = json.load(open(a.model)) if a.model else None
    train_path, train = None, {}
    if model is None:
        train_path = a.train or next(iter(sorted(glob.glob(str(ROOT / "results/processed/*_dataset.csv")),
                                                 key=os.path.getmtime, reverse=True)), None)
        train = policies.load_train(train_path) if train_path else {}
    exclude = set() if a.keep_self else {info.get("name")}
    choice, scores, reason, dinfo = policies.decide(a.policy, train, a.algo, policies.fvec(f), cands,
                                                    exclude, k=a.k, min_gain=a.min_gain, model=model)
    policy, nbrs = dinfo["policy_used"], None
    t_decide = time.perf_counter() - t0
    say(f"[5] Decision  policy={policy}  ->  {cfg_name(choice)}   ({reason})")
    if model is not None:
        say(f"              trained model: {a.model}")
    elif train_path:
        say(f"              training data: {train_path}" + ("" if a.keep_self else "  (query graph held out)"))
    for line in dinfo["explain"]:
        say("              " + line)
    if scores:
        say("              predicted speedup: " + ", ".join(
            f"{cfg_name(c)} {math.exp(s):.2f}x" for c, s in sorted(scores.items(), key=lambda kv: -kv[1])[:4]))
    overhead = f["extract_s"] + t_decide   # graph assumed loaded once, like evaluate.py

    # 6. run
    tmp = tempfile.mkdtemp(prefix="gps_")
    say(f"[6] Run       {cfg_name(choice)}: {a.warmup} warm-up + {a.trials} timed trials ...")
    res = run_config(choice, a.algo, gpath, flags, a, tmp)
    if not res["ok"]:
        sys.exit(f"run failed:\n{res['error']}")
    for line in res["driver_summary"]:
        say("              " + line.strip())
    t_wall = time.perf_counter() - t_start

    # 7. metrics
    one_shot = t_ingest + overhead + res["e2e_s"]
    say("")
    say(f"[7] Metrics (median of {a.trials} trials)")
    say(f"    correct                 {'yes' if res['correct'] else 'NO'}")
    say(f"    raw compute runtime     {fmt(res['compute_s'])}" +
        (f"   (of which MPI communication {fmt(res['comm_s'])})" if res.get("comm_s") else ""))
    say(f"    run end-to-end          {fmt(res['e2e_s'])}   = load {fmt(res['load_s'])} + preprocess "
        f"{fmt(res['preprocess_s'])} + setup/transfers + compute")
    say(f"    selection overhead      {fmt(overhead)}   ({100 * overhead / one_shot:.1f}% of one-shot total)")
    say(f"    (prototype loads the graph twice: extra {fmt(t_feat - f['extract_s'])} is not counted as selection overhead)")
    say(f"    one-shot total          {fmt(one_shot)}   = ingest + selection + one run")
    say(f"    wall clock of this demo {fmt(t_wall)}   (includes warm-up and repeated trials)")

    trace.update({"graph": info, "machine": machine, "features": f, "candidates": [list(c) for c in cands],
                  "excluded": excluded, "decision": {"policy": policy, "chosen": list(choice), "reason": reason,
                  "neighbors": nbrs, "predicted_log_speedup": {f"{k[0]}@{k[1]}": v for k, v in (scores or {}).items()},
                  "train": train_path}, "run": res,
                  "timing": {"ingest_s": t_ingest, "feature_wall_s": t_feat, "decide_s": t_decide,
                             "selection_overhead_s": overhead, "one_shot_total_s": one_shot, "demo_wall_s": t_wall}})

    # optional: measure every option
    if a.verify:
        say("")
        say("[verify] measuring every option (decision cost = setup + compute, excludes loading)")
        measured = {choice: res}
        for c in cands:
            if c not in measured:
                m = run_config(c, a.algo, gpath, flags, a, tmp)
                if m["ok"]:
                    measured[c] = m
        ordered = sorted(measured.items(), key=lambda kv: kv[1]["decision_cost_s"])
        seq_d = measured.get(("seq", 1), {}).get("decision_cost_s")
        for rank, (c, m) in enumerate(ordered, 1):
            sp = f"{seq_d / m['decision_cost_s']:.2f}x" if seq_d else "-"
            tag = "  <- chosen" if c == choice else ("  <- best" if rank == 1 else "")
            say(f"    {rank:2d}. {cfg_name(c):12s} {m['decision_cost_s']:.4g} s   speedup vs sequential {sp}{tag}")
        best = ordered[0][1]["decision_cost_s"]
        regret = res["decision_cost_s"] / best - 1
        say(f"    selector regret vs best option: {100 * regret:.1f}%   (0% = picked the best)")
        trace["verify"] = {cfg_name(c): m["decision_cost_s"] for c, m in ordered}
        trace["verify_regret"] = regret

    out_dir = ROOT / "results" / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(a.graph).stem
    path = out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{stem}_{a.algo}.json"
    with open(path, "w") as fh:
        json.dump(trace, fh, indent=2, default=str)
    say(f"\ntrace saved: {path}")
    if info["converted_path"]:
        os.remove(info["converted_path"])
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()