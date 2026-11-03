#!/usr/bin/env python3
"""Minimal local web UI for the selector (standard library only).

usage: python selector/app.py [--port 8000] [--model results/models/X_tree.json | --train DATASET.csv] [--open]
Binds to 127.0.0.1 only. One run at a time, so concurrent jobs cannot distort each other's timings.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
GPS = ROOT / "selector" / "gps.py"
LOCK = threading.Lock()
CONFIG = {}
MAX_BYTES = 2 * 1024 ** 3
ALGOS = ("bfs", "pagerank", "cc", "sssp", "tc")

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Graph analytics selector</title>
<style>
:root{color-scheme:light dark;--bg:#fafaf9;--fg:#1c1b19;--card:#fff;--line:#d9d6cf;--mute:#6b6962;--acc:#1d6f5f}
@media (prefers-color-scheme:dark){:root{--bg:#1b1a18;--fg:#ecebe7;--card:#252320;--line:#3a3834;--mute:#a4a198;--acc:#4fc3a8}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:900px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:0 0 8px}
.sub{color:var(--mute);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:0 0 14px}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:end}
label{display:block;font-size:13px;color:var(--mute);margin-bottom:3px}
select,input[type=file]{font:inherit;padding:6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
button{font:inherit;padding:8px 18px;border:0;border-radius:6px;background:var(--acc);color:#fff;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}
table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}
td:last-child,th:last-child{text-align:right}
.big{font-size:24px;font-weight:600;color:var(--acc);margin:0}
pre{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:10px;overflow:auto;font-size:12.5px;margin:6px 0 0}
.err{border-color:#b3261e}.mute{color:var(--mute)}
</style></head><body><main>
<h1>Graph analytics selector</h1>
<p class="sub">Upload a graph, pick an algorithm. The selector converts the graph, chooses an implementation, runs it and reports the metrics.</p>
<div class="card"><div class="row">
<div><label for="file">Graph file (edge list .txt/.el, or .mtx)</label><input id="file" type="file"></div>
<div><label for="algo">Algorithm</label><select id="algo">
<option value="bfs">BFS</option><option value="pagerank">PageRank</option>
<option value="cc">Connected components</option><option value="sssp">SSSP</option>
<option value="tc">Triangle counting</option></select></div>
<div><label for="dir">Graph type</label><select id="dir">
<option value="auto">auto-detect</option><option value="directed">directed</option><option value="undirected">undirected</option></select></div>
<div><label><input id="verify" type="checkbox"> measure all options too (slower)</label></div>
<div><button id="run">Run</button></div></div>
<p id="status" class="mute" style="margin:10px 0 0"></p></div>
<div id="out"></div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt=x=>x==null?'-':x<1e-3?(x*1e6).toFixed(0)+' \u00b5s':x<1?(x*1e3).toFixed(2)+' ms':x.toFixed(3)+' s';
const label=c=>c[0]==='seq'?'sequential':c[0]==='cuda'?'CUDA':(c[0]==='omp'?'OpenMP':'MPI')+' x'+c[1];
const rows=a=>a.map(r=>'<tr><td>'+esc(r[0])+'</td><td>'+esc(r[1])+'</td></tr>').join('');
function render(t,log){
  const run=t.run,d=t.decision,tm=t.timing,f=t.features,g=t.graph,m=t.machine;
  const pct=(100*tm.selection_overhead_s/tm.one_shot_total_s).toFixed(1);
  let h='<div class="card"><h2>Decision</h2><p class="big">'+esc(label(d.chosen))+'</p><p>'+esc(d.reason)+
    '<br><span class="mute">policy: '+esc(d.policy)+' | '+esc(g.format)+', '+(g.directed?'directed':'undirected')+
    ' ('+esc(g.directed_source)+') | machine: '+esc(m.cpu)+', '+m.cores+' cores'+(m.gpu?', '+esc(m.gpu.name):', no GPU detected')+'</span>'+
    (t.excluded&&t.excluded.length?'<br><span class="mute">Not available here (excluded from the choice): '+esc(t.excluded.join('; '))+'</span>':'')+'</p></div>';
  h+='<div class="card"><h2>Metrics (median of trials)</h2><table>'+rows([
    ['Correct',run.correct?'yes':'NO'],
    ['Raw compute runtime',fmt(run.compute_s)+(run.comm_s?' (MPI communication '+fmt(run.comm_s)+')':'')],
    ['Run end-to-end (load + preprocess + setup + compute)',fmt(run.e2e_s)],
    ['Selection overhead',fmt(tm.selection_overhead_s)+' ('+pct+'% of one-shot total)'],
    ['One-shot total (ingest + selection + one run)',fmt(tm.one_shot_total_s)],
    ['Wall clock of this request',fmt(tm.demo_wall_s)]])+'</table></div>';
  h+='<div class="card"><h2>Algorithm result</h2><pre>'+esc((run.driver_summary||[]).join('\n'))+'</pre></div>';
  h+='<div class="card"><h2>Graph features</h2><table>'+rows([['Vertices',f.n],['Edges (directed, after cleanup)',f.m],
    ['Mean degree',f.mean_degree.toFixed(3)],['Degree skew (max / mean)',f.degree_skew.toFixed(2)],
    ['Out-degree CV',f.out_cv.toFixed(3)],['Dangling fraction',f.dangling_frac.toFixed(3)],
    ['Feature extraction',fmt(f.extract_s)]])+'</table></div>';
  if(t.verify){
    const seq=t.verify['sequential'];const e=Object.entries(t.verify).sort((a,b)=>a[1]-b[1]);
    h+='<div class="card"><h2>All options, measured (setup + compute)</h2><table><tr><th>Option</th><th>Time / speedup vs sequential</th></tr>'+
      e.map((x,i)=>'<tr><td>'+esc(x[0])+(x[0]===label(d.chosen)?' (chosen)':'')+(i===0?' (best)':'')+'</td><td>'+fmt(x[1])+
      (seq?' / '+(seq/x[1]).toFixed(2)+'x':'')+'</td></tr>').join('')+
      (t.excluded||[]).map(x=>'<tr><td>'+esc(x)+'</td><td>excluded, not measured</td></tr>').join('')+
      '</table><p class="mute">Selector regret vs best: '+
      (100*t.verify_regret).toFixed(1)+'% (0% = picked the best)</p></div>';
  }
  h+='<div class="card"><details><summary>Full log</summary><pre>'+esc(log)+'</pre></details></div>';
  $('out').innerHTML=h;
}
$('run').onclick=async()=>{
  const file=$('file').files[0];
  if(!file){$('status').textContent='Choose a graph file first.';return;}
  const q=new URLSearchParams({algo:$('algo').value,name:file.name,direction:$('dir').value,verify:$('verify').checked?'1':'0'});
  $('run').disabled=true;$('out').innerHTML='';
  $('status').textContent='Running: converting, extracting features, deciding, executing (large graphs take a while)...';
  try{
    const resp=await fetch('/run?'+q,{method:'POST',body:file});
    const data=await resp.json();
    if(!data.ok){$('status').textContent='';$('out').innerHTML='<div class="card err"><h2>Run failed</h2><pre>'+esc(data.error||'unknown error')+'</pre></div>';}
    else{$('status').textContent='Done.';render(data.trace,data.log);}
  }catch(e){$('status').textContent='Request failed: '+e;}
  $('run').disabled=false;
};
</script></main></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[app] " + fmt % args + "\n")

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _err(self, code, msg):
        self._send(code, json.dumps({"ok": False, "error": msg}))

    def do_GET(self):
        if urlparse(self.path).path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/run":
            return self._err(404, "not found")
        q = parse_qs(u.query)
        algo = q.get("algo", [""])[0]
        if algo not in ALGOS:
            return self._err(400, "unknown algorithm")
        name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(q.get("name", ["graph.txt"])[0])) or "graph.txt"
        direction = q.get("direction", ["auto"])[0]
        verify = q.get("verify", ["0"])[0] == "1"
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > MAX_BYTES:
            return self._err(400, "empty or too large upload")
        if not LOCK.acquire(blocking=False):
            return self._err(429, "another run is in progress; wait for it to finish")
        tmp = tempfile.mkdtemp(prefix="gps_up_")
        try:
            path = os.path.join(tmp, name)
            remaining = length
            with open(path, "wb") as f:
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            cmd = [sys.executable, str(GPS), "--graph", path, "--algo", algo]
            if direction == "directed":
                cmd.append("--directed")
            elif direction == "undirected":
                cmd.append("--undirected")
            if verify:
                cmd.append("--verify")
            if CONFIG.get("model"):
                cmd += ["--model", CONFIG["model"]]
            elif CONFIG.get("train"):
                cmd += ["--train", CONFIG["train"]]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
            log = r.stdout + (("\n" + r.stderr) if r.stderr.strip() else "")
            m = re.search(r"trace saved: (.+)", r.stdout)
            trace = None
            if r.returncode == 0 and m:
                with open(m.group(1).strip()) as fh:
                    trace = json.load(fh)
            ok = r.returncode == 0 and trace is not None
            payload = {"ok": ok, "log": log, "trace": trace}
            if not ok:
                payload["error"] = (r.stderr or r.stdout)[-1200:]
            self._send(200, json.dumps(payload))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            LOCK.release()


def newest(pattern):
    files = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0]) if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", help="exported tree JSON (results/models/*_tree.json)")
    ap.add_argument("--train", help="training dataset CSV (used when no model is given)")
    ap.add_argument("--open", action="store_true", help="open the page in a browser")
    a = ap.parse_args()
    if not (ROOT / "build" / "gps_bench").exists():
        sys.exit("build/gps_bench not found: run scripts/build.sh first")
    CONFIG["model"] = a.model or (None if a.train else newest("results/models/*_tree.json"))
    CONFIG["train"] = a.train
    if CONFIG["model"]:
        print(f"selector model: {CONFIG['model']}")
    elif CONFIG["train"] or newest("results/processed/*_dataset.csv"):
        print("no exported model found: the selector trains from the newest dataset on each request")
    else:
        print("warning: no model or dataset found; the selector falls back to its rule policy")
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}"
    print(f"serving {url} (Ctrl+C to stop)")
    if a.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()