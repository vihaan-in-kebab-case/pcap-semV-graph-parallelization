#!/usr/bin/env bash
# Reproduces the selector results from recorded sweep data. No GPU needed.
#   scripts/reproduce-demo.sh [TAG]        default TAG=sweep_5090
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TAG="${1:-sweep_5090}"
if [ -d .venv ]; then set +u; . .venv/bin/activate; set -u; fi
if [ -f "results_${TAG}.tgz" ] && [ ! -f "results/processed/${TAG}_dataset.csv" ]; then tar xzf "results_${TAG}.tgz"; fi
ls results/raw/${TAG}_*.csv >/dev/null 2>&1 || { echo "no raw sweep for ${TAG}: put results_${TAG}.tgz in the repo root" >&2; exit 1; }

echo "== recorded runs check =="
bad="$(awk -F, 'FNR>1 && $19==0' results/raw/${TAG}_*.csv | wc -l | tr -d ' ')"
echo "incorrect rows: ${bad} (must be 0)"

echo "== rebuild datasets from the raw sweep =="
python selector/build_dataset.py --tag "$TAG" ${DROP_IMPL:+--drop-impl "$DROP_IMPL"}
if [ -f "results/features/${TAG}_demo_features.jsonl" ] && [ -f datasets/manifest_demo.csv ]; then
  python selector/build_dataset.py --tag "${TAG}_demo" --manifest datasets/manifest_demo.csv
fi

echo "== train =="
python selector/train.py --dataset "results/processed/${TAG}_dataset.csv" > "results/processed/${TAG}_trees.txt"
tail -n 1 "results/processed/${TAG}_trees.txt"

echo "== evaluation: GPU present =="
python selector/evaluate.py --dataset "results/processed/${TAG}_dataset.csv" --show-tree | tee "results/processed/${TAG}_evaluation.txt"
echo "== evaluation: GPU absent (CUDA excluded) =="
python selector/evaluate.py --dataset "results/processed/${TAG}_dataset.csv" --no-gpu | tee "results/processed/${TAG}_evaluation_nogpu.txt"

echo "== sample replays =="
D="results/processed/${TAG}_dataset.csv"
python selector/replay.py --train "$D" --graph rmat_n18_d8 --algo bfs
python selector/replay.py --train "$D" --graph rmat_n18_d8 --algo bfs --no-gpu
echo
echo "figures: results/figures/selector_${TAG}/ and results/figures/selector_${TAG}_nogpu/"