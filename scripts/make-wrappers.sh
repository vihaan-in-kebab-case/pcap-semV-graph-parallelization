cd scripts
for impl in openmp mpi cuda; do
  mkdir -p "$impl"
  for algo in bfs pagerank cc sssp tc; do
    sed -e "s/@IMPL@/$impl/g" -e "s/@ALGO@/$algo/g" > "$impl/run-$algo.sh" <<'EOF'
#!/usr/bin/env bash
# @IMPL@ baseline: @ALGO@ on every graph in $GRAPHS (options: see scripts/common/env.sh)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"
run_baseline @IMPL@ @ALGO@
EOF
  done
done
chmod +x *.sh */*.sh
cd ..