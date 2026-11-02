#!/usr/bin/env bash
# Graph features for every graph in $GRAPHS -> results/features/<RUN_TAG>_features.jsonl
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common/env.sh"
if [ ! -x "$BUILD_DIR/gps_bench" ]; then echo "run scripts/build.sh first" >&2; exit 1; fi
mkdir -p "$ROOT/results/features"
out="$ROOT/results/features/${RUN_TAG}_features.jsonl"
: > "$out"
for g in $GRAPHS; do
  line="$("$BUILD_DIR/gps_bench" --graph "$(graph_file "$g")" $(graph_flags "$g") --features \
          | grep '^FEATURES_JSON ' | sed 's/^FEATURES_JSON //')"
  echo "{\"graph\":\"$g\",${line#\{}" >> "$out"
  echo "$g: done"
done
echo "wrote $out"