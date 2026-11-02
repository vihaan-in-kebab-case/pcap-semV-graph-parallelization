#!/usr/bin/env bash
# Downloads both SNAP graphs into datasets/graphs/ and verifies SHA-256. Safe to re-run.
#   FORCE=1 scripts/fetch-datasets.sh   re-downloads even if the files exist
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${GRAPHS_DIR:-$ROOT/datasets/graphs}"
FORCE="${FORCE:-0}"
mkdir -p "$DEST"

download() {   # url out
  if command -v curl >/dev/null 2>&1; then curl -fL --retry 3 -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then wget -q -O "$2" "$1"
  else echo "error: need curl or wget" >&2; exit 1; fi
}
sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
fetch() {      # file-name url expected-sha256 (of the decompressed file)
  local name="$1" url="$2" want="$3" txt="$DEST/$1" got
  if [ -f "$txt" ] && [ "$FORCE" != "1" ]; then
    echo "[skip] $name already present"
  else
    echo "[get ] $url"
    download "$url" "$txt.gz.part"
    mv "$txt.gz.part" "$txt.gz"
    gunzip -c "$txt.gz" > "$txt"
    rm -f "$txt.gz"
  fi
  got="$(sha256 "$txt")"
  if [ "$got" = "$want" ]; then echo "[ ok ] $name sha256 verified"
  else echo "[WARN] $name sha256 is $got (expected $want)" >&2; fi
}

fetch facebook_combined.txt https://snap.stanford.edu/data/facebook_combined.txt.gz \
  f41c026ed8af3cc3359f1ca5573d0605fb09ae0eefa34544b820fd8c6e2ef296
fetch web-Google.txt https://snap.stanford.edu/data/web-Google.txt.gz \
  8c0f453f1eb1e24ad145e36e542b129083237e96e585abae768927bdb70167d1
echo; ls -lh "$DEST"