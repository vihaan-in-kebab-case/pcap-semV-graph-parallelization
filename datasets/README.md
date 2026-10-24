# Datasets

Dataset files are **not committed**. Download them into `datasets/graphs/`
(gitignored) with the commands below, and record the SHA-256 of each file.
The loader (`graph_loader.cpp`) compacts vertex ids, drops self-loops and
duplicate edges, and sorts neighbour lists, so edge counts after loading
can be smaller than the published counts.

## Downloads

| ID | Source | File in `datasets/graphs/` | Directed? | Nodes / edges (published) | SHA-256 |
|---|---|---|---|---|---|
| ego-Facebook | SNAP, https://snap.stanford.edu/data/ego-Facebook.html | `facebook_combined.txt` | undirected, each edge stored once | 4,039 / 88,234 | _fill in_ |
| web-Google | SNAP, https://snap.stanford.edu/data/web-Google.html | `web-Google.txt` | directed | 875,713 / 5,105,039 | _fill in_ |

```bash
cd datasets/graphs
curl -O https://snap.stanford.edu/data/facebook_combined.txt.gz
curl -O https://snap.stanford.edu/data/web-Google.txt.gz
gunzip -k *.gz
```

## Usage

- Undirected files that store each edge once (ego-Facebook): pass `--undirected`.
- Directed files (web-Google): run as loaded. CC and TC symmetrize automatically
  (weakly connected components; simple undirected triangles) and the
  symmetrization time is recorded as `preprocess_s`.
- SSSP weights are generated at run time (integer 1..100, seeded hash of the
  unordered endpoint pair, default seed 42); no weighted file is needed.
- Vertex ids in results are the compacted ids, not the original file ids.
- Matrix Market (`.mtx`) files are also supported; `symmetric` headers are
  expanded to both directions.

## Adding a dataset

Add a row to the table (source URL, file name, directedness, published size,
SHA-256) and the download command. Record the graph family and any
conversion applied.