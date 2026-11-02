tar xzf results_sweep_5090.tgz
python selector/replay.py --train results/processed/sweep_5090_dataset.csv --list
python selector/replay.py --train results/processed/sweep_5090_dataset.csv --graph rmat_n18_d8 --algo bfs
python selector/replay.py --train results/processed/sweep_5090_dataset.csv --queries results/processed/sweep_5090_demo_dataset.csv --graph demo_rmat_n18_d8 --algo pagerank
python selector/evaluate.py --dataset results/processed/sweep_5090_dataset.csv --show-tree