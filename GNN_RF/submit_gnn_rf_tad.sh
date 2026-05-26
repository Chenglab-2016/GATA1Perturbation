#!/bin/bash
bsub -J gnn_rf_tad \
  -q large_mem \
  -n 2 \
  -R "span[hosts=1]" \
  -R "rusage[mem=50G]" \
  -o gnn_rf_tad.%J.out \
  -e gnn_rf_tad.%J.err \
  bash -lc '
set -euo pipefail

module load conda3/202402

# safe with set -u
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-INTEL}"

source activate torch_env

# make sure env libstdc++ is used first
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python gnn_embed_rf_tad_v1.py \
  --data-dir ./dataset \
  --features ./dataset/features_table.csv \
  --hit-metrics-out ./results/gnn_rf_tad_hit_metrics.csv \
  --hit-out ./results/gnn_rf_tad_hit_records.pkl \
  --embed-dim 8 \
  --seed 42 \
  --target-hits 50 \
  --best-pred-out ./results/gnn_rf_tad_best_spearman_predictions.csv
'