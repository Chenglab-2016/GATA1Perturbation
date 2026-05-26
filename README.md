# GATAPerturbatio Repository Guide

This repository contains analysis notebooks, scripts, trained-model workflows, and supporting datasets for **"Systematic perturbation reveals pleiotropic cis-regulatory elements that evade current deep learning models"**.

## Folders

### `AlphaGenome/`

Workflows and data for AlphaGenome-based prediction and downstream analysis.

Key notebooks and scripts:

- `GATA1_ATAC.ipynb` - ATAC-seq related AlphaGenome prediction workflow.
- `GATA1_Chip.ipynb` - ChIP-seq related AlphaGenome prediction workflow.
- `RNA_1M.ipynb` - RNA prediction or expression-oriented analysis workflow.
- `Track_prediction.ipynb` - Track-level prediction analysis.
- `Var_scores.ipynb` - Variant scoring workflow.
- `RF_model.ipynb` - Random forest modeling and evaluation using AlphaGenome-derived features or outputs.
- `array_to_bw.py` - Utility script for converting array-style prediction outputs to BigWig-style track files.


### `ChIP_Seq_CNN/`

CNN-based ChIP-seq modeling workflow.

Key files:

- `ChIP_CNN.ipynb` - Notebook for training or evaluating a DeepSTARR-style CNN on ChIP-seq data.


### `ChromBPNet/`

ChromBPNet prediction and post-processing workflow.

Key files:

- `train_model_get_prediction.sh` - Shell script for training the model and generating predictions.
- `chrombpnet_predict_onefilev.py` - Python script for running ChromBPNet prediction on a single input file.
- `get_y_pred_y_test.ipynb` - Notebook for extracting predicted and observed target values.
- `get_delta_atac.ipynb` - Notebook for computing or analyzing delta ATAC values.

### `Evo2/`

Evo2-based variant or sequence perturbation analysis.

Key files:

- `evo2_diff.py` - Python script for computing Evo2-based differences or scores.
- `data/json_for_evo2.json` - JSON input data for the Evo2 workflow.

### `GNN_RF/`

Graph neural network embedding plus random forest modeling workflow, likely using TAD and graph-structured genomic features.

Key files:

- `gnn_embed_rf_tad_v1.py` - Main Python workflow for GNN embedding and random forest modeling with TAD-related inputs.
- `submit_gnn_rf_tad.sh` - Shell script for submitting or running the GNN/RF workflow.
- `plot_model_shap.ipynb` - Notebooks for SHAP visualization and model interpretation.

Data files in `GNN_RF/dataset/` include:

- `TAD.csv` - TAD table used by the model.
- `features_table.csv` - Feature table for model input.
- `nodes.pkl` - Serialized graph node data.
- `linkes.pkl` - Serialized graph edge/link data.

## Notes

- The genomic reference file hg19.fa and the BigWig file rep_log.bw required for ChIP_Seq_CNN are not included because of their large file sizes.
- Notebooks may depend on local data paths, Python environments, or genomics libraries. Check each notebook or script for environment-specific setup before running.
