#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chrombpnet_predict_onefile.py  (CSV-enabled)

Single-file OR batch-from-CSV ChromBPNet inference.

CSV mode
--------
Expect a 2-column CSV:
  col0: sequence_id (string, used for output file names)
  col1: either DNA sequence (A/C/G/T/N) or genomic region "chr:start-end"

The script auto-detects which each row is. If ANY row is a region, you must
provide --genome FASTA (pyfaidx required).

Outputs:
 - One NPZ per input row: <out_dir>/<sequence_id>.npz
 - An index CSV summarizing inputs and written NPZ paths

Examples
--------
# 1) CSV with sequences
python chrombpnet_predict_onefile.py \
  --model /path/to/chrombpnet_model.h5 \
  --csv inputs.csv \
  --inputlen 2114 --outputlen 1000 \
  --out-dir ./preds_csv

# 2) CSV with coordinates (requires FASTA)
python chrombpnet_predict_onefile.py \
  --model /path/to/chrombpnet_model.h5 \
  --csv inputs_coords.csv \
  --genome /path/to/hg38.fa \
  --inputlen 2114 --outputlen 1000 \
  --out-dir ./preds_csv

# 3) Bias/control input shared for all rows
python chrombpnet_predict_onefile.py \
  --model /path/to/chrombpnet_model.h5 \
  --csv inputs.csv \
  --inputlen 2114 --outputlen 1000 \
  --bias-fill 0.0 \
  --out-dir ./preds_csv
"""
import os
import re
import csv
import json
import argparse
import numpy as np

# ------------------------------
# Optional: quiet TF logs
# ------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ------------------------------
# Keras model loading
# ------------------------------
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K

def multinomial_nll(y_true, y_pred):
    # dummy; not used in inference
    return K.mean(y_pred)

get_custom_objects().update({"multinomial_nll": multinomial_nll})

# ------------------------------
# Optional FASTA fetch
# ------------------------------
try:
    import pyfaidx
except Exception:
    pyfaidx = None

def infer_model_inputlen(model) -> int:
    """
    Try to infer required sequence length from model.inputs[0].shape.
    Supports channels-last ([None, L, 4]) and channels-first ([None, 4, L]).
    """
    inp = model.inputs[0].shape  # TensorShape
    # Convert to list of ints or None
    dims = [d if isinstance(d, int) else (int(d) if d is not None else None) for d in inp]
    # Typical cases:
    # channels-last: [None, L, 4]
    # channels-first: [None, 4, L]
    if len(dims) == 3:
        if dims[-1] == 4 and dims[1] is not None:
            return int(dims[1])  # L
        if dims[1] == 4 and dims[2] is not None:
            return int(dims[2])  # L
    # If we can’t infer, return None to keep CLI value
    return None

def fetch_region_sequence(genome_fasta: str, region: str) -> str:
    """
    Fetch uppercase sequence for region 'chr:start-end' from FASTA.
    Requires pyfaidx.
    """
    if pyfaidx is None:
        raise RuntimeError("pyfaidx is required for region fetching (pip install pyfaidx).")
    fa = pyfaidx.Fasta(genome_fasta, as_raw=True, sequence_always_upper=True)
    chrom, span = region.split(":")
    start, end = map(int, span.replace(",", "").split("-"))
    seq = str(fa[chrom][start:end])
    fa.close()
    return seq.upper()

# ------------------------------
# One-hot encoder (channels-last)
# ------------------------------
_DNA_TO_COL = {"A": 0, "C": 1, "G": 2, "T": 3}

def dna_to_one_hot(seq_list, unknown_zero=True, dtype=np.float32):
    """
    Encode list of equal-length strings to one-hot [N, L, 4] (channels-last).
    Unknown bases -> all zeros (ChromBPNet common choice).
    """
    if len(seq_list) == 0:
        raise ValueError("seq_list is empty.")
    L = len(seq_list[0])
    if any(len(s) != L for s in seq_list):
        raise ValueError("All sequences must be the same length.")
    X = np.zeros((len(seq_list), L, 4), dtype=dtype)
    for i, s in enumerate(seq_list):
        s = s.upper()
        for j, b in enumerate(s):
            idx = _DNA_TO_COL.get(b, None)
            if idx is not None:
                X[i, j, idx] = 1.0
            elif not unknown_zero:
                # If you prefer N as uniform 0.25, set unknown_zero=False
                X[i, j, :] = 0.25
    return X

# ------------------------------
# Shape helpers
# ------------------------------
def center_crop_or_pad_L4(x_L4: np.ndarray, target_len: int, fill_val: float = 0.0) -> np.ndarray:
    """
    x_L4: [L,4] -> [target_len,4] by center crop or constant padding (channels-last).
    """
    L = x_L4.shape[0]
    if L == target_len:
        return x_L4
    if L > target_len:
        s = (L - target_len) // 2
        return x_L4[s:s + target_len]
    pad_total = target_len - L
    left = pad_total // 2
    right = pad_total - left
    if fill_val == 0.0:
        left_pad = np.zeros((left, 4), dtype=x_L4.dtype)
        right_pad = np.zeros((right, 4), dtype=x_L4.dtype)
    else:
        left_pad = np.full((left, 4), fill_val, dtype=x_L4.dtype)
        right_pad = np.full((right, 4), fill_val, dtype=x_L4.dtype)
    return np.vstack([left_pad, x_L4, right_pad])

def softmax_len_axis(profile_logits: np.ndarray) -> np.ndarray:
    """
    profile_logits: [B, Lc, 2]
    Softmax along the length axis (axis=1), independently per strand.
    """
    m = np.max(profile_logits, axis=1, keepdims=True)
    e = np.exp(profile_logits - m)
    denom = np.sum(e, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return e / denom

# ------------------------------
# Prediction core
# ------------------------------
def load_chrombpnet(model_path: str):
    """Load a Keras ChromBPNet model. compile=False avoids needing full loss/metrics."""
    model = load_model(model_path, compile=False)
    return model

def prepare_inputs(seq: str, inputlen: int, unknown_zero: bool = True) -> np.ndarray:
    """seq -> one-hot [1, inputlen, 4] (channels-last), with centered crop/pad."""
    seq = re.sub(r"[^ACGTNacgtn]", "N", seq).upper()
    x = dna_to_one_hot([seq], unknown_zero=unknown_zero, dtype=np.float32)  # [1, L, 4]
    x0 = center_crop_or_pad_L4(x[0], inputlen, fill_val=0.0)               # [inputlen, 4]
    return x0[None, ...]                                                   # [1, inputlen, 4]

def prepare_bias(inputlen: int, bias_npy: str = None, bias_fill: float = 0.0) -> np.ndarray:
    """Prepare optional bias/control track [1, inputlen, 1]."""
    if bias_npy:
        vec = np.load(bias_npy)  # allow shape [inputlen] or [inputlen, 1]
        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim == 1:
            if vec.shape[0] != inputlen:
                raise ValueError(f"--bias-npy length {vec.shape[0]} != inputlen {inputlen}")
            vec = vec[:, None]
        elif vec.ndim == 2:
            if vec.shape[0] != inputlen or vec.shape[1] != 1:
                raise ValueError(f"--bias-npy must be [inputlen,1], got {vec.shape}")
        else:
            raise ValueError(f"--bias-npy must be 1D or 2D, got {vec.ndim}D")
        xb = vec[None, ...]  # [1, inputlen, 1]
    else:
        xb = np.full((1, inputlen, 1), float(bias_fill), dtype=np.float32)
    return xb

def forward(model, x: np.ndarray, xb: np.ndarray = None):
    """
    Run model forward. Supports 1 or 2 inputs. Returns (profile_logits, log_counts).
    """
    if isinstance(model.inputs, (list, tuple)) and len(model.inputs) == 2:
        if xb is None:
            raise ValueError("Model expects a bias/control input; provide --bias-npy or --bias-fill.")
        preds = model.predict([x, xb], batch_size=1, verbose=0)
    else:
        preds = model.predict(x, batch_size=1, verbose=0)

    profile_logits = None
    log_counts = None
    if isinstance(preds, dict):
        profile_logits = preds.get("profile", None)
        log_counts = preds.get("counts", None)
    elif isinstance(preds, (list, tuple)):
        if len(preds) >= 2:
            profile_logits, log_counts = preds[0], preds[1]
        elif len(preds) == 1:
            profile_logits = preds[0]
    else:
        profile_logits = preds

    if profile_logits is None:
        raise RuntimeError("Could not find profile logits in model outputs.")

    if hasattr(profile_logits, "numpy"):
        profile_logits = profile_logits.numpy()
    if log_counts is not None and hasattr(log_counts, "numpy"):
        log_counts = log_counts.numpy()

    return profile_logits, log_counts

def run_single_prediction(
    model,
    seq_str: str,
    inputlen: int = 2114,
    outputlen: int = 1000,
    xb: np.ndarray = None,
    unknown_zero: bool = True,
):
    """
    Core for a single pre-fetched sequence string.
    Returns dict with results.
    """
    x = prepare_inputs(seq_str, inputlen=inputlen, unknown_zero=unknown_zero)  # [1, inputlen, 4]
    prof_logits, log_counts = forward(model, x, xb=xb)                         # [1, Lc, 2], [1,2] or None

    # Softmax along length axis -> per-base per-strand probabilities
    prof_probs = softmax_len_axis(prof_logits)                                  # [1, Lc, 2]

    # Center-crop to outputlen if needed
    Lc = prof_probs.shape[1]
    if outputlen is not None and outputlen != Lc:
        start = (Lc - outputlen) // 2
        prof_probs = prof_probs[:, start:start + outputlen, :]

    # Squeeze batch dimension for saving
    prof_probs = prof_probs[0]                      # [outputlen, 2]
    if log_counts is not None:
        log_counts = log_counts[0]                 # [2] or [1]
    return {
        "profile_probs": prof_probs,
        "log_counts": log_counts,
        "meta": {
            "inputlen": int(inputlen),
            "outputlen": int(outputlen),
        },
    }

# ------------------------------
# CSV helpers
# ------------------------------
_COORD_RE = re.compile(r"^([a-zA-Z0-9_.]+):(\d+)-(\d+)$")

def looks_like_region(s: str) -> bool:
    return _COORD_RE.match(s.replace(",", "")) is not None

def looks_like_sequence(s: str) -> bool:
    return re.fullmatch(r"[ACGTNacgtn]+", s or "") is not None

def sanitize_id(s: str) -> str:
    # safe-ish filename stem
    s = str(s).strip()
    s = re.sub(r"[^\w.\-]+", "_", s)
    return s if s else "unnamed"

def run_csv_batch(
    csv_path: str,
    model_path: str,
    genome_fasta: str,
    inputlen: int,
    outputlen: int,
    bias_npy: str,
    bias_fill: float,
    unknown_zero: bool,
    out_dir: str,
    has_header: bool,
    id_col: int,
    value_col: int,
):
    os.makedirs(out_dir, exist_ok=True)

    # Load model once
    model = load_chrombpnet(model_path)

    model_inputlen = infer_model_inputlen(model)
    if model_inputlen is not None and model_inputlen != inputlen:
        print(f"[INFO] Overriding --inputlen {inputlen} -> model-required {model_inputlen}")
        inputlen = model_inputlen

    # <<< MOVED BELOW the override: prepare bias with the finalized inputlen >>>
    xb = None
    if isinstance(model.inputs, (list, tuple)) and len(model.inputs) == 2:
        xb = prepare_bias(inputlen=inputlen, bias_npy=bias_npy, bias_fill=bias_fill)

    # Pass 1: detect whether any rows are regions; enforce FASTA if so.
    any_region = False
    with open(csv_path, "r", newline="") as fh:
        rdr = csv.reader(fh)
        if has_header:
            next(rdr, None)
        for row in rdr:
            if not row or len(row) <= max(id_col, value_col):
                continue
            val = (row[value_col] or "").strip()
            if looks_like_region(val):
                any_region = True
                break
    if any_region and not genome_fasta:
        raise ValueError("CSV includes coordinates; please provide --genome FASTA.")

    # Re-open and process
    index_rows = []
    with open(csv_path, "r", newline="") as fh:
        rdr = csv.reader(fh)
        if has_header:
            header = next(rdr, None)
        for i, row in enumerate(rdr, start=1):
            if not row or len(row) <= max(id_col, value_col):
                print(f"[WARN] Row {i}: missing columns, skipping")
                continue
            seq_id = sanitize_id(row[id_col])
            val = (row[value_col] or "").strip()
            mode = "sequence" if looks_like_sequence(val) else ("region" if looks_like_region(val) else "unknown")
            if mode == "unknown":
                print(f"[WARN] Row {i} ({seq_id}): second column not sequence/region, skipping")
                continue

            try:
                seq_str = val if mode == "sequence" else fetch_region_sequence(genome_fasta, val)
                results = run_single_prediction(
                    model=model,
                    seq_str=seq_str,
                    inputlen=inputlen,
                    outputlen=outputlen,
                    xb=xb,
                    unknown_zero=unknown_zero,
                )

                out_path = os.path.join(out_dir, f"{seq_id}.npz")
                meta = dict(results["meta"])
                meta.update({
                    "seq_id": seq_id,
                    "source": mode,
                    "value": val,
                })
                np.savez_compressed(
                    out_path,
                    profile=results["profile_probs"],
                    log_counts=(results["log_counts"] if results["log_counts"] is not None else np.array([])),
                    meta=json.dumps(meta),
                )
                print(f"[OK] Saved {out_path}  shape={results['profile_probs'].shape}")
                index_rows.append([seq_id, mode, val, out_path])
            except Exception as e:
                print(f"[ERROR] Row {i} ({seq_id}): {e}")

    # Write index CSV
    idx_path = os.path.join(out_dir, "index.csv")
    with open(idx_path, "w", newline="") as ofh:
        w = csv.writer(ofh)
        w.writerow(["seq_id", "kind", "value", "npz_path"])
        for r in index_rows:
            w.writerow(r)
    print(f"[OK] Wrote index: {idx_path}  ({len(index_rows)} rows)")

# ------------------------------
# CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ChromBPNet predictor (single or CSV batch)")
    p.add_argument("--model", required=True, help="Path to ChromBPNet .h5 (Keras) model")

    # SINGLE ITEM mode (kept from original)
    p.add_argument("--seq", type=str, default=None, help="Raw DNA string (A/C/G/T/N)")
    p.add_argument("--genome", type=str, default=None, help="Reference FASTA (required with --region or rows with coordinates)")
    p.add_argument("--region", type=str, default=None, help="Genomic region: chr:start-end")

    # CSV mode
    p.add_argument("--csv", type=str, default=None, help="Path to a 2-column CSV (id,value)")
    p.add_argument("--has-header", action="store_true", help="CSV has a header row")
    p.add_argument("--id-col", type=int, default=0, help="0-based index of ID column")
    p.add_argument("--value-col", type=int, default=1, help="0-based index of value (sequence or region) column")
    p.add_argument("--out-dir", type=str, default="preds_csv", help="Directory to write per-row NPZ and index.csv")

    p.add_argument("--inputlen", type=int, default=2114, help="Model input length")
    p.add_argument("--outputlen", type=int, default=1000, help="Center window length in outputs")
    p.add_argument("--bias-npy", type=str, default=None, help="Path to .npy bias vector [inputlen] or [inputlen,1] (shared for all rows)")
    p.add_argument("--bias-fill", type=float, default=0.0, help="Constant to fill bias input if model expects it")
    p.add_argument("--unknown-zero", action="store_true",
                   help="Unknown bases -> zeros (default). If NOT set, uses uniform 0.25 per base.")
    p.add_argument("--out-prefix", type=str, default="single_pred",
                   help="Output prefix for SINGLE-ITEM mode (writes .npz). Ignored in CSV mode.")
    return p.parse_args()

def main():
    args = parse_args()

    # If flag is present -> True, else -> False (default False means uniform 0.25 if not set)
    unknown_zero = True if args.unknown_zero else True  # keep default True

    if args.csv:
        # Batch mode
        run_csv_batch(
            csv_path=args.csv,
            model_path=args.model,
            genome_fasta=args.genome,
            inputlen=args.inputlen,
            outputlen=args.outputlen,
            bias_npy=args.bias_npy,
            bias_fill=args.bias_fill,
            unknown_zero=unknown_zero,
            out_dir=args.out_dir,
            has_header=args.has_header,
            id_col=args.id_col,
            value_col=args.value_col,
        )
        return

    # Single-item mode (original behavior)
    if (args.seq is None) == (args.region is None):
        raise ValueError("Provide exactly one of --seq or --region (or use --csv).")

    if args.region is not None and args.genome is None:
        raise ValueError("--region requires --genome FASTA")

    # Prepare inputs common to single mode
    model = load_chrombpnet(args.model)

    # Infer and enforce correct input length from model, if possible
    model_inputlen = infer_model_inputlen(model)
    if model_inputlen is not None and model_inputlen != args.inputlen:
        print(f"[INFO] Overriding --inputlen {args.inputlen} -> model-required {model_inputlen}")
        args.inputlen = model_inputlen
    xb = None
    if isinstance(model.inputs, (list, tuple)) and len(model.inputs) == 2:
        xb = prepare_bias(inputlen=args.inputlen, bias_npy=args.bias_npy, bias_fill=args.bias_fill)

    results = run_single_prediction(
        model=model,
        seq_str=seq_str,
        inputlen=args.inputlen,
        outputlen=args.outputlen,
        xb=xb,
        unknown_zero=unknown_zero,
    )

    # Write NPZ
    if os.path.dirname(args.out_prefix):
        os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    np.savez_compressed(
        f"{args.out_prefix}.npz",
        profile=results["profile_probs"],
        log_counts=(results["log_counts"] if results["log_counts"] is not None else np.array([])),
        meta=json.dumps(results["meta"]),
    )
    print(f"[OK] Saved {args.out_prefix}.npz")
    print(f"  profile shape: {results['profile_probs'].shape}  "
          f"log_counts: {None if results['log_counts'] is None else results['log_counts'].shape}")

if __name__ == "__main__":
    main()
