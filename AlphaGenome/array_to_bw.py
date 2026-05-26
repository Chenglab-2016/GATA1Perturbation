#!/usr/bin/env python3
import argparse, numpy as np, pyBigWig

def main():
    ap = argparse.ArgumentParser(description="Write a BigWig from a NumPy array and a known genomic window.")
    ap.add_argument("--npy", required=True, help="Path to values.npy (1D array)")
    ap.add_argument("--chrom", required=True, help="Chromosome name (e.g., chr1)")
    ap.add_argument("--start", type=int, required=True, help="0-based start (bp) of the first bin")
    ap.add_argument("--bin-size", type=int, required=True, help="Bin width (bp)")
    ap.add_argument("--chrom-sizes", required=True, help="Two-column file: chrom\\tsize")
    ap.add_argument("--out-bw", required=True, help="Output BigWig path")
    ap.add_argument("--nan-policy", choices=["skip","zero"], default="skip", help="How to handle NaNs")
    args = ap.parse_args()

    # load values
    v = np.load(args.npy).astype(float)  # shape (N,)
    N = v.shape[0]

    # load chrom sizes and init header
    chrom_sizes = []
    with open(args.chrom_sizes) as f:
        for line in f:
            if not line.strip(): continue
            c, s = line.split()[:2]
            chrom_sizes.append((c, int(s)))
    bw = pyBigWig.open(args.out_bw, "w")
    bw.addHeader(chrom_sizes)

    # compute starts/ends for each bin
    starts = args.start + args.bin_size * np.arange(N, dtype=np.int64)
    ends = starts + args.bin_size

    # clip to chrom length
    chrom_len = dict(chrom_sizes)[args.chrom]
    valid = starts < chrom_len
    starts, ends, v = starts[valid], np.minimum(ends[valid], chrom_len), v[valid]

    # handle NaNs
    if args.nan_policy == "skip":
        keep = ~np.isnan(v)
        starts, ends, v = starts[keep], ends[keep], v[keep]
    else:
        v = np.nan_to_num(v, nan=0.0)

    # write
    if len(v):
        bw.addEntries([args.chrom]*len(v), starts.tolist(), ends=ends.tolist(), values=v.tolist())

    bw.close()
    print(f"Wrote {len(v)} bins to {args.out_bw}")

if __name__ == "__main__":
    main()
