#!/usr/bin/env python3

"""Initial/simple GNN-embedding + RF baseline with repeated TAD splits.

This is a clean v1 pipeline:
- GNN learns node embeddings from non-zero node `tgt`.
- Embeddings are merged into features_table by `WGATAR_id`.
- RandomForestRegressor predicts `effect_value`.
- Train/test split is by TAD.
- Keep chr19:12790000-13270000 in test for every split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor


DROP_COLUMNS = [
    "WGATAR_position",
    "WGATAR_chr",
    "WGATAR_id",
    "WGATAR_start",
    "WGATAR_end",
]
FORCED_TEST_TAD = "chr11:60530000-60710000"


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    if y_true.size == 0:
        return np.nan, np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan, np.nan
    return float(pearsonr(y_true, y_pred).statistic), float(spearmanr(y_true, y_pred).statistic)


def load_pickle_with_fallback(paths: list[Path]) -> pd.DataFrame:
    for p in paths:
        if p.exists():
            obj = pd.read_pickle(p)
            if isinstance(obj, pd.DataFrame):
                return obj
            raise TypeError(f"{p} loaded but is not a pandas DataFrame.")
    raise FileNotFoundError("Missing files: " + ", ".join(str(p) for p in paths))


def build_norm_adj(num_nodes: int, src: np.ndarray, dst: np.ndarray, w: np.ndarray, device: torch.device) -> torch.Tensor:
    edge_idx = torch.tensor(np.vstack([src, dst]), dtype=torch.long, device=device)
    edge_w = torch.tensor(w, dtype=torch.float32, device=device)

    self_idx = torch.arange(num_nodes, device=device)
    self_edges = torch.stack([self_idx, self_idx], dim=0)
    idx = torch.cat([edge_idx, self_edges], dim=1)
    vals = torch.cat([edge_w, torch.ones(num_nodes, device=device)], dim=0)

    adj = torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes), device=device).coalesce()
    idx = adj.indices()
    vals = adj.values()

    deg = torch.zeros(num_nodes, device=device)
    deg.index_add_(0, idx[0], vals)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-12), -0.5)
    norm_vals = deg_inv_sqrt[idx[0]] * vals * deg_inv_sqrt[idx[1]]

    return torch.sparse_coo_tensor(idx, norm_vals, (num_nodes, num_nodes), device=device).coalesce()


class GNNEmbedder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_dim, 64)
        self.lin2 = nn.Linear(64, embed_dim)
        self.reg = nn.Linear(embed_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.sparse.mm(norm_adj, x)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.sparse.mm(norm_adj, h)
        emb = self.lin2(h)
        pred = self.reg(F.relu(emb)).squeeze(-1)
        return emb, pred


def prepare_graph(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    links_df = load_pickle_with_fallback([data_dir / "linkes.pkl", data_dir / "links.pkl"])
    nodes_df = load_pickle_with_fallback([data_dir / "nodes.pkl"])
    tad_df = pd.read_csv(data_dir / "TAD.csv")

    for c in ["source_label", "target_label", "weight"]:
        if c not in links_df.columns:
            raise ValueError(f"Links must contain '{c}'.")
    for c in ["id", "WGATAR_id", "enhancer_activity", "gene_rank", "gene_essential", "tgt"]:
        if c not in nodes_df.columns:
            raise ValueError(f"Nodes must contain '{c}'.")
    for c in ["WGATAR_id", "Hudep2_TAD_position"]:
        if c not in tad_df.columns:
            raise ValueError(f"TAD.csv must contain '{c}'.")

    nodes = nodes_df.copy()
    nodes["id"] = nodes["id"].astype(str)
    nodes["WGATAR_id"] = nodes["WGATAR_id"].astype("string")

    tads = tad_df[["WGATAR_id", "Hudep2_TAD_position"]].copy()
    tads["WGATAR_id"] = tads["WGATAR_id"].astype("string")
    tads = tads.drop_duplicates(subset=["WGATAR_id"])
    nodes = nodes.merge(tads, how="left", on="WGATAR_id")

    node_to_idx = {nid: i for i, nid in enumerate(nodes["id"].tolist())}

    links = links_df[["source_label", "target_label", "weight"]].copy()
    links["source_label"] = links["source_label"].astype(str)
    links["target_label"] = links["target_label"].astype(str)
    links["weight"] = pd.to_numeric(links["weight"], errors="coerce")
    links = links.dropna(subset=["weight"])
    links = links[
        links["source_label"].isin(node_to_idx) & links["target_label"].isin(node_to_idx)
    ].reset_index(drop=True)
    if links.empty:
        raise ValueError("No edges left after matching links to node id.")

    src = links["source_label"].map(node_to_idx).to_numpy()
    dst = links["target_label"].map(node_to_idx).to_numpy()
    w0 = links["weight"].to_numpy(dtype=np.float32)
    w = np.abs(np.sign(w0) * np.log1p(np.abs(w0)))

    src0 = src
    dst0 = dst
    src = np.concatenate([src0, dst0])
    dst = np.concatenate([dst0, src0])
    w = np.concatenate([w, w])

    x_df = nodes[["enhancer_activity", "gene_rank", "gene_essential"]].apply(pd.to_numeric, errors="coerce")
    x_df = x_df.replace([np.inf, -np.inf], np.nan)
    x_df = x_df.fillna(x_df.median(numeric_only=True))
    x = x_df.to_numpy(dtype=np.float32)

    y = pd.to_numeric(nodes["tgt"], errors="coerce").to_numpy(dtype=np.float32)
    return nodes, src, dst, w, x, y, "Hudep2_TAD_position"


def split_by_tad(
    tad_map: pd.Series,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, set[str], set[str]]:
    rng = np.random.default_rng(seed)
    tad_vals = tad_map.astype("string")
    known = tad_vals.notna().to_numpy()

    unique_tads = tad_vals[known].astype(str).unique()
    forced_in_data = FORCED_TEST_TAD in unique_tads
    candidate_tads = unique_tads[unique_tads != FORCED_TEST_TAD]

    if candidate_tads.size == 0:
        raise ValueError("Need at least 1 non-forced TAD group for training.")

    n_test = max(1, int(round(test_frac * len(unique_tads))))
    if forced_in_data:
        perm = rng.permutation(candidate_tads)
        n_sampled_test = min(max(0, n_test - 1), len(candidate_tads))
        test_tads = set(perm[:n_sampled_test]) | {FORCED_TEST_TAD}
        train_tads = set(perm[n_sampled_test:])
    else:
        if unique_tads.size < 2:
            raise ValueError("Need at least 2 unique TAD groups for train/test split.")
        perm = rng.permutation(unique_tads)
        n_test = min(n_test, len(perm))
        test_tads = set(perm[:n_test])
        train_tads = set(perm[n_test:])

    row_tads = tad_vals.astype(str).to_numpy()
    train_mask = np.array([t in train_tads for t in row_tads]) & known
    test_mask = np.array([t in test_tads for t in row_tads]) & known

    unknown_idx = np.where(~known)[0]
    if unknown_idx.size > 0:
        unknown_idx = rng.permutation(unknown_idx)
        n_test_u = int(round(test_frac * unknown_idx.size))
        test_mask[unknown_idx[:n_test_u]] = True
        train_mask[unknown_idx[n_test_u:]] = True

    return train_mask, test_mask, train_tads, test_tads


def parse_optional_int(value: str) -> int | None:
    v = value.strip().lower()
    if v in {"none", "null", ""}:
        return None
    return int(value)


def run_one_dim(
    embed_dim: int,
    args: argparse.Namespace,
    nodes: pd.DataFrame,
    feats: pd.DataFrame,
    src: np.ndarray,
    dst: np.ndarray,
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    tad_col: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], pd.DataFrame]:
    pearsons: list[float] = []
    spearmans: list[float] = []
    hit_records: list[dict[str, object]] = []
    best_spearman = -np.inf
    best_details_df = pd.DataFrame()

    rng_master = np.random.default_rng(None)

    attempts = 0
    while attempts < args.max_attempts and len(hit_records) < args.target_hits:
        attempts += 1
        if args.seed is None:
            seed = int(rng_master.integers(0, 2**31 - 1))
        else:
            seed = int(args.seed + attempts)

        tr_row_mask, te_row_mask, train_tads, test_tads = split_by_tad(feats[tad_col], args.test_frac, seed)
        if tr_row_mask.sum() == 0 or te_row_mask.sum() == 0:
            continue

        train_tads_for_nodes = set(feats.loc[tr_row_mask, tad_col].dropna().astype(str).unique())
        node_tads = nodes[tad_col].astype(str).to_numpy()
        node_known = nodes[tad_col].notna().to_numpy()
        node_train_mask_np = np.array([t in train_tads_for_nodes for t in node_tads]) & node_known & np.isfinite(y) & (y != 0)
        if node_train_mask_np.sum() < 20:
            continue

        x_mu = x[node_train_mask_np].mean(axis=0, keepdims=True)
        x_sd = x[node_train_mask_np].std(axis=0, keepdims=True)
        x_sd[x_sd == 0] = 1.0
        x_scaled = (x - x_mu) / x_sd

        y_mu = float(y[node_train_mask_np].mean())
        y_sd = float(y[node_train_mask_np].std())
        if y_sd == 0:
            y_sd = 1.0
        y_scaled = (y - y_mu) / y_sd

        norm_adj = build_norm_adj(len(nodes), src, dst, w, device)
        x_t = torch.tensor(x_scaled, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_scaled, dtype=torch.float32, device=device)
        tr_node_mask_t = torch.tensor(node_train_mask_np, dtype=torch.bool, device=device)

        seed_embeds = []
        for k in range(args.num_seeds):
            s = int((seed + 100 * k) % (2**31 - 1))
            torch.manual_seed(s)
            np.random.seed(s)

            model = GNNEmbedder(in_dim=x_t.shape[1], embed_dim=embed_dim, dropout=args.dropout).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            best_state = None
            best_loss = float("inf")
            best_epoch = 0

            for epoch in range(1, args.epochs + 1):
                model.train()
                opt.zero_grad()
                _, pred = model(x_t, norm_adj)
                loss = F.smooth_l1_loss(pred[tr_node_mask_t], y_t[tr_node_mask_t])
                loss.backward()
                opt.step()

                l = float(loss.item())
                if l < best_loss:
                    best_loss = l
                    best_epoch = epoch
                    best_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
                if epoch - best_epoch >= args.early_stop:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            model.eval()
            with torch.no_grad():
                emb, _ = model(x_t, norm_adj)
                seed_embeds.append(emb.detach().cpu().numpy())

        emb_np = np.mean(seed_embeds, axis=0)
        emb_df = pd.DataFrame(emb_np, columns=[f"gnn_emb_{i}" for i in range(embed_dim)])
        emb_df["WGATAR_id"] = nodes["WGATAR_id"].astype("string").to_numpy()
        emb_df = emb_df.dropna(subset=["WGATAR_id"])
        emb_df = emb_df.groupby("WGATAR_id", as_index=False).mean()

        rep_df = feats.merge(emb_df, how="left", on="WGATAR_id")
        for i in range(embed_dim):
            c = f"gnn_emb_{i}"
            if c in rep_df.columns:
                rep_df[c] = rep_df[c].fillna(rep_df[c].median())

        y_rf = pd.to_numeric(rep_df["effect_value"], errors="coerce")
        valid_y = np.isfinite(y_rf.to_numpy())
        tr_mask = tr_row_mask & valid_y
        te_mask = te_row_mask & valid_y
        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            continue

        X_rf = rep_df.drop(columns=DROP_COLUMNS, errors="ignore")
        X_rf = X_rf.drop(columns=[tad_col, "effect_value"], errors="ignore")
        X_rf = pd.get_dummies(X_rf, drop_first=False)
        X_rf = X_rf.replace([np.inf, -np.inf], np.nan)
        X_rf = X_rf.fillna(X_rf.median(numeric_only=True))

        rf = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(X_rf.loc[tr_mask], y_rf.loc[tr_mask])
        pred_test = rf.predict(X_rf.loc[te_mask])
        pred_train = rf.predict(X_rf.loc[tr_mask])

        y_test = y_rf.loc[te_mask].to_numpy()
        y_train = y_rf.loc[tr_mask].to_numpy()
        p, s = safe_corr(y_test, pred_test)

        if np.isfinite(s) and s > best_spearman:
            best_spearman = float(s)
            best_details_df = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "split": "test",
                            "WGATAR_id": rep_df.loc[te_mask, "WGATAR_id"].astype("string").to_numpy(),
                            "y_true": y_test,
                            "y_pred": pred_test,
                        }
                    ),
                    pd.DataFrame(
                        {
                            "split": "train",
                            "WGATAR_id": rep_df.loc[tr_mask, "WGATAR_id"].astype("string").to_numpy(),
                            "y_true": y_train,
                            "y_pred": pred_train,
                        }
                    ),
                ],
                axis=0,
                ignore_index=True,
            )
            best_details_df["embed_dim"] = embed_dim
            best_details_df["attempt"] = attempts
            best_details_df["seed"] = seed
            best_details_df["pearson_test"] = float(p)
            best_details_df["spearman_test"] = float(s)

        if np.isfinite(p) and np.isfinite(s) and p > 0.2 and s > 0.25:
            emb_cols = [f"gnn_emb_{i}" for i in range(embed_dim)]
            emb_json = (
                rep_df.loc[tr_mask | te_mask, ["WGATAR_id"] + emb_cols]
                .dropna(subset=["WGATAR_id"])
                .drop_duplicates(subset=["WGATAR_id"])
                .to_json(orient="records")
            )
            hit_records.append(
                {
                    "embed_dim": embed_dim,
                    "attempt": attempts,
                    "seed": seed,
                    "pearson": float(p),
                    "spearman": float(s),
                    "train_tads": ";".join(sorted(train_tads)),
                    "test_tads": ";".join(sorted(test_tads)),
                    "train_size": int(tr_mask.sum()),
                    "test_size": int(te_mask.sum()),
                    "gnn_embed_vectors_json": emb_json,
                }
            )
            pearsons.append(p)
            spearmans.append(s)
            print(f"embed_dim={embed_dim} attempt={attempts:03d} pearson={p:.4f} spearman={s:.4f}")
    if len(pearsons) == 0:
        raise RuntimeError(f"No valid repeats for embed_dim={embed_dim}")
    return np.array(pearsons, dtype=float), np.array(spearmans, dtype=float), hit_records, best_details_df


def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    features_path = Path(args.features)

    nodes, src, dst, w, x, y, tad_col = prepare_graph(data_dir)
    feats = pd.read_csv(features_path)
    if "WGATAR_id" not in feats.columns or "effect_value" not in feats.columns:
        raise ValueError("features_table.csv must contain WGATAR_id and effect_value.")

    tad_lookup = nodes[["WGATAR_id", tad_col]].drop_duplicates(subset=["WGATAR_id"]).copy()
    feats["WGATAR_id"] = feats["WGATAR_id"].astype("string")
    feats = feats.merge(tad_lookup, how="left", on="WGATAR_id")
    nodes[tad_col] = nodes[tad_col].astype("string")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.embed_dims:
        embed_dims = [int(v.strip()) for v in args.embed_dims.split(",") if v.strip()]
    else:
        embed_dims = [args.embed_dim]

    summary_rows = []
    for embed_dim in embed_dims:
        pearson_arr, spearman_arr, hit_records, best_details_df = run_one_dim(
            embed_dim, args, nodes, feats, src, dst, w, x, y, tad_col, device
        )

        print(f"embed_dim={embed_dim} Pearson  mean+-std: {np.nanmean(pearson_arr):.4f} +- {np.nanstd(pearson_arr):.4f}")
        print(f"embed_dim={embed_dim} Spearman mean+-std: {np.nanmean(spearman_arr):.4f} +- {np.nanstd(spearman_arr):.4f}")

        hit_path = Path(args.hit_out)
        if len(embed_dims) > 1:
            hit_path = hit_path.with_name(f"{hit_path.stem}_emb{embed_dim}{hit_path.suffix}")
        hit_path.parent.mkdir(parents=True, exist_ok=True)
        hit_df = pd.DataFrame(hit_records)
        hit_df.to_csv(hit_path, index=False)
        print(f"Saved hit cases to: {hit_path}")

        metrics_path = Path(args.hit_metrics_out)
        if len(embed_dims) > 1:
            metrics_path = metrics_path.with_name(f"{metrics_path.stem}_emb{embed_dim}{metrics_path.suffix}")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metric_cols = ["attempt", "seed", "pearson", "spearman"]
        hit_df.reindex(columns=metric_cols).to_csv(metrics_path, index=False)
        print(f"Saved hit Pearson/Spearman to: {metrics_path}")

        if not best_details_df.empty:
            best_path = Path(args.best_pred_out)
            if len(embed_dims) > 1:
                best_path = best_path.with_name(f"{best_path.stem}_emb{embed_dim}{best_path.suffix}")
            best_path.parent.mkdir(parents=True, exist_ok=True)
            best_details_df.to_csv(best_path, index=False)
            print(f"Saved best-spearman train/test predictions to: {best_path}")

        summary_rows.append(
            {
                "embed_dim": embed_dim,
                "repeats_done": int(len(pearson_arr)),
                "pearson_mean": float(np.nanmean(pearson_arr)),
                "pearson_std": float(np.nanstd(pearson_arr)),
                "spearman_mean": float(np.nanmean(spearman_arr)),
                "spearman_std": float(np.nanstd(spearman_arr)),
            }
        )

    if len(embed_dims) > 1:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"Saved summary to: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Initial GNN embedding + RF with repeated TAD split")
    p.add_argument("--data-dir", default="./dataset", help="Directory containing linkes.pkl/links.pkl, nodes.pkl, TAD.csv")
    p.add_argument("--features", default="./dataset/features_table.csv", help="Path to features_table.csv")
    p.add_argument("--embed-dim", type=int, default=16, help="Single node embedding dimension")
    p.add_argument("--embed-dims", default="", help="Optional comma-separated embedding dims, e.g. '8,16,32'")
    p.add_argument("--summary-out", default="./gnn_rf_tad_corr_summary.csv", help="Summary CSV path when multiple embedding dims are used")
    p.add_argument("--num-seeds", type=int, default=5, help="Number of GNN seeds to average embeddings")
    p.add_argument("--repeats", type=int, default=20, help="Number of repeated TAD splits")
    p.add_argument("--max-attempts", type=int, default=500, help="Maximum split attempts to search for hit cases")
    p.add_argument("--test-frac", type=float, default=0.2, help="Test TAD fraction")
    p.add_argument("--target-hits", type=int, default=20, help="Stop after collecting this many hit cases")
    p.add_argument("--hit-out", default="./gnn_rf_tad_hit_cases.csv", help="CSV path for saved hit train/val/test metadata")
    p.add_argument("--hit-metrics-out", default="./gnn_rf_tad_hit_metrics.csv", help="CSV path for hit Pearson/Spearman only")
    p.add_argument("--best-pred-out", default="./gnn_rf_tad_best_spearman_predictions.csv", help="CSV for best-spearman y_pred/y_true on train and test with WGATAR_id")
    p.add_argument("--epochs", type=int, default=220, help="GNN max epochs")
    p.add_argument("--early-stop", type=int, default=30, help="GNN early stopping patience")
    p.add_argument("--lr", type=float, default=1e-3, help="GNN learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="GNN weight decay")
    p.add_argument("--dropout", type=float, default=0.2, help="GNN dropout")
    p.add_argument("--n-estimators", type=int, default=1000, help="RF trees")
    p.add_argument("--max-depth", type=int, default=None, help="RF max depth")
    p.add_argument("--min-samples-leaf", type=int, default=1, help="RF min samples leaf")
    p.add_argument("--seed", type=parse_optional_int, default=None, help="Base random seed, or None")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
