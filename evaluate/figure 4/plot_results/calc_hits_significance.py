#!/usr/bin/env python3
import argparse
import os
import re
from glob import glob
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# -----------------------------
# Configuration
# -----------------------------
MODEL_RENAME = {
    "SEMPER_ESMCLIP": "SEMPER",
    "GNN_PINNACLE": "PINNACLE",
    "Graphlets_ORCA_CT": "ORCA-CT",
    "PPI_GT_Encoder": "PPI-GT",
    # Global models are ignored for this contextual stat test
}

def expand_paths(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if any(ch in p for ch in ["*", "?", "["]):
            out.extend(sorted(glob(p)))
        else:
            out.append(p)
    return list(dict.fromkeys(out))

def pick_first_present(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""

def main():
    ap = argparse.ArgumentParser(description="Calculate Paired Wilcoxon P-values for Hits@K")
    ap.add_argument("--trials_csv", nargs="+", required=True)
    ap.add_argument("--semper_model", default="SEMPER")
    ap.add_argument("--baselines", nargs="+", default=["PINNACLE", "PPI-GT", "ORCA-CT"])
    ap.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 25])
    args = ap.parse_args()

    paths = expand_paths(args.trials_csv)
    if not paths:
        raise SystemExit("[ERROR] No files matched --trials_csv inputs.")

    # 1. Load and Standardize Data
    frames = []
    for p in paths:
        if os.path.exists(p):
            frames.append(pd.read_csv(p))

    if not frames:
        raise SystemExit("[ERROR] No readable CSVs found.")

    df = pd.concat(frames, axis=0, ignore_index=True)

    model_col = pick_first_present(df, ["model", "Model"])
    ct_col = pick_first_present(df, ["cluster", "cell_type"])
    k_col = pick_first_present(df, ["k", "K"])
    hits_col = pick_first_present(df, ["hits_at_k", "hits@k", "hits"])

    df = df.rename(columns={model_col: "model", ct_col: "cluster", k_col: "k", hits_col: "hits_at_k"})
    df["model"] = df["model"].replace(MODEL_RENAME)
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df["hits_at_k"] = pd.to_numeric(df["hits_at_k"], errors="coerce")
    df = df.dropna(subset=["k", "hits_at_k"]).copy()

    # 2. Compute Mean Hits@K per Cluster
    # This averages across the 100 bootstrapped trials for each model/cluster combo
    df_cluster_means = df.groupby(["model", "cluster", "k"], as_index=False)["hits_at_k"].mean()

    # 3. Statistical Testing Loop
    print("=" * 60)
    print(" STATISTICAL SIGNIFICANCE: PAIRED WILCOXON (SEMPER > BASELINE)")
    print("=" * 60)

    for k in args.k_values:
        print(f"\n--- Metrics for Top-{k} Hits ---")
        df_k = df_cluster_means[df_cluster_means["k"] == k]
        
        semper_data = df_k[df_k["model"] == args.semper_model].set_index("cluster")
        
        if semper_data.empty:
            print(f"[Warning] No data found for {args.semper_model} at k={k}")
            continue

        for base in args.baselines:
            base_data = df_k[df_k["model"] == base].set_index("cluster")
            
            # Inner join ensures we only compare exactly matched cell type clusters
            merged = semper_data.join(base_data, lsuffix="_semper", rsuffix="_base", how="inner")
            n_pairs = len(merged)

            if n_pairs < 5:
                print(f"  {base:<12} : Not enough paired clusters (N={n_pairs})")
                continue

            # One-sided test: Alternative hypothesis is SEMPER is GREATER than baseline
            stat, p_val = wilcoxon(merged["hits_at_k_semper"], merged["hits_at_k_base"], alternative="greater")
            
            # Formatting output
            star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  vs {base:<10} | Paired Clusters: {n_pairs:<4} | P-value: {p_val:.2e} {star}")

if __name__ == "__main__":
    main()
