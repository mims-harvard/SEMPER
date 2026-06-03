#!/usr/bin/env python3

import argparse
import json
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

_ZWS = "\u200b\u200c\u200d\ufeff"

def _clean(s):
    if s is None: return ""
    return str(s).strip().translate({ord(c): None for c in _ZWS})

def parse_digits(s: str) -> str:
    m = re.search(r"(\d+)", str(s))
    return m.group(1) if m else ""

def get_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

def load_data(embed_path, label_path):
    print("[INFO] Loading embeddings...")
    embed = torch.load(embed_path, map_location="cpu")
    with open(label_path, "r") as f:
        labels = json.loads(f.read().replace("'", '"'))

    df = pd.DataFrame({"target": labels["Name"], "cell type": labels["Cell Type"]})
    mask = ~df["cell type"].astype(str).str.startswith(("CCI", "BTO", "Sanity"))
    df = df[mask].copy()
    df["cid_str"] = df["cell type"].apply(lambda x: parse_digits(str(x)))
    df = df[df["cid_str"] != ""]

    unique_cids = sorted(df["cid_str"].unique())
    print(f"[INFO] Found {len(unique_cids)} unique clusters.")
    cid_to_prots = df.groupby("cid_str")["target"].apply(list).to_dict()

    per_ct_emb = {}
    for k in tqdm(range(len(embed)), desc="Mapping clusters"):
        if k >= len(unique_cids): continue
        cid = unique_cids[k]
        prot_list = cid_to_prots.get(cid)
        tensor = embed.get(k)
        if tensor is None or prot_list is None: continue
        if tensor.shape[0] != len(prot_list): continue
        tensor = F.normalize(tensor.float(), p=2, dim=1)
        per_ct_emb[f"cluster:{cid}"] = tensor

    print(f"[INFO] Successfully mapped {len(per_ct_emb)} clusters.")
    return per_ct_emb

def get_pairwise_scores(tensor, n_pairs=100):
    n = tensor.shape[0]
    if n < 2: return []
    idx_a = torch.randint(0, n, (n_pairs,))
    idx_b = torch.randint(0, n, (n_pairs,))
    mask = idx_a != idx_b
    if mask.sum() == 0: return []
    return (tensor[idx_a[mask]] * tensor[idx_b[mask]]).sum(dim=1).tolist()

def get_cross_scores(tensor_a, tensor_b, n_pairs=100):
    na, nb = tensor_a.shape[0], tensor_b.shape[0]
    if na < 1 or nb < 1: return []
    idx_a = torch.randint(0, na, (n_pairs,))
    idx_b = torch.randint(0, nb, (n_pairs,))
    return (tensor_a[idx_a] * tensor_b[idx_b]).sum(dim=1).tolist()

def sample_pairwise(per_ct_emb, ct_to_super, n_pairs):
    super_to_cts = {}
    for ct in per_ct_emb:
        if ct in ct_to_super:
            sup = ct_to_super[ct]
            super_to_cts.setdefault(sup, []).append(ct)

    all_cts = list(per_ct_emb.keys())
    results = []
    print("[INFO] Sampling pairwise distributions...")
    for sup, ct_list in tqdm(super_to_cts.items()):
        for ct in ct_list:
            tensor_self = per_ct_emb[ct]
            for s in get_pairwise_scores(tensor_self, n_pairs):
                results.append({"Supercluster": sup, "Similarity": s,
                                 "Type": "Same Cell Type", "n_types": len(ct_list)})
            stranger = random.choice(all_cts)
            while stranger == ct:
                stranger = random.choice(all_cts)
            for s in get_cross_scores(tensor_self, per_ct_emb[stranger], n_pairs):
                results.append({"Supercluster": sup, "Similarity": s,
                                 "Type": "Diff Cell Type", "n_types": len(ct_list)})
    return pd.DataFrame(results)

def compute_stats(df):
    print("[INFO] Computing statistics...")
    superclusters = df["Supercluster"].unique()
    raw_pvals, stats_temp = [], []

    for sup in superclusters:
        mask = df["Supercluster"] == sup
        same = df[mask & (df["Type"] == "Same Cell Type")]["Similarity"].values
        diff = df[mask & (df["Type"] == "Diff Cell Type")]["Similarity"].values
        if len(same) == 0 or len(diff) == 0: continue

        try:
            stat, p = mannwhitneyu(same, diff, alternative="greater")
        except ValueError:
            stat, p = np.nan, 1.0

        raw_pvals.append(p)
        n_types = int(df[mask]["n_types"].iloc[0])
        stats_temp.append({
            "Supercluster": sup,
            "n_types":      n_types,
            "mean_same":    np.mean(same),
            "ci_same":      1.96 * np.std(same) / np.sqrt(len(same)),
            "mean_diff":    np.mean(diff),
            "ci_diff":      1.96 * np.std(diff) / np.sqrt(len(diff)),
            "shift":        np.mean(same) - np.mean(diff),
            "U_statistic":  stat,
            "p_raw":        p,
        })

    _, padj, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")
    for i, row in enumerate(stats_temp):
        row["p_adj"]  = padj[i]
        row["stars"]  = get_stars(padj[i])

    agg = pd.DataFrame(stats_temp).sort_values("shift", ascending=True).reset_index(drop=True)
    return agg


def plot_dumbbell(agg_df, out_prefix):
    color_inter = "#A0A0A0"
    color_intra = "#1f77b4"
    y_range = np.arange(1, len(agg_df) + 1)

    fig, ax = plt.subplots(figsize=(6, 12))
    ax.set_facecolor("white")

    ax.hlines(y_range, agg_df["mean_diff"], agg_df["mean_same"],
              color="grey", alpha=0.4, linewidth=1.5, zorder=1)
    ax.errorbar(agg_df["mean_diff"], y_range, xerr=agg_df["ci_diff"],
                fmt="none", ecolor=color_inter, elinewidth=1.5, capsize=0, zorder=2)
    ax.errorbar(agg_df["mean_same"], y_range, xerr=agg_df["ci_same"],
                fmt="none", ecolor=color_intra, elinewidth=1.5, capsize=0, zorder=2)
    ax.scatter(agg_df["mean_diff"], y_range, color=color_inter,
               s=50, edgecolors="none", zorder=3, label="Inter-cell-type")
    ax.scatter(agg_df["mean_same"], y_range, color=color_intra,
               s=50, edgecolors="none", zorder=3, label="Intra-cell-type")

    sns.despine(ax=ax, left=True, top=True, right=True)
    ax.set_yticks(y_range)
    ax.set_yticklabels(
        [f"{r['Supercluster']} (n={r['n_types']}) {r['stars']}"
         for _, r in agg_df.iterrows()],
        fontsize=10
    )
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0),
              frameon=False, fontsize=10, markerscale=1.2)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(f"{out_prefix}.{ext}",
                    dpi=300 if ext == "png" else None,
                    transparent=True, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved {out_prefix}.pdf / .png")


def main():
    ap = argparse.ArgumentParser(
        description="Fig 2a: intra- vs. inter-cell-type pairwise cosine similarity."
    )
    ap.add_argument("--embed-pt",          required=True,
                    help="Path to esm_projected.pth")
    ap.add_argument("--labels",            required=True,
                    help="Path to esm_projected_labels.txt")
    ap.add_argument("--supercluster-xlsx", required=True,
                    help="Siletti et al. Table S3 (.xlsx or .csv)")
    ap.add_argument("--out-prefix",        required=True,
                    help="Output prefix for CSV and figures (e.g. outputs/evaluate/fig2_embedding_space/fig2a)")
    ap.add_argument("--n-pairs",           type=int, default=100000,
                    help="Pairs to sample per cell type (default: 100000)")
    ap.add_argument("--seed",              type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load
    per_ct_emb = load_data(args.embed_pt, args.labels)

    # Supercluster map
    if args.supercluster_xlsx.endswith(".csv"):
        df_sup = pd.read_csv(args.supercluster_xlsx)
    else:
        df_sup = pd.read_excel(args.supercluster_xlsx)
    ct_to_super = {
        f"cluster:{parse_digits(str(r['Cluster ID']))}": _clean(r["Supercluster"])
        for _, r in df_sup.iterrows()
        if parse_digits(str(r["Cluster ID"])) and _clean(r["Supercluster"])
    }

    # Sample
    df_scores = sample_pairwise(per_ct_emb, ct_to_super, args.n_pairs)
    df_scores.to_csv(f"{args.out_prefix}_raw_scores.csv", index=False)
    print(f"[SAVED] {args.out_prefix}_raw_scores.csv")

    # Stats
    agg_df = compute_stats(df_scores)
    agg_df.to_csv(f"{args.out_prefix}_aggregate_stats.csv", index=False)
    print(f"[SAVED] {args.out_prefix}_aggregate_stats.csv")

    # Plot
    plot_dumbbell(agg_df, args.out_prefix)


if __name__ == "__main__":
    main()
