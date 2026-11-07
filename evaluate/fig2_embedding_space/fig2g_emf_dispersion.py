#!/usr/bin/env python3

import argparse
import json
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu


_ZWS = "\u200b\u200c\u200d\ufeff"

def _clean(s: Optional[str]) -> str:
    if s is None: return ""
    return str(s).strip().translate({ord(c): None for c in _ZWS})

def norm_ensp(x: str) -> str:
    s = _clean(x)
    if s.startswith("9606."): s = s[5:]
    m = re.match(r"^(ENSP\d+)(?:\.\d+)?$", s)
    return m.group(1) if m else s

def parse_celltype_id_digits(label: str) -> str:
    s = _clean(label)
    m = (re.match(r"^(\d+)", s) or
         re.search(r"cluster:(\d+)", s) or
         re.search(r"(\d+)", s))
    return m.group(1) if m else ""

def format_p_value(p: float) -> str:
    if p == 0:    return "p < 1e-300"
    if p < 0.001: return f"p < {p:.1e}"
    return f"p = {p:.4f}"

def load_celltype_embeddings(
    embed_path: str,
    label_dict_path: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    print("[INFO] Loading embeddings...")
    embed = torch.load(embed_path, map_location="cpu")
    with open(label_dict_path, "r") as f:
        labels = json.loads(f.read().replace("'", '"'))

    cci_rows = [c for c in labels["Cell Type"] if str(c).startswith("CCI")]
    index_to_label: Dict[int, str] = {}
    for idx, c in enumerate(cci_rows):
        c = str(c)
        label = c.split("CCI_", 1)[1] if "CCI_" in c else (
            re.search(r"cluster:\d+", c).group(0)
            if re.search(r"cluster:\d+", c)
            else f"cluster:{parse_celltype_id_digits(c)}"
        )
        if not label.startswith("cluster:"):
            label = f"cluster:{parse_celltype_id_digits(label)}"
        index_to_label[idx] = label

    protein_names, protein_celltypes = [], []
    for c, p in zip(labels["Cell Type"], labels["Name"]):
        cs = str(c)
        if cs.startswith(("BTO", "CCI", "Sanity")): continue
        protein_names.append(p)
        protein_celltypes.append(cs)

    ct_to_prots = (
        pd.DataFrame({"target": protein_names, "cell type": protein_celltypes})
        .groupby("cell type")["target"].apply(list).to_dict()
    )

    per_ct: Dict[str, Dict[str, np.ndarray]] = {}
    for k, tensor in embed.items():
        idx   = int(k)
        if idx not in index_to_label: continue
        label = index_to_label[idx]
        prots = ct_to_prots.get(label)
        if prots is None or tensor.shape[0] != len(prots): continue
        d = {}
        for i, raw_id in enumerate(prots):
            ensp = norm_ensp(raw_id.split(".", 1)[1] if "." in raw_id else raw_id)
            d[ensp] = tensor[i].detach().cpu().numpy().astype(np.float32)
        per_ct[label] = d

    print(f"[INFO] Loaded {len(per_ct)} clusters.")
    return per_ct

def centroid_mean_l2(X: np.ndarray) -> float:
    if X.shape[0] < 1: return float("nan")
    mu = X.mean(axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(X - mu, axis=1)))

def mean_pairwise_l2(X: np.ndarray) -> float:
    k = X.shape[0]
    if k < 2: return float("nan")
    G    = X @ X.T
    sq   = np.sum(X * X, axis=1, keepdims=True)
    dist = np.sqrt(np.maximum(sq + sq.T - 2.0 * G, 0.0))
    iu   = np.triu_indices(k, 1)
    return float(np.mean(dist[iu]))

def compute_protein_dispersions(
    per_ct: Dict[str, Dict[str, np.ndarray]],
    min_contexts: int,
    l2_normalize: bool,
) -> pd.DataFrame:
    # Invert: protein -> [(ct, vec), ...]
    prot2ctx: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for ct, id2vec in per_ct.items():
        for ensp, v in id2vec.items():
            prot2ctx.setdefault(ensp, []).append((ct, v))

    rows = []
    for ensp, ctx_list in prot2ctx.items():
        k = len(ctx_list)
        if k < min_contexts: continue
        X = np.stack([v for _, v in ctx_list]).astype(np.float32)
        if l2_normalize:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = X / norms
        rows.append({
            "protein":              ensp,
            "n_contexts":           k,
            "disp_centroid_mean_l2":  centroid_mean_l2(X),
            "disp_mean_pairwise_l2":  mean_pairwise_l2(X),
        })
        if len(rows) % 5000 == 0:
            print(f"[INFO] Scored {len(rows)} proteins...", end="\r", file=sys.stderr)

    print(f"\n[INFO] Computed dispersion for {len(rows):,} proteins.")
    return pd.DataFrame(rows)


def load_emf_genes(xlsx_path: str) -> List[str]:
    df = pd.read_excel(xlsx_path)
    needed = {"symbol", "alias"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Expected columns {needed}, missing: {missing}. "
                         f"Found: {df.columns.tolist()}")

    genes = set()
    for _, row in df.iterrows():
        sym = str(row["symbol"]).strip() if pd.notnull(row["symbol"]) else ""
        if sym: genes.add(sym.upper())
        alias = row.get("alias", "")
        if pd.notnull(alias) and str(alias).strip():
            for a in re.split(r"[;,|/]\s*|\s*,\s*", str(alias)):
                a = a.strip()
                if a: genes.add(a.upper())

    print(f"[INFO] Loaded {len(df)} MoonDB EMF entries → "
          f"{len(genes)} unique gene symbols (primary + aliases).")
    return sorted(genes)


def map_genes_to_ensp(
    gene_symbols: List[str],
    ensp_json: str,
) -> Dict[str, List[str]]:
    with open(ensp_json) as f:
        ensp_to_gene = json.load(f)

    gene_to_ensps: Dict[str, List[str]] = defaultdict(list)
    for ensp, gene in ensp_to_gene.items():
        gene_to_ensps[str(gene).upper()].append(norm_ensp(ensp))

    mapping = {}
    for gene in gene_symbols:
        ensps = gene_to_ensps.get(gene.upper(), [])
        if ensps:
            mapping[gene] = ensps
    print(f"[INFO] Mapped {len(mapping)}/{len(gene_symbols)} EMF gene symbols to ENSP IDs.")
    return mapping

def plot_histogram(
    background_vals: np.ndarray,
    highlight_vals:  np.ndarray,
    metric_name: str,
    out_prefix:  str,
) -> None:
    mean_bg = np.mean(background_vals)
    mean_hl = np.mean(highlight_vals)

    if len(highlight_vals) > 1 and len(background_vals) > 1:
        stat, p_val = mannwhitneyu(highlight_vals, background_vals, alternative="two-sided")
        p_text = format_p_value(p_val)
    else:
        p_text = "p = N/A"

    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)
    ax.set_facecolor("white")

    sns.histplot(background_vals, stat="density", color="#A0A0A0", alpha=0.4,
                 edgecolor="none", label=f"Null (N={len(background_vals):,})",
                 bins=50, ax=ax)
    sns.histplot(highlight_vals,  stat="density", color="#D62728", alpha=0.7,
                 edgecolor="none", label=f"EMF (N={len(highlight_vals):,})",
                 bins=30, ax=ax)

    ax.axvline(mean_bg, color="#555555", linestyle="--", linewidth=1.5)
    ax.axvline(mean_hl, color="#D62728",  linestyle="--", linewidth=1.5)

    sns.despine(ax=ax)
    ax.set_xlabel("Dispersion", fontsize=10)
    ax.set_ylabel("Density",    fontsize=10)
    ax.tick_params(labelsize=8)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(8)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(f"{out_prefix}_histogram.{ext}", transparent=True,
                    bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved {out_prefix}_histogram.pdf / .png")
    print(f"  Mean (null): {mean_bg:.4e}  |  Mean (EMF): {mean_hl:.4e}  |  {p_text}")


def main():
    ap = argparse.ArgumentParser(
        description="Fig 2g: centroid dispersion for EMF vs. background proteins."
    )
    ap.add_argument("--embed-pt",      required=True,
                    help="Path to esm_projected.pth")
    ap.add_argument("--labels",        required=True,
                    help="Path to esm_projected_labels.txt")
    ap.add_argument("--ensp-json",     required=True,
                    help="Path to ensp_to_gene.json")
    ap.add_argument("--emf-xlsx",      required=True,
                    help="Path to moonlightproteins_moondb_allemf_idmap.xlsx (gene symbol column: symbol). "
                         "Included in data/moonlightproteins_moondb_allemf_idmap.xlsx. "
                         "Source: MoonDB (https://moondb.hb.univ-amu.fr)")
    ap.add_argument("--out-prefix",    required=True,
                    help="Output prefix for CSV and figures")
    ap.add_argument("--metric",        default="disp_centroid_mean_l2",
                    choices=["disp_centroid_mean_l2", "disp_mean_pairwise_l2"],
                    help="Dispersion metric to plot (default: disp_centroid_mean_l2)")
    ap.add_argument("--min-contexts",  type=int, default=1,
                    help="Min cell type contexts to score a protein (default: 1)")
    ap.add_argument("--exclude-k1",    action="store_true",
                    help="Exclude proteins with only 1 context from the comparison")
    ap.add_argument("--top-k",         type=int, default=200,
                    help="Number of top-dispersion outliers to save (default: 200)")
    ap.add_argument("--no-l2",         action="store_true",
                    help="Skip L2 normalization before computing distances")
    args = ap.parse_args()

    # Load
    per_ct = load_celltype_embeddings(args.embed_pt, args.labels)
    df = compute_protein_dispersions(per_ct, args.min_contexts, not args.no_l2)
    df.sort_values(args.metric, ascending=False, inplace=True)
    df.to_csv(f"{args.out_prefix}_protein_dispersion.csv", index=False)
    df.head(args.top_k).to_csv(f"{args.out_prefix}_top{args.top_k}_outliers.csv", index=False)
    print(f"[SAVED] {args.out_prefix}_protein_dispersion.csv")

    # Optionally exclude k=1
    df_rank = df[df["n_contexts"] >= 2].copy() if args.exclude_k1 else df.copy()

    emf_genes  = load_emf_genes(args.emf_xlsx)
    gene_map   = map_genes_to_ensp(emf_genes, args.ensp_json)
    emf_ensps  = set(e for ensps in gene_map.values() for e in ensps)

    emf_vals  = df_rank.loc[df_rank["protein"].isin(emf_ensps),  args.metric].dropna().values
    bg_vals   = df_rank.loc[~df_rank["protein"].isin(emf_ensps), args.metric].dropna().values

    print(f"\n[INFO] EMF proteins in dispersion distribution: {len(emf_vals):,}")
    print(f"[INFO] Background proteins:                      {len(bg_vals):,}")

    if len(emf_vals) == 0:
        print("[WARN] No EMF proteins matched. Check --emf-genes and --ensp-json.")
        return

    plot_histogram(bg_vals, emf_vals, args.metric, args.out_prefix)
    print("\n[DONE]")


if __name__ == "__main__":
    main()
