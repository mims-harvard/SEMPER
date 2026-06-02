import argparse
import gzip
import json
import os
import re
from collections import defaultdict
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd
import torch
import scanpy as sc
import anndata
import matplotlib.pyplot as plt

# ----------------- Text & ID Helpers -----------------
_ZWS = "\u200b\u200c\u200d\ufeff"

def _clean(s: Optional[str]) -> str:
    if s is None: return ""
    return str(s).strip().translate({ord(c): None for c in _ZWS})

def norm_ensp(x: str) -> str:
    s = _clean(x)
    if s.startswith("9606."): s = s[5:]
    m = re.match(r"^(ENSP\d+)(?:\.\d+)?$", s)
    return m.group(1) if m else s

def parse_digits(label: str) -> str:
    s = _clean(label)
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else ""

# ----------------- Loaders -----------------
def load_json_maybe_gz(path: str):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f: return json.load(f)
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def load_go2ont(path: str) -> Dict[str, str]:
    raw = load_json_maybe_gz(path)
    return {str(go): str(ont) for go, ont in raw.items() if str(ont) in {"BP", "MF", "CC"}}

def load_ensp2lowestgo(path: str) -> Dict[str, Dict[str, Set[str]]]:
    raw = load_json_maybe_gz(path)
    out = {}
    for k, v in raw.items():
        ensp = norm_ensp(k)
        if not isinstance(v, dict): continue
        out[ensp] = {
            ont: set(str(x) for x in v.get(ont, []) if str(x).startswith("GO:"))
            for ont in ["BP", "MF", "CC"]
        }
    return out

def load_single_cluster_embedding(embed_path: str, label_dict_path: str, target_id: str):
    print(f"[INFO] Loading embeddings for Cluster ID: {target_id}...")
    embed = torch.load(embed_path, map_location="cpu")
    with open(label_dict_path, "r", encoding="utf-8") as f:
        labels = json.loads(f.read().replace("'", '"'))

    cci_rows = [c for c in labels["Cell Type"] if str(c).startswith("CCI")]
    target_idx = next((i for i, c in enumerate(cci_rows) if parse_digits(str(c)) == str(target_id)), -1)
    if target_idx == -1: raise SystemExit(f"[ERROR] Target cluster '{target_id}' not found.")

    possible_keys = [str(target_idx), int(target_idx), str(target_id), int(target_id), f"cluster:{target_id}"]
    tensor = next((embed[k] for k in possible_keys if k in embed), None)
    if tensor is None: raise SystemExit(f"[ERROR] Could not locate tensor for {target_id}. Tried keys: {possible_keys}")

    prot_lists = defaultdict(list)
    for n, ct in zip(labels["Name"], labels["Cell Type"]):
        d = parse_digits(str(ct))
        if d: prot_lists[d].append(n)
        
    target_prots = prot_lists.get(str(target_id), [])
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    limit = min(len(target_prots), arr.shape[0])
    
    protein_map = {}
    for i in range(limit):
        full_id = str(target_prots[i])
        ensp = norm_ensp(full_id.split(".", 1)[1] if "." in full_id else full_id)
        protein_map[ensp] = arr[i]
        
    return protein_map

# ----------------- Stats -----------------
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0: return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out

def hypergeom_pval_righttail(k: int, K: int, n: int, N: int) -> float:
    from math import lgamma, exp
    if k <= 0: return 1.0
    def logC(a, b): return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)
    denom = logC(N, n)
    imax = min(n, K)
    logs = [logC(K, i) + logC(N - K, n - i) - denom for i in range(k, imax + 1)]
    m = max(logs)
    return float(sum(exp(x - m) for x in logs) * exp(m))

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-pt", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--target-id", required=True)
    ap.add_argument("--ensp2lowestgo", required=True)
    ap.add_argument("--go2ontology", required=True)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--n-neighbors", type=int, default=20)
    ap.add_argument("--ontology", choices=["BP", "MF", "CC", "ALL"], default="BP")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    
    # 1. Load & L2 Normalize
    p2v = load_single_cluster_embedding(args.embed_pt, args.labels, args.target_id)
    genes = list(p2v.keys())
    X = np.stack([p2v[g] for g in genes])
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms
    
    # 2. Pipeline: Neighbors -> Leiden -> UMAP
    print(f"--- Running Unified Pipeline (Res: {args.resolution}) ---")
    adata = anndata.AnnData(X=X)
    adata.obs_names = genes
    
    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, metric='cosine', use_rep='X')
    sc.tl.leiden(adata, resolution=args.resolution, key_added='leiden')
    sc.tl.umap(adata)
    
    # 3. Export Assignments (Proteins -> Cluster + UMAP Coords)
    df_assign = pd.DataFrame({
        "Protein_ENSP": genes,
        "Leiden_Cluster": adata.obs['leiden'].values,
        "UMAP_1": adata.obsm['X_umap'][:, 0],
        "UMAP_2": adata.obsm['X_umap'][:, 1]
    })
    df_assign.to_csv(f"{args.out_prefix}_cluster_assignments.csv", index=False)
    print(f"[SUCCESS] Saved protein-level cluster assignments.")

    # 4. Plotting
    import matplotlib.pyplot as plt

    # Set Nature-style fonts
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    sc.set_figure_params(dpi=300, frameon=False, figsize=(6, 6))
    
    # Generate the base scanpy plot
    # - legend_loc=None: Removes the giant numbers stamped over the data
    # - title="": Removes the default title (so you can put it in the figure caption)
    # - size & alpha: Tweaks the dots to look like a continuous cloud
    ax = sc.pl.umap(
        adata, 
        color=['leiden'], 
        legend_loc=None, 
        title="", 
        size=15,       # Adjust this if dots are too small/large (scanpy scaling varies by n_cells)
        alpha=0.8, 
        show=False,
        frameon=False,
        edgecolors='none'

    )
    
    # Scanpy can return a single axis or a list/dict depending on the version and inputs
    if isinstance(ax, dict): ax = ax['leiden']
    elif isinstance(ax, list): ax = ax[0]
        
    # Ensure axes and default scanpy labels are completely off
    ax.axis('off')
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    # --- Add the CellWhisperer "L" Scalebar ---
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    x_range = xmax - xmin
    y_range = ymax - ymin
    
    # Place the origin of the arrows in the bottom left
    orig_x = xmin + x_range * 0.05
    orig_y = ymin + y_range * 0.05
    
    # Arrow lengths
    len_x = x_range * 0.15
    len_y = y_range * 0.15
    
    # Draw the arrows
    ax.annotate('', xy=(orig_x + len_x, orig_y), xytext=(orig_x, orig_y),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))
    ax.annotate('', xy=(orig_x, orig_y + len_y), xytext=(orig_x, orig_y),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))
                
    # Add the "UMAP 1" and "UMAP 2" text
    ax.text(orig_x + len_x/2, orig_y - y_range*0.03, 'UMAP 1', 
            ha='center', va='top', fontsize=12, fontweight='bold')
    ax.text(orig_x - x_range*0.03, orig_y + len_y/2, 'UMAP 2', 
            ha='right', va='center', rotation=90, fontsize=12, fontweight='bold')

    # Save with transparent=True for easy Illustrator/PowerPoint assembly
    plt.savefig(f"{args.out_prefix}_umap.png", bbox_inches='tight', transparent=True)
    plt.savefig(f"{args.out_prefix}_umap.pdf", bbox_inches='tight', transparent=True)
    print(f"[SUCCESS] Saved Nature-style UMAP plots.")
    plt.close()

    # 5. GO Enrichment
    print(f"--- Running GO Enrichment ({args.ontology}) ---")
    ensp2lowest = load_ensp2lowestgo(args.ensp2lowestgo)
    go2ont = load_go2ont(args.go2ontology)
    onts = ["BP", "MF", "CC"] if args.ontology == "ALL" else [args.ontology]
    
    # Universe Background
    go_bg = {ont: defaultdict(set) for ont in onts}
    for g in genes:
        ann = ensp2lowest.get(g)
        if not ann: continue
        for ont in onts:
            for term in ann.get(ont, []):
                if term in go2ont: go_bg[ont][term].add(g)
    
    clusters = defaultdict(list)
    for g, c_label in zip(genes, adata.obs['leiden'].values):
        clusters[c_label].append(g)

    N = len(genes)
    results = []
    
    for cid, c_genes in clusters.items():
        c_genes_set = set(c_genes)
        n = len(c_genes_set)
        if n < 5: continue 
        
        for ont in onts:
            pvals, meta = [], []
            for term, bg_set in go_bg[ont].items():
                K = len(bg_set)
                k = len(c_genes_set & bg_set)
                if k < 2: continue
                
                p = hypergeom_pval_righttail(k, K, n, N)
                pvals.append(p)
                meta.append((term, k, K))
            
            if not pvals: continue
            qvals = bh_fdr(np.array(pvals))
            
            for (term, k, K), p, q in zip(meta, pvals, qvals):
                if q < 0.10: 
                    results.append({
                        "target_cluster": args.target_id,
                        "module": cid,
                        "ontology": ont,
                        "GO_term": term,
                        "pval": p, "fdr": q,
                        "k_module": k, "n_module_size": n,
                        "K_bg": K, "N_bg": N,
                        "enrichment_score": (k/n) / (K/N) if K > 0 else 0
                    })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(["module", "fdr"])
        df.to_csv(f"{args.out_prefix}_enrichment.csv", index=False)
        print(f"[SUCCESS] Saved GO enrichment to {args.out_prefix}_enrichment.csv")
    else:
        print("[WARN] No significant enrichment found.")

if __name__ == "__main__":
    main()
