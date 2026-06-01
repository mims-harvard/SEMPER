import argparse
import gzip
import json
import os
import re
from collections import defaultdict
from typing import Dict, Optional, Set, List, Tuple
import warnings

import numpy as np
import pandas as pd
import torch

# Graph-based clustering
import scanpy as sc
import anndata

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
    """Extracts '395' from 'cluster:395', 'CCI_395', or just '395'."""
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
    out = {}
    for go, ont in raw.items():
        if str(ont) in {"BP", "MF", "CC"}:
            out[str(go)] = str(ont)
    return out

def load_ensp2lowestgo(path: str) -> Dict[str, Dict[str, Set[str]]]:
    raw = load_json_maybe_gz(path)
    out = {}
    for k, v in raw.items():
        ensp = norm_ensp(k)
        if not isinstance(v, dict): continue
        out[ensp] = {
            "BP": set(str(x) for x in v.get("BP", []) if str(x).startswith("GO:")),
            "MF": set(str(x) for x in v.get("MF", []) if str(x).startswith("GO:")),
            "CC": set(str(x) for x in v.get("CC", []) if str(x).startswith("GO:")),
        }
    return out

def load_single_cluster_embedding(embed_path: str, label_dict_path: str, target_id: str) -> Dict[str, np.ndarray]:
    """
    Loads SEMPER embeddings and extracts ONLY the specified target cluster.
    Robustly handles KeyErrors by trying Index, ID, String, and Int formats.
    """
    print(f"[INFO] Loading embeddings, searching for Cluster ID: {target_id}...")
    
    # Load data
    embed = torch.load(embed_path, map_location="cpu")
    with open(label_dict_path, "r", encoding="utf-8") as f:
        labels = json.loads(f.read().replace("'", '"'))

    # 1. Find the numerical index for the target cluster
    cci_rows = [c for c in labels["Cell Type"] if str(c).startswith("CCI")]
    target_idx = -1
    
    for idx, c in enumerate(cci_rows):
        if parse_digits(str(c)) == str(target_id):
            target_idx = idx
            break

    if target_idx == -1:
        raise SystemExit(f"[ERROR] Target cluster '{target_id}' not found in labels file.")

    print(f"[INFO] Found target '{target_id}' at list index {target_idx}.")

    # 2. Try all possible keys to find the embedding tensor
    # The .pt file might Key by Index ('319'), by ID ('395'), or as Integers.
    possible_keys = [
        str(target_idx),       # "319" (String Index)
        int(target_idx),       # 319   (Int Index)
        str(target_id),        # "395" (String ID)
        int(target_id),        # 395   (Int ID)
        f"cluster:{target_id}" # "cluster:395"
    ]
    
    found_key = None
    tensor = None
    
    for k in possible_keys:
        if k in embed:
            found_key = k
            tensor = embed[k]
            break
            
    if tensor is None:
        # DEBUG: Print actual keys to help user debug
        actual_keys = list(embed.keys())
        print(f"\n[DEBUG] FAILED. The script tried keys: {possible_keys}")
        print(f"[DEBUG] The actual keys in your .pt file look like: {actual_keys[:5]} ...")
        print(f"[DEBUG] Total keys in .pt: {len(actual_keys)}")
        raise SystemExit(f"[ERROR] Could not locate embedding for Cluster {target_id}.")

    print(f"[INFO] Successfully loaded tensor using key: '{found_key}'")
    
    # 3. Get Protein List for this Cluster
    # We need the protein names corresponding to this cluster
    prot_lists = defaultdict(list)
    for n, ct in zip(labels["Name"], labels["Cell Type"]):
        d = parse_digits(str(ct))
        if d: prot_lists[d].append(n)
        
    target_prots = prot_lists.get(str(target_id), [])
    
    if not target_prots:
        raise SystemExit(f"[ERROR] No proteins found in labels for cluster {target_id}.")

    # 4. Map Proteins to Vectors
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    limit = min(len(target_prots), arr.shape[0])
    
    protein_map = {}
    for i in range(limit):
        full_id = str(target_prots[i])
        ensp = norm_ensp(full_id.split(".", 1)[1] if "." in full_id else full_id)
        protein_map[ensp] = arr[i]
        
    print(f"[INFO] Loaded {len(protein_map)} proteins for Cluster {target_id}.")
    return protein_map
# ----------------- Stats & Analysis -----------------

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
    logs = []
    for i in range(k, imax + 1):
        logs.append(logC(K, i) + logC(N - K, n - i) - denom)
    m = max(logs)
    return float(sum(exp(x - m) for x in logs) * exp(m))

# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser(description="SEMPER Leiden Clustering (Single Cluster Focus)")
    
    # Inputs
    ap.add_argument("--embed-pt", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--target-id", required=True, help="The specific cluster ID digits (e.g., '395')")
    
    # Annotations
    ap.add_argument("--ensp2lowestgo", required=True)
    ap.add_argument("--go2ontology", required=True)
    
    # Params
    ap.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution (default 1.0)")
    ap.add_argument("--n-neighbors", type=int, default=20)
    ap.add_argument("--ontology", choices=["BP", "MF", "CC", "ALL"], default="BP")
    ap.add_argument("--out-prefix", required=True)
    
    args = ap.parse_args()
    
    # 1. LOAD SINGLE CLUSTER DATA
    print(f"--- Analyzing Target Cluster: {args.target_id} ---")
    p2v = load_single_cluster_embedding(args.embed_pt, args.labels, args.target_id)
    
    genes = list(p2v.keys())
    X = np.stack([p2v[g] for g in genes])
    
    # L2 Normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms
    
    N = len(genes)
    
    # 2. LEIDEN CLUSTERING
    print(f"--- Running Leiden (res={args.resolution}) on {N} proteins ---")
    
    adata = anndata.AnnData(X=X)
    adata.obs_names = genes
    
    # Neighbors & Leiden
    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, metric='cosine', use_rep='X')
    try:
        sc.tl.leiden(adata, resolution=args.resolution, key_added='leiden')
    except ImportError:
        print("[ERROR] 'leidenalg' missing. Run: pip install leidenalg")
        return

    # Extract
    cluster_labels = adata.obs['leiden'].values
    clusters = defaultdict(list)
    for g, c_label in zip(genes, cluster_labels):
        clusters[c_label].append(g)
        
    print(f"[RESULT] Detected {len(clusters)} functional modules within Cluster {args.target_id}.")

    # 3. GO ENRICHMENT
    print(f"--- Running GO Enrichment ({args.ontology}) ---")
    
    ensp2lowest = load_ensp2lowestgo(args.ensp2lowestgo)
    go2ont = load_go2ont(args.go2ontology)
    onts = ["BP", "MF", "CC"] if args.ontology == "ALL" else [args.ontology]
    
    # Build Universe Background (Only proteins in Cluster 395)
    go_bg = {ont: defaultdict(set) for ont in onts}
    for g in genes:
        ann = ensp2lowest.get(g)
        if not ann: continue
        for ont in onts:
            for term in ann.get(ont, []):
                if term in go2ont:
                    go_bg[ont][term].add(g)
    
    results = []
    
    for cid, c_genes in clusters.items():
        c_genes_set = set(c_genes)
        n = len(c_genes_set)
        
        if n < 5: continue 
        
        for ont in onts:
            pvals = []
            meta = [] 
            
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
                        "leiden_module": cid,
                        "ontology": ont,
                        "GO_term": term,
                        "pval": p,
                        "fdr": q,
                        "k_module": k,
                        "n_module_size": n,
                        "K_bg": K,
                        "N_bg": N,
                        "enrichment_score": (k/n) / (K/N)
                    })

    # 4. SAVE
    out_csv = f"{args.out_prefix}_c{args.target_id}_leiden.csv"
    df = pd.DataFrame(results)
    
    if not df.empty:
        df = df.sort_values(["leiden_module", "fdr"])
        df.to_csv(out_csv, index=False)
        print(f"[SUCCESS] Saved results to {out_csv}")
    else:
        print("[WARN] No significant enrichment found.")

if __name__ == "__main__":
    main()
