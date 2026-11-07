#!/usr/bin/env python3

import argparse
import csv
import gzip
import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
    if p == 0:   return "p < 1e-300"
    if p < 0.001: return f"p < {p:.1e}"
    return f"p = {p:.4f}"

def load_json_maybe_gz(path: str):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f: return json.load(f)
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def load_ensp2go(path: str) -> Dict[str, Set[str]]:
    raw = load_json_maybe_gz(path)
    return {norm_ensp(k): set(v) if isinstance(v, list) else v
            for k, v in raw.items()}

def load_go2ont(path: str) -> Dict[str, str]:
    raw = load_json_maybe_gz(path)
    return {go: ont for go, ont in raw.items() if ont in {"BP", "MF", "CC"}}

def build_go2proteins(
    ensp2go: Dict[str, Set[str]],
    go2ont: Dict[str, str],
    min_size: int,
    max_size: int,
) -> Dict[str, Dict[str, List[str]]]:
    tmp: Dict[str, Dict[str, Set[str]]] = {
        o: defaultdict(set) for o in ("BP", "MF", "CC")
    }
    for ensp, gos in ensp2go.items():
        for go in gos:
            ont = go2ont.get(go)
            if ont in tmp:
                tmp[ont][go].add(ensp)
    return {
        ont: {go: sorted(ps) for go, ps in tmp[ont].items()
              if min_size <= len(ps) <= max_size}
        for ont in ("BP", "MF", "CC")
    }

def build_mean_embeddings(
    embed_pt: str,
    labels_path: str,
    l2_normalize: bool = True,
) -> Dict[str, np.ndarray]:
    print("[INFO] Loading embeddings...")
    embed = torch.load(embed_pt, map_location="cpu")

    with open(labels_path, "r", encoding="utf-8") as f:
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

    sum_vec: Dict[str, np.ndarray] = {}
    count: Dict[str, int]          = defaultdict(int)

    for k, tensor in embed.items():
        idx = int(k)
        if idx not in index_to_label: continue
        label    = index_to_label[idx]
        prot_list = ct_to_prots.get(label)
        if prot_list is None: continue
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        for i in range(min(arr.shape[0], len(prot_list))):
            raw_id = str(prot_list[i])
            ensp = norm_ensp(raw_id.split(".", 1)[1] if "." in raw_id else raw_id)
            v = arr[i]
            sum_vec[ensp] = sum_vec.get(ensp, np.zeros_like(v, dtype=np.float64)) + v
            count[ensp]  += 1

    result: Dict[str, np.ndarray] = {}
    for ensp, s in sum_vec.items():
        v = (s / count[ensp]).astype(np.float32)
        if l2_normalize:
            nrm = np.linalg.norm(v)
            if nrm > 1e-9: v /= nrm
        result[ensp] = v

    print(f"[INFO] Built mean embeddings for {len(result):,} proteins.")
    return result

# Mean pairwise distance of unit vectors: squared Euclidean distance is 2x cosine distance.
# The paper (Fig 2b-d) plots cosine distance; set by --distance in main().
DIST_SCALE = 0.5

def _dispersion(n: int, sum_vec: np.ndarray) -> float:
    if n < 2: return float("nan")
    mean_vec     = sum_vec / n
    norm_mean_sq = float(np.dot(mean_vec, mean_vec))
    return DIST_SCALE * max(0.0, 2.0 * (1.0 - norm_mean_sq) * (n / (n - 1.0)))

def term_dispersion(proteins: List[str], emb: Dict[str, np.ndarray]) -> Tuple[int, float]:
    s, n = None, 0
    for p in proteins:
        v = emb.get(p)
        if v is None: continue
        s = (v.astype(np.float64) if s is None else s + v)
        n += 1
    if n < 2: return 0, float("nan")
    return n, _dispersion(n, s)

def global_null(emb: Dict[str, np.ndarray]) -> float:
    n = len(emb)
    if n < 2: return 0.0
    s = sum(v.astype(np.float64) for v in emb.values())
    return _dispersion(n, s)

def compute_dispersions(
    emb: Dict[str, np.ndarray],
    go_groups: Dict[str, Dict[str, List[str]]],
    out_prefix: str,
) -> Tuple[float, pd.DataFrame]:
    null_disp = global_null(emb)
    print(f"[INFO] Global null dispersion: {null_disp:.6f}")

    out_csv = f"{out_prefix}_go_dispersions.csv.gz"
    rows = []
    with gzip.open(out_csv, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ontology", "go_term", "n_prots",
                          "term_dispersion", "null_dispersion", "delta"])
        for ont in ("BP", "MF", "CC"):
            for go_term, prots in go_groups[ont].items():
                n, td = term_dispersion(prots, emb)
                if n < 2: continue
                delta = td - null_disp
                writer.writerow([ont, go_term, n,
                                  f"{td:.6f}", f"{null_disp:.6f}", f"{delta:.6f}"])
                rows.append({"ontology": ont, "go_term": go_term, "n_prots": n,
                              "term_dispersion": td, "null_dispersion": null_disp,
                              "delta": delta})

    print(f"[SAVED] {out_csv}")
    return null_disp, pd.DataFrame(rows)

def run_ttest(ont: str, deltas: List[float]) -> None:
    n = len(deltas)
    if n < 2:
        print(f"=== {ont} (n={n}) ===\n  Not enough data.")
        return
    mean_d = np.mean(deltas)
    t_stat, p_two = stats.ttest_1samp(deltas, 0.0)
    p_val = p_two / 2.0 if t_stat < 0 else 1.0
    print(f"\n=== {ont} (n={n}) ===")
    print(f"  Mean delta:       {mean_d:.6f}  (negative = tighter than random)")
    print(f"  T-statistic:      {t_stat:.4f}")
    print(f"  P-value (1-tail): {p_val:.4e}")
    sig = p_val < 0.05 and mean_d < 0
    print(f"  Result: {'SIGNIFICANT' if sig else 'not significant'}")

COLORS      = {"BP": "#a8dadc", "MF": "#1d3557", "CC": "#457b9d"}
LINE_COLORS = {"BP": "#457b9d", "MF": "#0b1d35", "CC": "#1d4e70"}

def plot_histogram(df: pd.DataFrame, ont: str, null_disp: float, out_prefix: str) -> None:
    subset = df[df["ontology"] == ont]
    if len(subset) < 2:
        print(f"[SKIP] {ont}: not enough data.")
        return

    mean_delta    = subset["delta"].mean()
    mean_term     = subset["term_dispersion"].mean()
    t_stat, p_two = stats.ttest_1samp(subset["delta"].values, 0.0)
    p_val         = p_two / 2.0 if t_stat < 0 else 1.0

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=subset, x="term_dispersion",
                 color=COLORS.get(ont, "blue"), bins=50, stat="count",
                 alpha=0.8, linewidth=0, ax=ax)

    ax.axvline(null_disp,   color="#e63946", linestyle="--", linewidth=2.5,
               label=f"Random Null ({null_disp:.2f})")
    ax.axvline(mean_term,   color=LINE_COLORS.get(ont, "blue"), linestyle="-",
               linewidth=2.5, label=f"SEMPER Mean ({mean_term:.2f})")

    ax.text(0.05, 0.95,
            f"Mean Shift: {mean_delta:.4f}\n{format_p_value(p_val)}",
            transform=ax.transAxes, fontsize=12,
            verticalalignment="top", color="#333333")

    ax.set_xlabel("SEMPER embedding distance", fontsize=12, labelpad=10)
    ax.set_ylabel("Pathway counts",            fontsize=12, labelpad=10)
    sns.despine(top=True, right=True, ax=ax)
    ax.tick_params(direction="out", length=4, width=1.0)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(f"{out_prefix}_{ont}.{ext}",
                    dpi=300 if ext == "png" else None,
                    transparent=True, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved {out_prefix}_{ont}.pdf / .png")

def main():
    ap = argparse.ArgumentParser(
        description="Fig 2b–d: GO term dispersion vs random null."
    )
    ap.add_argument("--ensp2go",        required=True,
                    help="ENSP → GO term annotations (.json or .json.gz)")
    ap.add_argument("--go2ontology",    required=True,
                    help="GO term → ontology namespace (.json or .json.gz)")
    ap.add_argument("--embed-pt",       required=True,
                    help="Path to esm_projected.pth")
    ap.add_argument("--labels",         required=True,
                    help="Path to esm_projected_labels.txt")
    ap.add_argument("--out-prefix",     required=True,
                    help="Output prefix for CSV and figures")
    ap.add_argument("--min-term-size",  type=int, default=10,
                    help="Minimum proteins per GO term (default: 10)")
    ap.add_argument("--max-term-size",  type=int, default=500,
                    help="Maximum proteins per GO term (default: 500)")
    ap.add_argument("--no-l2-normalize", action="store_true",
                    help="Skip L2 normalization of embeddings")
    ap.add_argument("--distance", choices=["cosine", "sqeuclidean"], default="cosine",
                    help="Mean pairwise cosine distance (as in the paper) or squared "
                         "Euclidean distance (2x cosine for unit vectors). Default: cosine")
    args = ap.parse_args()
    global DIST_SCALE
    DIST_SCALE = 0.5 if args.distance == "cosine" else 1.0

    # Load
    print("[1] Loading GO annotations...")
    ensp2go  = load_ensp2go(args.ensp2go)
    go2ont   = load_go2ont(args.go2ontology)

    print("[2] Loading SEMPER embeddings...")
    emb = build_mean_embeddings(args.embed_pt, args.labels,
                                l2_normalize=not args.no_l2_normalize)

    print("[3] Building GO term → protein groups...")
    go_groups = build_go2proteins(ensp2go, go2ont,
                                  args.min_term_size, args.max_term_size)
    for ont in ("BP", "MF", "CC"):
        print(f"    {ont}: {len(go_groups[ont]):,} terms")

    print("[4] Computing dispersions...")
    null_disp, df = compute_dispersions(emb, go_groups, args.out_prefix)

    print("\n[5] Statistical tests:")
    for ont in ("BP", "MF", "CC"):
        run_ttest(ont, df[df["ontology"] == ont]["delta"].tolist())

    print("\n[6] Plotting histograms...")
    for ont in ("BP", "MF", "CC"):
        plot_histogram(df, ont, null_disp, args.out_prefix)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
