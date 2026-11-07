#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path
from glob import glob
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as font_manager


plt.rcParams['pdf.fonttype'] = 42

# -----------------------------
# Configuration & Aesthetics
# -----------------------------
MODEL_RENAME = {
    "SEMPER_ESMCLIP": "SEMPER",
    "ProTrek_seq_mean": "ProTrek Seq",
    "ProTrek_struct_mean": "ProTrek Struct",
    "ESMC_600M": "ESM-C",
    "GNN_PINNACLE": "PINNACLE",
    "Graphlets_ORCA_CT": "ORCA-CT",
    "PPI_GT_Encoder": "PPI-GT",
    "Graphlets_ORCA": "ORCA"
}

DISEASE_TITLES = {
    "PD": "Parkinson's disease",
    "AZ": "Alzheimer's disease",
    "MS": "Multiple sclerosis",
    "ALS": "Amyotrophic lateral sclerosis",
    "HD": "Huntington's disease",
    "UNKNOWN": "Disease model"
}

DISEASE_PALETTES = {
    "PD": ["#D62728", "#DE4E4F", "#E57475", "#EC9A9B", "#F1B7B7", "#F5CACA", "#F8DADA", "#FAE8E8"],
    "AZ": ["#1F77B4", "#4392C6", "#68ACD7", "#8CC6E8", "#A9D5EF", "#BFE0F4", "#D5EBF8", "#E6F3FB"],
    "MS": ["#2CA02C", "#4EB44E", "#6FC76F", "#90DA90", "#A8E4A8", "#BCECBC", "#D2F3D2", "#E5F8E5"],
    "ALS": ["#9467BD", "#A883CB", "#BBA0D8", "#CEBCE5", "#DDD0ED", "#E6DDF2", "#EFE9F6", "#F6F2F9"],
    "HD": ["#FF7F0E", "#FF993E", "#FFB26E", "#FFCC9F", "#FFDAB8", "#FFE5CC", "#FFEFDF", "#FFF6EE"]
}

DEFAULT_FALLBACK_COLORS = ["#7F7F7F", "#8C8C8C", "#999999", "#A6A6A6", "#B3B3B3", "#CCCCCC", "#E6E6E6", "#F2F2F2"]


# -----------------------------
# Utilities
# -----------------------------
def safe_mkdir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def sanitize_filename(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s


def expand_paths(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if any(ch in p for ch in ["*", "?", "["]):
            out.extend(sorted(glob(p)))
        else:
            out.append(p)
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def pick_first_present(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def detect_disease_from_paths(paths: List[str]) -> str:
    """
    Scans the filename for the disease acronym, allowing for underscores, 
    hyphens, or dots as separators so 'ALS' doesn't trigger inside 'trials'.
    """
    for p in paths:
        filename = os.path.basename(p).upper()
        for disease in DISEASE_PALETTES.keys():
            # Matches start of string or separator, then disease, then end of string or separator
            if re.search(rf"(^|[_.-]){disease}([_.-]|$)", filename):
                return disease
    return "UNKNOWN"


def compute_model_order(df_k: pd.DataFrame, semper_prefix: str = "SEMPER") -> List[str]:
    mscore = df_k.groupby("model")["hits_at_k"].mean()
    models = mscore.index.astype(str).tolist()
    
    semper = [m for m in models if m.startswith(semper_prefix)]
    other = [m for m in models if not m.startswith(semper_prefix)]

    semper_sorted = sorted(semper, key=lambda m: -float(mscore.get(m, np.nan)))
    other_sorted = sorted(other, key=lambda m: -float(mscore.get(m, np.nan)))
    return semper_sorted + other_sorted


def collapse_global_models(df_k: pd.DataFrame, global_models: List[str], global_label: str = "GLOBAL") -> pd.DataFrame:
    if df_k.empty:
        return df_k

    df_k = df_k.copy()
    df_k["model"] = df_k["model"].astype(str)

    gset = set(map(str, global_models))
    is_global = df_k["model"].isin(gset)

    keep = df_k.loc[~is_global].copy()

    if is_global.any():
        collapsed = (
            df_k.loc[is_global]
            .groupby("model", as_index=False)["hits_at_k"]
            .mean()
        )
        collapsed["cluster"] = global_label
        out = pd.concat([keep, collapsed[["model", "cluster", "hits_at_k"]]], axis=0, ignore_index=True)
    else:
        out = keep

    return out


def get_rank_based_palette(disease: str, model_order: List[str]) -> dict:
    shades = DISEASE_PALETTES.get(disease, DEFAULT_FALLBACK_COLORS)
    palette = {}
    for i, model in enumerate(model_order):
        color_idx = min(i, len(shades) - 1)
        palette[model] = shades[color_idx]
    return palette


def plot_nature_style_hits(
    df_k_cluster: pd.DataFrame,
    k: int,
    out_png: str,
    model_order: List[str],
    disease: str,
    dpi: int = 300,
    rotate_xticks: int = 45,
) -> None:
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    palette = get_rank_based_palette(disease, model_order)

    w = max(6.0, 0.75 * len(model_order))
    h = 5.0
    fig, ax = plt.subplots(figsize=(w, h))

    df_clusters = df_k_cluster[df_k_cluster["cluster"] != "GLOBAL"]
    df_globals = df_k_cluster[df_k_cluster["cluster"] == "GLOBAL"]

    # 1. Plot the cell-type specific models (normal size dots)
    if not df_clusters.empty:
        sns.stripplot(
            data=df_clusters,
            x="model",
            y="hits_at_k",
            order=model_order,
            hue="model",
            palette=palette,
            jitter=0.25,
            alpha=0.85,
            size=5,
            ax=ax,
            legend=False
        )

    # 2. Plot the global models (Anchor Dots)
    if not df_globals.empty:
        sns.stripplot(
            data=df_globals,
            x="model",
            y="hits_at_k",
            order=model_order,
            hue="model",
            palette=palette,
            jitter=False,      
            alpha=1.0,         
            size=9,            
            marker="o",        
#            edgecolor="black", 
#            linewidth=1.2,
            ax=ax,
            legend=False
        )

    ax.set_xlabel("")
    ax.set_ylabel(f"Top-{k} Hits")
    
    plot_title = DISEASE_TITLES.get(disease, disease)
    ax.set_title(plot_title, pad=15)

    # Ceiling = k, the theoretical max Top-k Hits (Hits@25 -> 0..25).
    ax.set_ylim(0, k)

    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, rotation=rotate_xticks, ha="right")

    ax.xaxis.grid(False)
    ax.yaxis.grid(True, linestyle='-', alpha=0.5)
    sns.despine(ax=ax, left=False, bottom=False, trim=True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[OK] wrote {out_png}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials_csv", nargs="+", required=True)
    ap.add_argument(
        "--global_models",
        nargs="*",
        default=["ProTrek Seq", "ProTrek Struct", "ESM-C", "ORCA"],
    )
    ap.add_argument("--out_dir", default="hits_at_k_model_plots")
    ap.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 25])
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    safe_mkdir(args.out_dir)
    paths = expand_paths(args.trials_csv)
    if not paths:
        raise SystemExit("[ERROR] No files matched --trials_csv inputs.")
        
    detected_disease = detect_disease_from_paths(paths)

    frames = []
    for p in paths:
        if not os.path.exists(p):
            continue
        df_i = pd.read_csv(p)
        frames.append(df_i)

    if not frames:
        raise SystemExit("[ERROR] No readable CSVs found.")

    df = pd.concat(frames, axis=0, ignore_index=True)

    model_col = pick_first_present(df, ["model", "Model", "embedding", "Embedding"])
    ct_col = pick_first_present(df, ["cluster", "cell_type", "cell type", "Cell Type", "ct"])
    k_col = pick_first_present(df, ["k", "K"])
    hits_col = pick_first_present(df, ["hits_at_k", "hits@k", "hits", "Top-k Hits", "Top-K Hits"])

    df = df.rename(columns={model_col: "model", ct_col: "cluster", k_col: "k", hits_col: "hits_at_k"}).copy()
    df["model"] = df["model"].replace(MODEL_RENAME)

    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df["hits_at_k"] = pd.to_numeric(df["hits_at_k"], errors="coerce")
    df = df[np.isfinite(df["k"]) & np.isfinite(df["hits_at_k"])].copy()
    
    df["model"] = df["model"].astype(str)
    df["cluster"] = df["cluster"].astype(str)
    df["k"] = df["k"].astype(int)

    df_k_cluster = df.groupby(["model", "cluster", "k"], as_index=False)["hits_at_k"].mean()

    for k in args.k_values:
        df_k = df_k_cluster[df_k_cluster["k"] == int(k)].copy()
        if len(df_k) == 0:
            continue

        df_k_plot = collapse_global_models(df_k, global_models=args.global_models, global_label="GLOBAL")
        model_order = compute_model_order(df_k_plot, semper_prefix="SEMPER")

        out_png = os.path.join(
            args.out_dir,
            sanitize_filename(f"{detected_disease}_hits_at_{k}_models.png"),
        )

        plot_nature_style_hits(
            df_k_cluster=df_k_plot,
            k=int(k),
            out_png=out_png,
            model_order=model_order,
            disease=detected_disease,
            dpi=args.dpi
        )

    print(f"[DONE] Completed plotting for {detected_disease}.")


if __name__ == "__main__":
    main()
