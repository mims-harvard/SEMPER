#!/usr/bin/env python3
import argparse, os, math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import matplotlib.font_manager as font_manager


plt.rcParams['pdf.fonttype'] = 42

# ----- Nature Figure Styling -----
plt.rcParams.update({
#    'font.family': 'sans-serif',
#    'font.sans-serif': ['Arial', 'Helvetica', 'sans-serif'],
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300
})

# ---------------- helpers ----------------

def extract_cluster_id(celltype: str):
    if pd.isna(celltype): return None
    m = re.match(r"^\s*(\d+)", str(celltype))
    return m.group(1) if m else None

def safe_auprc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2 or np.isfinite(y_score).sum() == 0:
        return math.nan
    return float(average_precision_score(y_true, y_score))

def safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2 or np.isfinite(y_score).sum() == 0:
        return math.nan
    return float(roc_auc_score(y_true, y_score))

def parse_models(arg_list):
    out = []
    for item in arg_list:
        if "=" not in item:
            raise SystemExit(f"--model expects label=path, got {item}")
        label, path = item.split("=", 1)
        out.append((label.strip(), path.strip()))
    return out

def compute_metrics_by_group(df, group_col, label_col, score_col, min_n=10):
    rows = []
    sizes = df.groupby(group_col, dropna=False).size().rename("n_total")
    keep = set(sizes[sizes >= min_n].index)

    for gname, g in df.groupby(group_col, dropna=False):
        if gname not in keep: continue
        g = g.dropna(subset=[label_col, score_col]).copy()
        if g.empty: continue

        y = g[label_col].astype(int).to_numpy()
        s = g[score_col].astype(float).to_numpy()

        rows.append({
            group_col: gname,
            "n": int(len(g)),
            "auprc": safe_auprc(y, s),
            "auroc": safe_auroc(y, s),
        })
    return pd.DataFrame(rows)

# ---------------- plotting ----------------

def horizontal_dotplot_clean(perf_df, metric, title, agg_set, color_map, outfile, order="mean-desc"):
    models = perf_df["model"].unique().tolist()
    
    # Sorting logic: Use the mean of the spread for CT models, and the single aggregate value for global models
    if order == "mean-desc":
        means = perf_df[~perf_df["is_aggregate"]].groupby("model")["metric_value"].mean()
        agg_vals = perf_df[perf_df["is_aggregate"]].set_index("model")["metric_value"]
        
        for m in agg_set:
            if m in agg_vals.index:
                means[m] = agg_vals[m]
                
        means = means.sort_values(ascending=False)
        models = means.index.tolist()

    ypos = {m: i for i, m in enumerate(models)}
    
    # Original comfortable figure size
    plt.figure(figsize=(14, max(4, 0.45 * len(models) + 1.5)))
    ax = plt.gca()
    rng = np.random.default_rng(42)

    for m in models:
        g = perf_df[perf_df["model"] == m]
        y0 = ypos[m]
        
        if m in agg_set:
            # GLOBAL MODEL: Plot single point, slightly larger so it's distinguishable
            val = g[g["is_aggregate"]]["metric_value"].values
            if len(val) > 0:
                ax.scatter(val, [y0], color=color_map[m], s=60, edgecolor='black', linewidth=0.8, zorder=3)
        else:
            # CELL TYPE MODEL: Plot ONLY the spread (jittered dots)
            spread_df = g[~g["is_aggregate"]]
            jit = (rng.random(len(spread_df)) - 0.5) * 0.25 # Slightly wider jitter to show density clearly
            ax.scatter(spread_df["metric_value"].to_numpy(), y0 + jit, 
                       color=color_map[m], alpha=0.75, s=28, linewidth=0, zorder=2)

    # Aesthetics
    ax.set_yticks([ypos[m] for m in models])
    ax.set_yticklabels(models)
    ax.set_xlabel(metric.upper(), fontweight='bold')
    ax.set_ylim(-1, len(models))
    
    if metric.lower() in {"auprc", "auroc"}:
        ax.set_xlim(0, 1.02)
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.grid(axis="x", linestyle="--", alpha=0.35)
    else:
        ax.grid(axis="both", linestyle="--", alpha=0.25)

    if title:
        ax.set_title(title, pad=15)

    plt.tight_layout()
    plt.savefig(f"{outfile}.png", dpi=300)
    plt.savefig(f"{outfile}.pdf", format="pdf") # Vector format for Nature
    plt.close()

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Horizontal mixed dot plots from zero-shot score CSVs.")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--group-by", choices=["celltype", "cluster"], default="cluster")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--score-col", default="score_model")
    ap.add_argument("--celltype-col", default="cell_type_raw")      
    ap.add_argument("--clusterlabel-col", default="cluster_label")
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--order", choices=["mean-desc", "input"], default="mean-desc")
    ap.add_argument("--aggregate-models", default="")
    ap.add_argument("--out-prefix", default="zeroshot_compare_dot")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    models = parse_models(args.model)
    agg_set = {m.strip() for m in args.aggregate_models.split(",") if m.strip()}

    # Lock colors based on input model order to ensure Panels B, C, D match perfectly
    # Using the Okabe-Ito colorblind-friendly palette
    color_map = {}
    for label, _ in models:
        lbl_upper = label.upper()
        
        # 1. SEMPER Models (The Heroes - Oranges)
        if "SEMPER" in lbl_upper:
            color_map[label] = "#D55E00"  # Bold Vermillion
            
        # 2. Cell-Type Baselines (The Competitors - Blues)
        elif "PINNACLE" in lbl_upper:
            color_map[label] = "#0072B2"  # Dark Blue
        elif "PPI-GT" in lbl_upper:
            color_map[label] = "#56B4E9"  # Sky Blue
        elif "ORCA-CT" in lbl_upper:
            color_map[label] = "#0285cf"  #(or a 3rd blue/slate)
            
        # 3. Global Baselines (The References - Greys)
        else:  
            # For ESM_C, ProTrek, graphlets_global
            color_map[label] = "#034b73"  # Neutral Grey

    rows = []
    for label, path in models:
        df = pd.read_csv(path, dtype=str)
        df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
        df[args.score_col] = pd.to_numeric(df[args.score_col], errors="coerce")
        df = df.dropna(subset=[args.label_col, args.score_col]).copy()

        y_true = df[args.label_col].astype(int).to_numpy()
        y_score = df[args.score_col].astype(float).to_numpy()

        auprc = safe_auprc(y_true, y_score)
        auroc = safe_auroc(y_true, y_score)

        rows.append({"model": label, "group": "AGGREGATE", "metric": "auprc", "metric_value": auprc, "is_aggregate": True})
        rows.append({"model": label, "group": "AGGREGATE", "metric": "auroc", "metric_value": auroc, "is_aggregate": True})

        if label in agg_set:
            continue

        if args.group_by == "celltype":
            df["group"] = df[args.celltype_col].astype(str)
        elif args.group_by == "cluster":
            if args.clusterlabel_col in df.columns:
                df["group"] = df[args.clusterlabel_col].astype(str)
            else:
                df["group"] = df[args.celltype_col].apply(lambda x: f"cluster:{extract_cluster_id(x)}")

        perf = compute_metrics_by_group(df, group_col="group", label_col=args.label_col, score_col=args.score_col, min_n=args.min_n)
        for _, r in perf.iterrows():
            rows.append({"model": label, "group": r["group"], "metric": "auprc", "metric_value": r["auprc"], "is_aggregate": False})
            rows.append({"model": label, "group": r["group"], "metric": "auroc", "metric_value": r["auroc"], "is_aggregate": False})

    long_df = pd.DataFrame(rows)
    if long_df.empty: raise SystemExit("No data to plot.")

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    for metr in ["auprc", "auroc"]:
        sub = long_df[long_df["metric"] == metr].dropna(subset=["metric_value"])
        if sub.empty: continue
        out = f"{args.out_prefix}_{args.group_by}_{metr.upper()}_clean"
        horizontal_dotplot_clean(sub, metr, args.title, agg_set, color_map, out, order=args.order)

if __name__ == "__main__":
    main()
