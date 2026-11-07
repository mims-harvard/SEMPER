#!/usr/bin/env python3
"""
Paired one-sided Wilcoxon signed-rank test of per-cell-type AUPRC (or AUROC),
SEMPER vs each cell type-specific baseline (Supplementary Table 5).

Inputs are the per-model score CSVs written by the score_*.py scripts. For each
model, AUPRC is computed within every cell type (cluster_label) that has at least
--min-n scored rows. Only cell types valid for both SEMPER and the baseline are
paired. The test is scipy.stats.wilcoxon(semper, baseline, alternative="greater").
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import wilcoxon

# ---------------- helpers (same per-group metrics as plot_auprc.py) ----------------

def safe_auprc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2 or np.isfinite(y_score).sum() == 0:
        return np.nan
    return float(average_precision_score(y_true, y_score))

def safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2 or np.isfinite(y_score).sum() == 0:
        return np.nan
    return float(roc_auc_score(y_true, y_score))

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
            "auprc": safe_auprc(y, s),
            "auroc": safe_auroc(y, s),
        })
    # drop cell types with an undefined metric so only valid cell types are paired
    return pd.DataFrame(rows).dropna().set_index(group_col)

def load_metrics(path, args):
    df = pd.read_csv(path, dtype=str)
    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
    df[args.score_col] = pd.to_numeric(df[args.score_col], errors="coerce")
    return compute_metrics_by_group(df, args.group_col, args.label_col, args.score_col, args.min_n)

def parse_models(arg_list):
    out = []
    for item in arg_list:
        if "=" not in item:
            raise SystemExit(f"--baseline expects label=path, got {item}")
        label, path = item.split("=", 1)
        out.append((label.strip(), path.strip()))
    return out

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Paired one-sided Wilcoxon of per-cell-type AUPRC, SEMPER vs baselines.")
    ap.add_argument("--semper", required=True, help="SEMPER score CSV")
    ap.add_argument("--baseline", action="append", required=True, help="label=path for each baseline score CSV")
    ap.add_argument("--metric", choices=["auprc", "auroc"], default="auprc")
    ap.add_argument("--group-col", default="cluster_label")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--score-col", default="score_model")
    ap.add_argument("--min-n", type=int, default=50,
                    help="min scored rows per cell type (paper: 50 co-expression/connectivity, 30 co-abundance/colocalization)")
    ap.add_argument("--out-csv", default=None, help="optional: write the test results here")
    args = ap.parse_args()

    semper = load_metrics(args.semper, args)
    m = args.metric

    print(f"One-sided paired Wilcoxon, H1: SEMPER > baseline  (per-cell-type {m.upper()}, min-n {args.min_n})")
    print(f"{'baseline':<12} {'N':>5} {'W':>12} {'P':>10}  {'median SEMPER':>14} {'median base':>12}")

    results = []
    for label, path in parse_models(args.baseline):
        base = load_metrics(path, args)
        # pair on cell types where both models have a valid metric
        merged = semper.join(base, lsuffix="_semper", rsuffix="_base", how="inner")
        n = len(merged)
        if n < 5:
            print(f"{label:<12} {n:>5}  too few paired cell types")
            continue

        a, b = merged[f"{m}_semper"], merged[f"{m}_base"]
        stat, p = wilcoxon(a, b, alternative="greater")
        print(f"{label:<12} {n:>5} {stat:>12.1f} {p:>10.2e}  {a.median():>14.3f} {b.median():>12.3f}")
        results.append({"baseline": label, "metric": m, "N": n, "W": float(stat), "P": float(p),
                        "median_semper": float(a.median()), "median_baseline": float(b.median())})

    if args.out_csv:
        pd.DataFrame(results).to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
