#!/usr/bin/env python3
"""
Figure 1e: Hits@10 of each protein language model (PLM) backbone alone versus
SEMPER built on that backbone.

Each --pair gives a display name, the backbone-alone trials CSV
(run_evaluations/neuro_eval_esm_pklpt_ct_network.py) and the SEMPER trials CSV
(run_evaluations/neuro_eval_semper_ct_network.py), both scored on the same
pooled seed plan. Rows are drawn top to bottom in the order given.

Aggregation (--agg), computed per model from the trial-level hits_at_k column:
  mean_of_ct_max : best trial (max over seed sets) in each cell type, then the
                   mean across cell types. This is the aggregation plotted in the
                   paper figure.
  mean           : mean over seed sets in each cell type, then the mean across
                   cell types (the per-cell type summary used in Figure 4).

Example:
  python plot_fig1e.py \
      --pair "ESM-C 600M" trials/neuro_ESMC600M__ct_network__trials.csv trials/neuro_SEMPER_ESMC600M_t010__trials.csv \
      --pair "ESM-2 8M"   trials/neuro_ESM2_8M__ct_network__trials.csv  trials/neuro_SEMPER_ESM2_8M_t010__trials.csv \
      --k 10 --out_dir figures/
"""
import argparse
import os
from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


# -----------------------------
# Aggregation
# -----------------------------
def load_trials(csv_path: str, k: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["cluster", "trial", "k", "hits_at_k"])
    df = df[df["k"].astype(int) == k].copy()
    if df.empty:
        raise SystemExit(f"[ERROR] No rows with k={k} in {csv_path}")
    df["cluster"] = df["cluster"].astype(str)
    return df


def trial_keys(df: pd.DataFrame) -> Set[Tuple[str, int]]:
    return set(zip(df["cluster"], df["trial"].astype(int)))


def aggregate_hits(df: pd.DataFrame, agg: str, keep: Optional[Set[Tuple[str, int]]] = None) -> float:
    if keep is not None:
        mask = [(c, int(t)) in keep for c, t in zip(df["cluster"], df["trial"])]
        df = df[mask]
    per_ct = df.groupby("cluster")["hits_at_k"]
    if agg == "mean_of_ct_max":
        return float(per_ct.max().mean())
    if agg == "mean":
        return float(per_ct.mean().mean())
    raise ValueError(agg)


def collect(pairs, k: int, agg: str, common_trials: bool) -> pd.DataFrame:
    rows = []
    for name, plm_csv, semper_csv in pairs:
        plm = load_trials(plm_csv, k)
        semper = load_trials(semper_csv, k)
        keep = trial_keys(plm) & trial_keys(semper) if common_trials else None
        rows.append({
            "name": name,
            "plm_hits": aggregate_hits(plm, agg, keep),
            "semper_hits": aggregate_hits(semper, agg, keep),
            "plm_n_trials": len(trial_keys(plm)) if keep is None else len(keep),
            "semper_n_trials": len(trial_keys(semper)) if keep is None else len(keep),
            "n_celltypes_plm": plm["cluster"].nunique(),
            "n_celltypes_semper": semper["cluster"].nunique(),
            "plm_csv": plm_csv,
            "semper_csv": semper_csv,
        })
    out = pd.DataFrame(rows)
    out["delta"] = out["semper_hits"] - out["plm_hits"]
    return out


# -----------------------------
# Plot
# -----------------------------
def plot_dumbbell(df: pd.DataFrame, out_base: str, k: int, xmax: float, dpi: int) -> None:
    n = len(df)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    fig.subplots_adjust(left=0.24, right=0.66, bottom=0.12, top=0.97)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(direction='out', length=5, width=1.2)

    ypos = np.arange(n)[::-1]  # first pair at the top
    for y, (_, r) in zip(ypos, df.iterrows()):
        ax.plot([r["plm_hits"], r["semper_hits"]], [y, y], color="black", lw=1.4, zorder=1)
    ax.scatter(df["plm_hits"], ypos, marker="s", s=70, facecolors="white",
               edgecolors="black", linewidths=1.6, zorder=3, label="Protein Language\nModel (PLM)")
    ax.scatter(df["semper_hits"], ypos, marker="s", s=70, facecolors="black",
               edgecolors="black", linewidths=1.6, zorder=4, label="SEMPER with\nPLM backbone")

    ax.set_yticks(ypos)
    ax.set_yticklabels(df["name"], fontsize=10)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0, xmax)
    ax.set_xticks(np.arange(0, xmax + 1e-9, 2))
    ax.set_xlabel(f"Hits@{k}", fontsize=10, labelpad=6)
    ax.legend(frameon=True, fontsize=8, loc="center left", bbox_to_anchor=(1.04, 0.25))
    for ext in ("png", "pdf"):
        plt.savefig(f"{out_base}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_base}.png / .pdf")


def main():
    ap = argparse.ArgumentParser(description="Figure 1e: PLM backbone alone vs SEMPER on that backbone (Hits@k).")
    ap.add_argument("--pair", nargs=3, action="append", required=True,
                    metavar=("NAME", "PLM_TRIALS_CSV", "SEMPER_TRIALS_CSV"),
                    help="Repeat once per backbone; rows are drawn top to bottom in this order.")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--agg", default="mean_of_ct_max", choices=["mean_of_ct_max", "mean"])
    ap.add_argument("--common_trials", action="store_true",
                    help="Score both models only on (cell type, trial) pairs present in both CSVs. "
                         "The backbone-alone runs use --skip_if_missing_seed and can have fewer trials.")
    ap.add_argument("--xmax", type=float, default=8.0)
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no_plot", action="store_true", help="Only write the values CSV.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = collect(args.pair, args.k, args.agg, args.common_trials)

    tag = f"k{args.k}_{args.agg}" + ("_commontrials" if args.common_trials else "")
    out_csv = os.path.join(args.out_dir, f"fig1e_values_{tag}.csv")
    df.to_csv(out_csv, index=False)
    with pd.option_context("display.width", 200):
        print(df[["name", "plm_hits", "semper_hits", "delta", "plm_n_trials", "semper_n_trials"]].to_string(index=False))
    print(f"[OK] wrote {out_csv}")

    if not args.no_plot:
        plot_dumbbell(df, os.path.join(args.out_dir, f"fig1e_{tag}"), args.k, args.xmax, args.dpi)


if __name__ == "__main__":
    main()
