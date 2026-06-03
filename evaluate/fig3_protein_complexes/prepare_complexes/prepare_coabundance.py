"""
filter_posneg_coabundance.py

Takes the existing posneg.csv (already containing co-expressed positives)
and applies an additional co-abundance filter using Laman Trip et al.
brain association scores.

For each positive complex (label=1), ALL pairwise combinations of its
ENSP members must have a brain association score >= threshold.
Negatives (label=0) are kept as-is.

Outputs two new filtered CSVs, one per threshold (0.5 and 0.8),
ready to be passed directly to zero_shot_single_model.py.

Usage:
    python filter_posneg_coabundance.py \
        --posneg-csv   /path/to/posneg.csv \
        --assoc-brain  /path/to/cohorts_combined_brain_tumor_avg_outer_prob.csv \
        --ensp-json /path/to/ENSP_to_pref.json \
        --out-dir      ./coabundance_posneg/
"""

import argparse
import itertools
import json
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

# ------------------------------------------------------------------ #
#  Utilities                                                           #
# ------------------------------------------------------------------ #

def load_ensp_to_pref(json_path: str) -> dict:
    with open(json_path, "r") as f:
        mp = json.load(f)
    norm = {}
    for k, v in mp.items():
        k, v = str(k), str(v).strip()
        if k.startswith("9606."):
            bare = k.split("9606.", 1)[1]
            norm[bare] = v
            norm[k]    = v
        else:
            norm[k] = v
            if k.startswith("ENSP"):
                norm["9606." + k] = v
    return norm


def norm_ensp(x: str) -> str:
    """Strip species prefix and version suffix."""
    s = str(x).strip()
    if s.startswith("9606."):
        s = s[5:]
    m = re.match(r"^(ENSP\d+)(?:\.\d+)?$", s)
    return m.group(1) if m else s


def split_members(cell: str) -> list:
    """Parse semicolon/comma-separated ENSP member list."""
    if pd.isna(cell):
        return []
    members = []
    for t in str(cell).replace(",", ";").split(";"):
        t = t.strip()
        if t and t.lower() not in {"na", "nan", "none", "null"}:
            members.append(norm_ensp(t))
    return list(dict.fromkeys(members))  # deduplicate preserving order


def load_brain_atlas(csv_path: str, ensp_to_pref: dict) -> dict:
    """
    Load Laman Trip brain association scores.
    Keys: canonical sorted tuple of gene symbols (gA, gB) where gA < gB.
    Also builds ENSP-pair -> score lookup by mapping ENSP -> gene symbol first.
    Returns two lookups:
        gene_lookup : (gA, gB) -> score   [gene symbol pairs]
        ensp_lookup : (ENSP1, ENSP2) -> score  [ENSP pairs, canonical order]
    """
    print(f"[INFO] Loading brain atlas: {csv_path}")
    df = pd.read_csv(csv_path)
    score_col = [c for c in df.columns if c not in ("prot1", "prot2")][0]

    gene_lookup = {}
    for row in tqdm(df.itertuples(index=False), total=len(df),
                    desc="Building atlas lookup"):
        gA, gB = sorted([row.prot1, row.prot2])
        gene_lookup[(gA, gB)] = getattr(row, score_col)

    print(f"[INFO] Atlas loaded: {len(gene_lookup):,} gene-pair entries")

    # Build ENSP -> gene symbol reverse map for fast lookup
    # We'll do ENSP -> gene at query time using ensp_to_pref
    return gene_lookup


def get_score_for_ensp_pair(ensp1: str, ensp2: str,
                              gene_lookup: dict,
                              ensp_to_pref: dict) -> float:
    """
    Look up Laman Trip score for a pair of ENSP IDs.
    Maps ENSP -> gene symbol first, then queries gene_lookup.
    Returns np.nan if either protein is not in the atlas.
    """
    g1 = ensp_to_pref.get(ensp1, "")
    g2 = ensp_to_pref.get(ensp2, "")
    if not g1 or not g2:
        return np.nan
    gA, gB = sorted([g1, g2])
    return gene_lookup.get((gA, gB), np.nan)


def all_pairs_above_threshold(members: list,
                               gene_lookup: dict,
                               ensp_to_pref: dict,
                               threshold: float) -> bool:
    """
    Returns True if ALL pairwise combinations of members have
    brain association score >= threshold.
    Returns False if any pair is below threshold OR not in atlas.
    """
    if len(members) < 2:
        return False

    for ensp1, ensp2 in itertools.combinations(members, 2):
        score = get_score_for_ensp_pair(
            ensp1, ensp2, gene_lookup, ensp_to_pref
        )
        if np.isnan(score) or score < threshold:
            return False
    return True


# ------------------------------------------------------------------ #
#  Main filtering logic                                                #
# ------------------------------------------------------------------ #

def filter_posneg(posneg_df: pd.DataFrame,
                   gene_lookup: dict,
                   ensp_to_pref: dict,
                   threshold: float,
                   members_col: str = "members_ensp",
                   label_col: str = "label") -> pd.DataFrame:
    """
    Keep:
      - All negatives (label == 0) unchanged
      - Positives (label == 1) only if ALL member pairs >= threshold

    Returns filtered DataFrame in the same column order as input.
    """
    negatives = posneg_df[posneg_df[label_col].astype(int) == 0].copy()
    positives = posneg_df[posneg_df[label_col].astype(int) == 1].copy()

    print(f"\n[INFO] Threshold = {threshold}")
    print(f"  Input positives : {len(positives):,}")
    print(f"  Input negatives : {len(negatives):,}")

    keep_mask = []
    for _, row in tqdm(positives.iterrows(),
                        total=len(positives),
                        desc=f"  Filtering positives (threshold={threshold})"):
        members = split_members(row[members_col])
        keep = all_pairs_above_threshold(
            members, gene_lookup, ensp_to_pref, threshold
        )
        keep_mask.append(keep)

    positives_filtered = positives[keep_mask].copy()

    print(f"  Positives kept  : {len(positives_filtered):,} "
          f"({100*len(positives_filtered)/max(len(positives),1):.1f}%)")
    print(f"  Positives dropped: {len(positives) - len(positives_filtered):,}")

    result = pd.concat([positives_filtered, negatives], ignore_index=True)
    print(f"  Output total rows: {len(result):,}")
    return result


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posneg-csv",   required=True,
                    help="Existing posneg.csv (co-expression filtered)")
    ap.add_argument("--assoc-brain",  required=True,
                    help="Laman Trip brain association CSV")
    ap.add_argument("--ensp-json", required=True,
                    help="ENSP_to_pref.json")
    ap.add_argument("--out-dir",      default="./coabundance_posneg",
                    help="Output directory for filtered CSVs")
    ap.add_argument("--members-col",  default="members_ensp")
    ap.add_argument("--label-col",    default="label")
    ap.add_argument("--thresholds",   nargs="+", type=float,
                    default=[0.5, 0.8],
                    help="Co-abundance thresholds to apply (default: 0.5 0.8)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load inputs
    ensp_to_pref = load_ensp_to_pref(args.ensp_json)
    gene_lookup  = load_brain_atlas(args.assoc_brain, ensp_to_pref)

    print(f"\n[INFO] Loading posneg CSV: {args.posneg_csv}")
    posneg_df = pd.read_csv(args.posneg_csv, dtype=str)
    print(f"[INFO] Loaded {len(posneg_df):,} rows | "
          f"columns: {list(posneg_df.columns)}")

    label_counts = posneg_df[args.label_col].value_counts()
    print(f"[INFO] Label distribution:\n{label_counts.to_string()}")

    # Check required columns
    required = {args.members_col, args.label_col}
    missing  = required - set(posneg_df.columns)
    if missing:
        raise SystemExit(f"[ERROR] posneg CSV missing columns: {missing}")

    # Apply each threshold and save
    summary_rows = []
    for threshold in args.thresholds:
        filtered_df = filter_posneg(
            posneg_df, gene_lookup, ensp_to_pref,
            threshold=threshold,
            members_col=args.members_col,
            label_col=args.label_col,
        )

        tag     = str(threshold).replace(".", "p")
        out_csv = os.path.join(args.out_dir,
                               f"posneg_coabundance_t{tag}.csv")
        filtered_df.to_csv(out_csv, index=False)
        print(f"\n[INFO] Saved: {out_csv}")

        n_pos = int((filtered_df[args.label_col].astype(int) == 1).sum())
        n_neg = int((filtered_df[args.label_col].astype(int) == 0).sum())
        summary_rows.append({
            "threshold":   threshold,
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "out_csv":     out_csv,
        })

    # Print summary and next steps
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for row in summary_rows:
        print(f"  threshold={row['threshold']}  "
              f"positives={row['n_positives']:,}  "
              f"negatives={row['n_negatives']:,}")
        print(f"    -> {row['out_csv']}")

    print("\n[NEXT STEPS] Run zero_shot_single_model.py on each output:")
    for row in summary_rows:
        tag = str(row['threshold']).replace(".", "p")
        print(f"""
  python zero_shot_single_model.py \\
    --eval-csv {row['out_csv']} \\
    --model-name semper_esm_coabundance_t{tag} \\
    --model-pt   <path/to/esm_projected.pth> \\
    --model-labels <path/to/esm_projected_labels.txt> \\
    --out-scores-csv out_scores/semper_esm_coabundance_t{tag}_scores.csv \\
    --out-metrics-json out_scores/semper_esm_coabundance_t{tag}_metrics.json""")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
