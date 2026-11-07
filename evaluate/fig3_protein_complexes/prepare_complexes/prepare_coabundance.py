"""
prepare_coabundance.py
-----------------------
Panel d: co-expressed AND brain co-abundant (Laman Trip et al.).

Step 1: take the co-expression posneg.csv (prepare_coexpressed.py) and keep
the positives (label=1) for which ALL pairwise combinations of ENSP members
have a brain association score >= threshold (paper: 0.5).

Step 2: draw this definition's OWN size-matched decoys for the kept positives
(same cell type, same k, 1:10, seed 42), using step2_generate from
prepare_coexpressed.py. The co-expression decoys are not reused.

Outputs, per threshold:
  coab_assignments_t<thr>.csv   kept positives in assignments format
  posneg_coabundance_t<thr>.csv positives + own decoys, ready for scoring

Usage:
    python prepare_coabundance.py \
        --posneg-csv   data/prepared/coexpressed/posneg.csv \
        --assoc-brain  /path/to/cohorts_combined_brain_tumor_avg_outer_prob.csv \
        --ensp-json    /path/to/ENSP_to_pref.json \
        --ppi-dir      data/ppi_edgelists/ \
        --protrek-pkl  data/ensp_id2vec_mean.pkl.gz \
        --thresholds   0.5 \
        --neg-per-pos  10 --avoid-pos-members --dedupe-globally --seed 42 \
        --out-dir      data/prepared/coabundance/
"""

import argparse
import itertools
import json
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

# decoy generation is shared with the co-expression definition
from prepare_coexpressed import load_pickle_gz, step2_generate
from prepare_coexpressed import norm_ensp as norm_ensp_prefix

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


def to_assignments(positives_df: pd.DataFrame,
                   members_col: str = "members_ensp") -> pd.DataFrame:
    """Kept positives in the assignments format read by step2_generate."""
    pos = positives_df
    return pd.DataFrame({
        "complex_name":  pos.get("complex_name", ""),
        "complex_id":    pos.get("complex_id", ""),
        "subunits_ensp": pos[members_col],
        "members_norm":  pos[members_col],
        "n_members":     pos.get("k", ""),
        "cell_type":     pos["cell_type"],
        "status":        "co-expressed",
    })


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posneg-csv",   required=True,
                    help="Co-expression posneg.csv from prepare_coexpressed.py")
    ap.add_argument("--assoc-brain",  required=True,
                    help="Laman Trip brain association CSV")
    ap.add_argument("--ensp-json", required=True,
                    help="ENSP_to_pref.json")
    ap.add_argument("--ppi-dir",      required=True,
                    help="Cell type PPI edgelists (decoy pools)")
    ap.add_argument("--protrek-pkl",  required=True,
                    help="ProTrek ENSP->vec pickle.gz; defines the allowed protein universe")
    ap.add_argument("--out-dir",      default="./coabundance_posneg",
                    help="Output directory")
    ap.add_argument("--members-col",  default="members_ensp")
    ap.add_argument("--label-col",    default="label")
    ap.add_argument("--thresholds",   nargs="+", type=float,
                    default=[0.5],
                    help="Co-abundance thresholds to apply (paper: 0.5)")
    ap.add_argument("--neg-per-pos",       type=int, default=10)
    ap.add_argument("--neg-max-tries",     type=int, default=20000)
    ap.add_argument("--avoid-pos-members", action="store_true")
    ap.add_argument("--dedupe-globally",   action="store_true")
    ap.add_argument("--seed",              type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load inputs
    ensp_to_pref = load_ensp_to_pref(args.ensp_json)
    gene_lookup  = load_brain_atlas(args.assoc_brain, ensp_to_pref)
    allowed = set(norm_ensp_prefix(k) for k in load_pickle_gz(args.protrek_pkl).keys())

    print(f"\n[INFO] Loading posneg CSV: {args.posneg_csv}")
    posneg_df = pd.read_csv(args.posneg_csv, dtype=str)
    print(f"[INFO] Loaded {len(posneg_df):,} rows | "
          f"columns: {list(posneg_df.columns)}")

    # Check required columns
    required = {args.members_col, args.label_col, "cell_type"}
    missing  = required - set(posneg_df.columns)
    if missing:
        raise SystemExit(f"[ERROR] posneg CSV missing columns: {missing}")

    summary_rows = []
    for threshold in args.thresholds:
        # Step 1: co-abundance filter on positives (co-expression decoys are dropped)
        filtered_df = filter_posneg(
            posneg_df, gene_lookup, ensp_to_pref,
            threshold=threshold,
            members_col=args.members_col,
            label_col=args.label_col,
        )
        positives = filtered_df[filtered_df[args.label_col].astype(int) == 1]

        tag         = str(threshold).replace(".", "p")
        assign_csv  = os.path.join(args.out_dir, f"coab_assignments_t{tag}.csv")
        out_csv     = os.path.join(args.out_dir, f"posneg_coabundance_t{tag}.csv")
        summary_json = os.path.join(args.out_dir, f"posneg_coabundance_t{tag}.summary.json")
        to_assignments(positives, args.members_col).to_csv(assign_csv, index=False)
        print(f"[INFO] Wrote {len(positives):,} co-abundance positives -> {assign_csv}")

        # Step 2: this definition's own size-matched decoys
        step2_generate(assign_csv, args.ppi_dir, allowed,
                       args.neg_per_pos, args.neg_max_tries,
                       args.avoid_pos_members, args.dedupe_globally,
                       args.seed, out_csv, summary_json)

        out = pd.read_csv(out_csv, dtype=str)
        summary_rows.append({
            "threshold":   threshold,
            "n_positives": int((out["label"].astype(int) == 1).sum()),
            "n_negatives": int((out["label"].astype(int) == 0).sum()),
            "out_csv":     out_csv,
        })

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for row in summary_rows:
        print(f"  threshold={row['threshold']}  "
              f"positives={row['n_positives']:,}  "
              f"negatives={row['n_negatives']:,}")
        print(f"    -> {row['out_csv']}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
