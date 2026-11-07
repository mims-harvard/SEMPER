#!/usr/bin/env python3
"""
Build a seed plan for zero-shot target retrieval (Methods, "Zero-shot retrieval of
candidate drug targets").

For every cell type network, the positives are the disease targets whose proteins
are in that network. Cell types with fewer than k_seed + 1 positives are skipped.
Each of n_trials trials samples k_seed positives as seeds; the remaining positives
are held out and scored by the retrieval scripts in run_evaluations/.

The seed plan fixes the seeds so every model is evaluated on identical trials.

Outputs:
  <out_prefix>_seedplan.json   clusters -> universe_ensp, positive_ensp, trials -> seed_ensp
  <out_prefix>_summary.json    per cell type coverage
  <out_prefix>_targets.json    {"clinically_relevant_targets": [...]}, the --pd_gene_file
                               expected by run_evaluations/*.py

Example (Parkinson's disease, paper settings):
  python make_seedplan.py \
      --targets_csv ../../data/benchmarks/clinical_targets/PD_clinical_targets.csv \
      --ppi_dir     ../../data/networks/ppi_edgelists/ \
      --ensp_json   ../data/ensp_to_gene.json \
      --out_prefix  seedplans/PD
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Set

import pandas as pd


def norm_ensp(x: str) -> str:
    s = str(x).strip()
    return s[5:] if s.startswith("9606.") else s


def load_targets(args) -> List[str]:
    if args.targets_csv:
        df = pd.read_csv(args.targets_csv)
        genes = df["targetId_genename"].dropna().tolist()
    else:
        with open(args.disease_json) as f:
            genes = json.load(f)[args.ot_key]
    return sorted({g.strip().upper() for g in genes if isinstance(g, str) and g.strip()})


def load_ensp_to_hgnc(path: str) -> Dict[str, str]:
    with open(path) as f:
        m = json.load(f)
    return {norm_ensp(k): v.strip().upper() for k, v in m.items() if isinstance(v, str) and v.strip()}


def load_ppi_nodes(path: str) -> Set[str]:
    nodes = set()
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                nodes.update(norm_ensp(p) for p in parts[:2])
    return nodes


def main():
    ap = argparse.ArgumentParser(description="Generate a seed plan for target retrieval.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--targets_csv", help="Released target table, e.g. data/benchmarks/clinical_targets/PD_clinical_targets.csv.")
    src.add_argument("--disease_json", help="Alternative: JSON with an HGNC list under --ot_key.")
    ap.add_argument("--ot_key", default="clinically_relevant_targets")
    ap.add_argument("--ppi_dir", required=True, help="Directory of cell type edgelists named <cluster>_edgelist.txt.")
    ap.add_argument("--ensp_json", required=True, help="ENSP -> HGNC mapping (evaluate/data/ensp_to_gene.json).")
    ap.add_argument("--k_seed", type=int, default=15, help="Seeds per trial (paper: 15).")
    ap.add_argument("--n_trials", type=int, default=100, help="Trials per cell type (paper: 100).")
    ap.add_argument("--rng_seed", type=int, default=42)
    ap.add_argument("--min_positives", type=int, default=None, help="Default: k_seed + 1.")
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    rng = random.Random(args.rng_seed)
    min_pos = args.min_positives if args.min_positives is not None else args.k_seed + 1

    targets = load_targets(args)
    ensp_to_hgnc = load_ensp_to_hgnc(args.ensp_json)

    edgelists = sorted(fn for fn in os.listdir(args.ppi_dir) if fn.endswith("_edgelist.txt"))
    if not edgelists:
        sys.exit(f"[ERROR] No *_edgelist.txt files in {args.ppi_dir}")

    clusters, per_celltype = {}, {}
    target_set = set(targets)
    for fn in edgelists:
        cluster_id = fn[: -len("_edgelist.txt")]
        universe = sorted(e for e in load_ppi_nodes(os.path.join(args.ppi_dir, fn)) if e in ensp_to_hgnc)
        positives = sorted(e for e in universe if ensp_to_hgnc[e] in target_set)
        status = "kept" if len(positives) >= min_pos else "skipped_too_few_positives"
        per_celltype[cluster_id] = {"U_eval": len(universe), "Pos_eval": len(positives), "status": status}
        if status != "kept":
            continue
        trials = [{"trial": t, "seed_ensp": sorted(rng.sample(positives, args.k_seed))} for t in range(args.n_trials)]
        clusters[cluster_id] = {"ppi_file": fn, "universe_ensp": universe, "positive_ensp": positives, "trials": trials}

    seedplan = {
        "task": "zero_shot_target_retrieval",
        "k_seed": args.k_seed,
        "n_trials": args.n_trials,
        "rng_seed": args.rng_seed,
        "universe_definition": "U_eval[cell] = nodes(PPI_edgelist[cell]) ∩ ENSP_to_HGNC_mapping",
        "positives_definition": "positive_ensp[cell] = {ensp in U_eval[cell] : HGNC(ensp) in targets}",
        "clusters": clusters,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    with open(f"{args.out_prefix}_seedplan.json", "w") as f:
        json.dump(seedplan, f, indent=2)
    with open(f"{args.out_prefix}_summary.json", "w") as f:
        json.dump({"min_positives": min_pos, "n_targets": len(targets),
                   "celltypes_total": len(edgelists), "celltypes_kept": len(clusters),
                   "per_celltype": per_celltype}, f, indent=2)
    with open(f"{args.out_prefix}_targets.json", "w") as f:
        json.dump({args.ot_key: targets}, f, indent=2)
    print(f"[DONE] kept {len(clusters)}/{len(edgelists)} cell types -> {args.out_prefix}_seedplan.json")


if __name__ == "__main__":
    main()
