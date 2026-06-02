#!/usr/bin/env python3
"""
prepare_coexpressed.py
-----------------------
Panel d: end-to-end data preparation for zero-shot complex identification
using the definition: complex members are CO-EXPRESSED in the cell type
(all members present in the cell type's PPI node set, no edge requirement).

Output status: "co-expressed"

Usage:
    python prepare_coexpressed.py \
        --complexes    data/filtered_complexes_min3.csv \
        --ppi-dir      data/ppi_edgelists/ \
        --protrek-pkl  data/ensp_id2vec_mean.pkl.gz \
        --out-dir      data/prepared/coexpressed/ \
        --neg-per-pos  10 \
        --avoid-pos-members \
        --dedupe-globally
"""

import argparse, gzip, json, os, pickle, re, sys, random
from collections import defaultdict
from typing import List, Dict, Set, Optional
import pandas as pd

def norm_ensp(x):
    s = str(x).strip()
    return s[5:] if s.startswith("9606.") else s

def split_members(cell):
    if pd.isna(cell): return []
    out = []
    for t in str(cell).replace(",", ";").split(";"):
        t = t.strip()
        if t and t.lower() not in {"na","nan","none"}: out.append(norm_ensp(t))
    seen = set(); uniq = []
    for t in out:
        if t not in seen: seen.add(t); uniq.append(t)
    return uniq

def normalize_cell_label(s):
    s = str(s).strip()
    return s[:-len("_edgelist")] if s.endswith("_edgelist") else s

def load_pickle_gz(path):
    with gzip.open(path, "rb") as f: return pickle.load(f)

def get_file_universe(path):
    universe = set()
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                parts = line.split()
                if len(parts) >= 2:
                    universe.add(norm_ensp(parts[0]))
                    universe.add(norm_ensp(parts[1]))
    except Exception as e:
        print(f"[WARN] {path}: {e}", file=sys.stderr)
    return universe

def load_pool_nodes(path, allowed):
    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    cols = df.columns.tolist()
    u, v = ("u","v") if "u" in cols and "v" in cols else (cols[0], cols[1])
    df = df[[u,v]].dropna()
    df[u] = df[u].map(norm_ensp); df[v] = df[v].map(norm_ensp)
    mask = df[u].isin(allowed) & df[v].isin(allowed)
    return set(df[u][mask]) | set(df[v][mask])

def sample_neg(rng, pool, k, forbidden, used, max_tries, avoid=None):
    pool2 = [x for x in pool if x not in avoid] if avoid else pool
    if len(pool2) < k: return None
    for _ in range(max_tries):
        cand = frozenset(rng.sample(pool2, k))
        if cand not in forbidden and cand not in used: return set(cand)
    return None

def step1_assign(complexes_csv, members_col, ppi_dir, files_like, keep_cols, out_csv):
    files = [os.path.join(ppi_dir, fn)
             for fn in sorted(os.listdir(ppi_dir))
             if fn.endswith("_edgelist.txt") and
             (not files_like or re.search(files_like, fn))]
    if not files: sys.exit(f"[ERROR] No edgelists in {ppi_dir}")

    df = pd.read_csv(complexes_csv, dtype=str)
    default_keep = ["row_index","ComplexID","Complex_name"]
    extra = [c.strip() for c in keep_cols.split(",") if c.strip()]
    kcols = list(dict.fromkeys(default_keep + extra))

    complex_data = []
    for _, row in df.iterrows():
        members = split_members(row.get(members_col,""))
        if not members: continue
        base = {c: row[c] for c in kcols if c in df.columns}
        base["members_norm"] = ";".join(members)
        base["n_members"]    = len(members)
        complex_data.append({"base": base, "set": set(members), "n": len(members)})

    out_rows = []
    for path in files:
        cell_name = os.path.splitext(os.path.basename(path))[0]
        universe  = get_file_universe(path)
        if not universe: continue
        for c_obj in complex_data:
            if c_obj["set"].issubset(universe):
                r = dict(c_obj["base"])
                r.update({"cell_type": cell_name, "status": "co-expressed",
                           "present_count": c_obj["n"], "coverage": 1.0})
                out_rows.append(r)

    if not out_rows: print("[WARN] No co-expressed matches found.")
    else:
        pd.DataFrame(out_rows).to_csv(out_csv, index=False)
        print(f"[Step 1] Wrote {len(out_rows)} assignments -> {out_csv}")

def step2_generate(assign_csv, ppi_dir, allowed, neg_per_pos, max_tries,
                    avoid_pos_members, dedupe_globally, seed, out_csv, summary_json):
    rng = random.Random(seed)
    df  = pd.read_csv(assign_csv, dtype=str)
    positives = []
    for _, r in df.iterrows():
        if str(r.get("status","")).strip() != "co-expressed": continue
        members = split_members(r.get("members_norm", r.get("subunits_ensp","")))
        members = [m for m in members if m in allowed]
        if len(members) < 2: continue
        positives.append({"cell_type_raw": str(r.get("cell_type","")).strip(),
                           "members": members, "k": len(members),
                           "complex_name": r.get("complex_name",""),
                           "complex_id": r.get("complex_id", r.get("ComplexID","")),
                           "row_index": r.get("row_index","")})

    if not positives: sys.exit("[ERROR] No positives after allowed filter.")
    print(f"[Step 2] {len(positives)} positives.")

    forbidden = set(frozenset(p["members"]) for p in positives)
    by_cell   = defaultdict(list)
    for p in positives:
        p["cell_norm"] = normalize_cell_label(p["cell_type_raw"])
        by_cell[p["cell_norm"]].append(p)

    edgelists = {}
    for fn in os.listdir(ppi_dir):
        if not fn.endswith(".txt"): continue
        stem = os.path.splitext(fn)[0]; path = os.path.join(ppi_dir, fn)
        edgelists[stem] = path
        if stem.endswith("_edgelist"): edgelists[stem[:-9]] = path

    out_rows, summary, global_used = [], {"total_positives": len(positives), "total_negatives": 0, "cells": {}}, set()

    for i, p in enumerate(positives):
        p["_key"] = f"pos_{i}"
        out_rows.append({"pos_key": p["_key"], "complex_id": p["complex_id"],
            "complex_name": p["complex_name"], "cell_type": p["cell_type_raw"],
            "label": 1, "members_ensp": ";".join(sorted(p["members"])),
            "k": p["k"], "source_row_index": p["row_index"], "notes": "co-expressed"})

    for cell, plist in by_cell.items():
        ppi_path = edgelists.get(cell) or edgelists.get(f"{cell}_edgelist")
        if not ppi_path:
            summary["cells"][cell] = {"positives": len(plist), "negatives": 0, "warning": "missing_edgelist"}
            continue
        pool = sorted(load_pool_nodes(ppi_path, allowed))
        neg_count = 0
        for p in plist:
            k = int(p["k"]); avoid = set(p["members"]) if avoid_pos_members else None
            used_here = set(); produced = 0
            while produced < neg_per_pos:
                cand = sample_neg(rng, pool, k, forbidden, used_here, max_tries, avoid)
                if cand is None: break
                fc = frozenset(cand); used_here.add(fc)
                if dedupe_globally:
                    if fc in global_used: continue
                    global_used.add(fc)
                out_rows.append({"pos_key": p["_key"], "complex_id": p["complex_id"],
                    "complex_name": p["complex_name"], "cell_type": p["cell_type_raw"],
                    "label": 0, "members_ensp": ";".join(sorted(cand)),
                    "k": k, "source_row_index": p["row_index"], "notes": "neg_samecell_samek"})
                produced += 1; neg_count += 1
        summary["cells"][cell] = {"positives": len(plist), "negatives": neg_count}
        summary["total_negatives"] += neg_count

    pd.DataFrame(out_rows).to_csv(out_csv, index=False)
    with open(summary_json, "w") as f: json.dump(summary, f, indent=2)
    print(f"[Step 2] Wrote {len(out_rows)} rows -> {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexes",         required=True)
    ap.add_argument("--members-col",       default="subunits_ensp")
    ap.add_argument("--ppi-dir",           required=True)
    ap.add_argument("--protrek-pkl",       required=True)
    ap.add_argument("--out-dir",           required=True)
    ap.add_argument("--keep-cols",         default="complex_name,subunits_ensp")
    ap.add_argument("--files-like",        default=None)
    ap.add_argument("--neg-per-pos",       type=int, default=10)
    ap.add_argument("--neg-max-tries",     type=int, default=20000)
    ap.add_argument("--avoid-pos-members", action="store_true")
    ap.add_argument("--dedupe-globally",   action="store_true")
    ap.add_argument("--seed",              type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    assign_csv   = os.path.join(args.out_dir, "assignments.csv")
    posneg_csv   = os.path.join(args.out_dir, "posneg.csv")
    summary_json = os.path.join(args.out_dir, "posneg_summary.json")

    allowed = set(norm_ensp(k) for k in load_pickle_gz(args.protrek_pkl).keys())
    step1_assign(args.complexes, args.members_col, args.ppi_dir,
                 args.files_like, args.keep_cols, assign_csv)
    step2_generate(assign_csv, args.ppi_dir, allowed,
                   args.neg_per_pos, args.neg_max_tries,
                   args.avoid_pos_members, args.dedupe_globally,
                   args.seed, posneg_csv, summary_json)
    print(f"\n[DONE] posneg.csv ready at {posneg_csv}")

if __name__ == "__main__":
    main()
