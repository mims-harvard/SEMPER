#!/usr/bin/env python3
"""
prepare_hpa_colocalized.py
---------------------------
Panel e: end-to-end data preparation for zero-shot complex identification
using the definition: complex members are CO-EXPRESSED in the cell type AND
CO-LOCALIZED to the same subcellular compartment via Human Protein Atlas
annotations.

Output status: "colocalized+coexpressed"

Data requirements:
  HPA subcellular localization TSV — download from:
    https://www.proteinatlas.org/about/download
    File: subcellular_location.tsv.zip

Usage:
    python prepare_hpa_colocalized.py \
        --complexes    data/filtered_complexes_min3.csv \
        --hpa          data/subcellular_location_data_ensp.tsv \
        --ppi-dir      data/ppi_edgelists/ \
        --protrek-pkl  data/ensp_id2vec_mean.pkl.gz \
        --out-dir      data/prepared/hpa_colocalized/ \
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
    if not s or s.lower() in {"na","nan","none"}: return ""
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
    return [x for x in uniq if x]

def split_locations(loc_str):
    if pd.isna(loc_str): return set()
    s = str(loc_str).strip()
    if not s or s.lower() in {"na","nan","none"}: return set()
    return {x.strip() for x in s.split(";") if x.strip()}

def normalize_cell_label(s):
    s = str(s).strip()
    return s[:-len("_edgelist")] if s.endswith("_edgelist") else s

def load_pickle_gz(path):
    with gzip.open(path, "rb") as f: return pickle.load(f)

def detect_hpa_ensp_col(df, user_col=""):
    if user_col:
        if user_col in df.columns: return user_col
        for c in df.columns:
            if c.lower() == user_col.lower(): return c
        sys.exit(f"[ERROR] --hpa-ensp-col '{user_col}' not found.")
    for c in ["ENSP","ensp","Protein","Protein ID","Ensembl protein ID"]:
        if c in df.columns: return c
    for c in df.columns:
        if "ensp" in c.lower(): return c
    sys.exit("[ERROR] Cannot auto-detect ENSP column in HPA. Use --hpa-ensp-col.")

def get_file_universe(path):
    universe = set()
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("#") or not line.strip(): continue
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

def sample_neg_colocalized(rng, pool_by_loc, k, forbidden, used, max_tries, avoid=None):
    valid_locs = []
    filtered = {}
    for loc, prots in pool_by_loc.items():
        vp = [p for p in prots if p not in avoid] if avoid else prots
        if len(vp) >= k:
            valid_locs.append(loc); filtered[loc] = vp
    if not valid_locs: return None
    for _ in range(max_tries):
        loc  = rng.choice(valid_locs)
        cand = frozenset(rng.sample(filtered[loc], k))
        if cand not in forbidden and cand not in used:
            return {"members": set(cand), "location": loc}
    return None

def step1_assign(complexes_csv, members_col, hpa_path, hpa_ensp_col, hpa_loc_cols,
                  ppi_dir, files_like, keep_cols, out_csv):
    # Load HPA
    sep = "\t" if hpa_path.endswith((".tsv",".txt")) else ","
    hpa_df = pd.read_csv(hpa_path, sep=sep, dtype=str)
    ensp_col = detect_hpa_ensp_col(hpa_df, hpa_ensp_col)
    loc_cols = [c.strip() for c in hpa_loc_cols.split(",") if c.strip() and c.strip() in hpa_df.columns]
    if not loc_cols: sys.exit("[ERROR] No valid HPA location columns found.")

    ensp_to_locs = defaultdict(set)
    for _, row in hpa_df.iterrows():
        pid = norm_ensp(row.get(ensp_col,""))
        if not pid: continue
        for c in loc_cols: ensp_to_locs[pid] |= split_locations(row.get(c,""))

    print(f"[Step 1] HPA: {len(ensp_to_locs)} proteins with location data.")

    df = pd.read_csv(complexes_csv, dtype=str)
    default_keep = ["row_index"]; extra = [c.strip() for c in keep_cols.split(",") if c.strip()]
    kcols = list(dict.fromkeys(default_keep + extra))

    valid_complexes = []
    for _, row in df.iterrows():
        members = split_members(row.get(members_col,""))
        if not members: continue
        lsets = [ensp_to_locs.get(m, set()) for m in members]
        if any(not s for s in lsets): continue
        common = set.intersection(*lsets)
        if not common: continue
        base = {c: row[c] for c in kcols if c in df.columns}
        base["members_norm"] = ";".join(members)
        base["n_members"]    = len(members)
        base["shared_locations"] = ";".join(sorted(common))
        valid_complexes.append({"base": base, "set": set(members), "n": len(members)})

    print(f"[Step 1] {len(valid_complexes)} complexes satisfy HPA spatial constraints.")

    files = [os.path.join(ppi_dir, fn)
             for fn in sorted(os.listdir(ppi_dir))
             if fn.endswith("_edgelist.txt") and (not files_like or re.search(files_like, fn))]
    if not files: sys.exit(f"[ERROR] No edgelists in {ppi_dir}")

    out_rows = []
    for path in files:
        cell_name = os.path.splitext(os.path.basename(path))[0]
        universe  = get_file_universe(path)
        if not universe: continue
        for c_obj in valid_complexes:
            if c_obj["set"].issubset(universe):
                r = dict(c_obj["base"])
                r.update({"cell_type": cell_name, "status": "colocalized+coexpressed",
                           "present_count": c_obj["n"], "coverage": 1.0})
                out_rows.append(r)

    if not out_rows: print("[WARN] No colocalized+coexpressed matches.")
    else:
        out = pd.DataFrame(out_rows)
        first = ["cell_type","status","shared_locations"]
        rest  = [c for c in out.columns if c not in first]
        out[first + rest].to_csv(out_csv, index=False)
        print(f"[Step 1] Wrote {len(out_rows)} assignments -> {out_csv}")

def step2_generate(assign_csv, ppi_dir, hpa_path, hpa_ensp_col, hpa_loc_cols,
                    allowed, neg_per_pos, max_tries,
                    avoid_pos_members, dedupe_globally, seed, out_csv, summary_json):
    rng = random.Random(seed)

    sep = "\t" if hpa_path.endswith((".tsv",".txt")) else ","
    hpa_df = pd.read_csv(hpa_path, sep=sep, dtype=str)
    ensp_col = detect_hpa_ensp_col(hpa_df, hpa_ensp_col)
    loc_cols = [c.strip() for c in hpa_loc_cols.split(",") if c.strip() and c.strip() in hpa_df.columns]
    ensp_to_locs = defaultdict(set)
    for _, row in hpa_df.iterrows():
        pid = norm_ensp(row.get(ensp_col,""))
        if pid and pid in allowed:
            for c in loc_cols: ensp_to_locs[pid] |= split_locations(row.get(c,""))

    df = pd.read_csv(assign_csv, dtype=str)
    positives = []
    for _, r in df.iterrows():
        if str(r.get("status","")).strip() != "colocalized+coexpressed": continue
        members = split_members(r.get("members_norm", r.get("subunits_ensp","")))
        members = [m for m in members if m in allowed]
        if len(members) < 2: continue
        positives.append({"cell_type_raw": str(r.get("cell_type","")).strip(),
                           "members": members, "k": len(members),
                           "complex_name": r.get("complex_name",""),
                           "complex_id": r.get("complex_id", r.get("ComplexID","")),
                           "row_index": r.get("row_index",""),
                           "shared_locations": r.get("shared_locations","")})

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
            "k": p["k"], "source_row_index": p["row_index"],
            "shared_locations": p["shared_locations"], "notes": "colocalized+coexpressed"})

    for cell, plist in by_cell.items():
        ppi_path = edgelists.get(cell) or edgelists.get(f"{cell}_edgelist")
        if not ppi_path:
            summary["cells"][cell] = {"positives": len(plist), "negatives": 0, "warning": "missing_edgelist"}
            continue
        nodes_in_cell = load_pool_nodes(ppi_path, allowed)
        pool_by_loc = defaultdict(list)
        for node in nodes_in_cell:
            for loc in ensp_to_locs.get(node, set()): pool_by_loc[loc].append(node)

        neg_count = 0
        for p in plist:
            k = int(p["k"]); avoid = set(p["members"]) if avoid_pos_members else None
            used_here = set(); produced = 0
            while produced < neg_per_pos:
                result = sample_neg_colocalized(rng, pool_by_loc, k, forbidden, used_here, max_tries, avoid)
                if result is None: break
                fc = frozenset(result["members"]); used_here.add(fc)
                if dedupe_globally:
                    if fc in global_used: continue
                    global_used.add(fc)
                out_rows.append({"pos_key": p["_key"], "complex_id": p["complex_id"],
                    "complex_name": p["complex_name"], "cell_type": p["cell_type_raw"],
                    "label": 0, "members_ensp": ";".join(sorted(result["members"])),
                    "k": k, "source_row_index": p["row_index"],
                    "shared_locations": result["location"], "notes": "neg_colocalized"})
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
    ap.add_argument("--hpa",               required=True,
                    help="HPA subcellular location TSV. "
                         "Download: https://www.proteinatlas.org/about/download")
    ap.add_argument("--hpa-ensp-col",      default="")
    ap.add_argument("--hpa-location-cols", default="Main location,Additional location")
    ap.add_argument("--ppi-dir",           required=True)
    ap.add_argument("--protrek-pkl",       required=True)
    ap.add_argument("--out-dir",           required=True)
    ap.add_argument("--keep-cols",         default="ComplexID,Complex_name")
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
    step1_assign(args.complexes, args.members_col, args.hpa, args.hpa_ensp_col,
                 args.hpa_location_cols, args.ppi_dir, args.files_like, args.keep_cols, assign_csv)
    step2_generate(assign_csv, args.ppi_dir, args.hpa, args.hpa_ensp_col,
                   args.hpa_location_cols, allowed,
                   args.neg_per_pos, args.neg_max_tries,
                   args.avoid_pos_members, args.dedupe_globally,
                   args.seed, posneg_csv, summary_json)
    print(f"\n[DONE] posneg.csv ready at {posneg_csv}")

if __name__ == "__main__":
    main()
