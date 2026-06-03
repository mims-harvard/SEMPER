#!/usr/bin/env python3
"""
neuro_eval_graphlets_global_ct_network.py

Global graphlet-orbit (ORCA) evaluator in the SAME "CT network" candidate setting:
- Seeds still come from seedplan trials (per cluster/cell type).
- Candidates come from the *cell-type network protein set* for that cluster, derived from the
  same SEMPER/PINNACLE labels_dict (so universe is CT-specific, not ProTrek).
- Embeddings are global graphlet orbits (transposenet_orbits.txt + transposenet_nodes.json),
  then subset to CT candidates.
- Positives come from an OpenTargets disease gene JSON (HGNC list), intersected with CT candidates.

Example:
python neuro_eval_graphlets_global_ct_network.py \
  --seedplan neuro_seedplan.protrek_intersection.json \
  --ensp_json /n/holylfs06/LABS/mzitnik_lab/Lab/adong/TransposeNetv3_data/TransposeNet_v3/ENSP_to_pref.json \
  --pd_gene_file /n/holylfs06/LABS/mzitnik_lab/Lab/adong/PINNACLE_newTransposeNet_DDP_tests/PD_genelist_similarity_test/neurodegen_disease/raw_targets_EFO_0000618.json \
  --ot_key clinically_relevant_targets \
  --label_dict /n/holylfs06/LABS/mzitnik_lab/Lab/adong/SEMPER_v3TransposeNet_tests/.../pinnacle_projected_labels.txt \
  --node-map /n/holylfs06/LABS/mzitnik_lab/Lab/adong/SEMPER_v3TransposeNet_tests/fig3_CORUMcomplexes/transposenet_nodes.json \
  --orbits /n/holylfs06/LABS/mzitnik_lab/Lab/adong/CORUM_benchmarking/protrek_analysis/graphlets_train/transposenet_orbits.txt \
  --model_name Graphlets_ORCA \
  --out_prefix neuro_GraphletsORCA_global_ctnet_OTpos \
  --device cpu \
  --k_values 5 10 25 \
  --skip_if_missing_seed

Notes:
- This script intentionally ignores payload["universe_ensp"] and payload["positive_ensp"].
- It uses label_dict to define CT candidate proteins, seedplan ONLY for seed trials, and OT for positives.
"""

import argparse
import json
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


# ----------------------------
# ID normalization
# ----------------------------

def norm_ensp(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    if s.startswith("9606."):
        s = s[len("9606."):]
    if "." in s:
        s = s.split(".", 1)[-1]
    m = re.match(r"^(ENSP\d+)", s)
    return m.group(1) if m else s


def norm_hgnc(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().upper()


# ----------------------------
# Load OpenTargets disease genes (HGNC)
# ----------------------------

def load_ot_json(pd_gene_file: str) -> dict:
    with open(pd_gene_file, "r") as f:
        return json.load(f)


# ----------------------------
# Load CT protein lists from labels_dict
# ----------------------------

def load_ct_protein_lists_from_label_dict(label_dict_path: str) -> Dict[str, List[str]]:
    """
    Returns: cluster_id(str) -> list of ENSP proteins in that CT network.
    """
    with open(label_dict_path, "r") as f:
        labels_raw = f.read()
    labels_raw = labels_raw.replace("'", "\"")
    labels_dict = json.loads(labels_raw)

    protein_names = []
    protein_celltypes = []
    for c, p in zip(labels_dict["Cell Type"], labels_dict["Name"]):
        c = str(c)
        if c.startswith("BTO") or c.startswith("Sanity"):
            continue
        if c.startswith("CCI"):
            continue
        protein_names.append(p)
        protein_celltypes.append(c)

    proteins = pd.DataFrame({"target": protein_names, "cell_type": protein_celltypes})
    celltype_protein_dict = (
        proteins.pivot_table(values="target", index="cell_type", aggfunc={"target": list})
        .to_dict()["target"]
    )

    cci_cells = [str(c) for c in labels_dict["Cell Type"] if str(c).startswith("CCI_")]
    seen = set()
    cci_cells_unique = []
    for c in cci_cells:
        if c not in seen:
            seen.add(c)
            cci_cells_unique.append(c)

    idx_to_cci = {i: ct for i, ct in enumerate(cci_cells_unique)}

    def normalize_cluster_id(cci_label: str) -> str:
        s = cci_label
        if s.startswith("CCI_"):
            s = s[len("CCI_"):]
        if s.startswith("cluster:"):
            s = s.split(":", 1)[1]
        return s

    out: Dict[str, List[str]] = {}

    for _, cci_label in idx_to_cci.items():
        prot_list_key = None
        if cci_label in celltype_protein_dict:
            prot_list_key = cci_label
        else:
            stripped = cci_label[len("CCI_"):] if cci_label.startswith("CCI_") else cci_label
            if stripped in celltype_protein_dict:
                prot_list_key = stripped
        if prot_list_key is None:
            continue

        full_ids = celltype_protein_dict[prot_list_key]
        ensp_list = []
        for full_id in full_ids:
            ensp = str(full_id).split(".", 1)[-1] if "." in str(full_id) else str(full_id)
            ensp = norm_ensp(ensp)
            if ensp:
                ensp_list.append(ensp)

        out[normalize_cluster_id(cci_label)] = ensp_list

    if len(out) == 0:
        raise RuntimeError(f"Could not parse any CT protein lists from {label_dict_path}")
    return out


# ----------------------------
# Graphlets loading (ORCA global)
# ----------------------------

def load_node_map(node_map_path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    with open(node_map_path, "r") as f:
        node_map = json.load(f)
    idx_to_protein = {int(k): v for k, v in node_map["idx_to_protein"].items()}
    protein_to_idx = {k: int(v) for k, v in node_map["protein_to_idx"].items()}
    return idx_to_protein, protein_to_idx


def load_orbits_dict(orbits_path: str, idx_to_protein: Dict[int, str]) -> Dict[str, torch.Tensor]:
    """
    ORCA orbit counts: row i corresponds to node i (0-based).
    Returns dict: {ENSP -> torch.FloatTensor[n_orbits]}
    """
    mat = np.loadtxt(orbits_path)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D orbit matrix, got shape {mat.shape}")

    n_nodes, _ = mat.shape
    out: Dict[str, torch.Tensor] = {}

    for i in range(n_nodes):
        if i not in idx_to_protein:
            raise ValueError(f"Index {i} missing from idx_to_protein in node map.")
        pid_raw = idx_to_protein[i]
        pid = norm_ensp(pid_raw)
        if not pid:
            continue
        out[pid] = torch.tensor(mat[i], dtype=torch.float32)

    if len(out) == 0:
        raise RuntimeError("No graphlet embeddings parsed. Check node map IDs and orbit file.")
    return out


# ----------------------------
# Metrics / ranking
# ----------------------------

def apr_at_k(ranked_hgnc: List[str], relevant_set: set, k: int) -> float:
    num_relevant_in_data = len(relevant_set.intersection(ranked_hgnc))
    if num_relevant_in_data == 0:
        return 0.0
    precision_sum = 0.0
    hits = 0
    upto = min(k, len(ranked_hgnc))
    for i in range(upto):
        if ranked_hgnc[i] in relevant_set:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / num_relevant_in_data


def cosine_rank_candidates(E_all: torch.Tensor, seed_vecs: torch.Tensor) -> torch.Tensor:
    E = E_all / (E_all.norm(dim=1, keepdim=True) + 1e-12)
    S = seed_vecs / (seed_vecs.norm(dim=1, keepdim=True) + 1e-12)
    return (E @ S.t()).mean(dim=1)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate global graphlets (ORCA) on CT-network candidates, seeds from seedplan, positives from OT HGNC.")

    ap.add_argument("--seedplan", required=True)
    ap.add_argument("--ensp_json", required=True)

    ap.add_argument("--pd_gene_file", required=True)
    ap.add_argument("--ot_key", default="clinically_relevant_targets")

    ap.add_argument("--label_dict", required=True, help="labels_dict used to define CT network proteins (candidate universe)")

    ap.add_argument("--orbits", required=True, help="ORCA orbits output (transposenet_orbits.txt)")
    ap.add_argument("--node-map", required=True, help="Node map JSON (transposenet_nodes.json)")
    ap.add_argument("--model_name", default="Graphlets_Global")

    ap.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 25])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out_prefix", default="neuro_eval_GraphletsGlobal_ctnet_OTpos")

    ap.add_argument("--skip_if_missing_seed", action="store_true")
    ap.add_argument("--max_candidates", type=int, default=0)

    args = ap.parse_args()

    # seedplan (seeds only)
    with open(args.seedplan, "r") as f:
        seedplan = json.load(f)
    clusters = seedplan["clusters"]

    # mapping ENSP->HGNC
    with open(args.ensp_json, "r") as f:
        ensp_to_hgnc_raw = json.load(f)
    ensp_to_hgnc = {norm_ensp(k): norm_hgnc(v) for k, v in ensp_to_hgnc_raw.items() if v}

    # OT HGNC
    ot_json = load_ot_json(args.pd_gene_file)
    if args.ot_key not in ot_json:
        raise SystemExit(f"[ERROR] Key '{args.ot_key}' not found in {args.pd_gene_file}. Keys: {list(ot_json.keys())[:20]}")
    ot_hgnc = set(norm_hgnc(g) for g in ot_json[args.ot_key] if g)
    print(f"[INFO] OT HGNC loaded: n={len(ot_hgnc)} (key={args.ot_key})")

    # CT candidate protein lists
    print("[INFO] Loading CT protein lists from label_dict...")
    ct_to_proteins = load_ct_protein_lists_from_label_dict(args.label_dict)
    print(f"[INFO] CTs found in label_dict: {len(ct_to_proteins)}")

    # Global graphlet embeddings
    print("[INFO] Loading node map + ORCA orbits (global)...")
    idx_to_protein, _ = load_node_map(args.node_map)
    emb_dict = load_orbits_dict(args.orbits, idx_to_protein)
    dim = next(iter(emb_dict.values())).numel()
    print(f"[INFO] Loaded {len(emb_dict)} graphlet vectors; dim={dim}")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device={device}")

    trial_rows = []
    skip_counts = {
        "missing_cluster_in_label_dict": 0,
        "missing_candidate_embeddings": 0,
        "missing_seed_embeddings": 0,
        "ok_trials": 0
    }

    cluster_debug_rows = []

    with torch.inference_mode():
        for cluster_id, payload in clusters.items():
            trials = payload.get("trials", [])

            if cluster_id not in ct_to_proteins:
                skip_counts["missing_cluster_in_label_dict"] += len(trials)
                continue

            ct_prots = [norm_ensp(x) for x in ct_to_proteins[cluster_id]]
            cand = [e for e in ct_prots if e in emb_dict]
            if args.max_candidates and len(cand) > args.max_candidates:
                cand = cand[: args.max_candidates]

            if len(cand) == 0:
                skip_counts["missing_candidate_embeddings"] += len(trials)
                continue

            E_all = torch.stack([emb_dict[e] for e in cand]).to(device)

            # Relevant HGNC = OT genes present in this CT candidate set
            relevant_hgnc = set()
            for e in cand:
                h = ensp_to_hgnc.get(e)
                if h and h in ot_hgnc:
                    relevant_hgnc.add(h)

            cand_hgnc = [ensp_to_hgnc.get(e, None) for e in cand]

            cluster_debug_rows.append({
                "cluster": cluster_id,
                "n_candidates_ct_total": len(ct_prots),
                "n_candidates_with_graphlets": len(cand),
                "n_relevant_hgnc_in_candidates": len(relevant_hgnc),
                "n_trials": len(trials),
            })

            for tr in trials:
                trial_id = tr.get("trial")
                seed_ensp = [norm_ensp(x) for x in tr.get("seed_ensp", []) if norm_ensp(x)]

                seeds_present = [e for e in seed_ensp if e in emb_dict]
                if args.skip_if_missing_seed and len(seeds_present) != len(seed_ensp):
                    skip_counts["missing_seed_embeddings"] += 1
                    continue
                if len(seeds_present) == 0:
                    skip_counts["missing_seed_embeddings"] += 1
                    continue

                seed_vecs = torch.stack([emb_dict[e] for e in seeds_present]).to(device)

                scores = cosine_rank_candidates(E_all, seed_vecs)

                seed_set = set(seed_ensp)
                idxs = [i for i, e in enumerate(cand) if e not in seed_set]
                if len(idxs) == 0:
                    skip_counts["missing_candidate_embeddings"] += 1
                    continue

                scores_sub = scores[torch.tensor(idxs, device=device)]
                order = torch.argsort(scores_sub, descending=True).tolist()
                ranked_idxs = [idxs[i] for i in order]

                ranked_hgnc = []
                for i in ranked_idxs:
                    h = cand_hgnc[i]
                    if h:
                        ranked_hgnc.append(h)

                for k in args.k_values:
                    top_k = ranked_hgnc[:k]
                    hits_k = sum(1 for g in top_k if g in relevant_hgnc)
                    apr_k = apr_at_k(ranked_hgnc, relevant_hgnc, k)

                    denom = min(k, len(relevant_hgnc)) if len(relevant_hgnc) > 0 else 0
                    recovery_pct = (100.0 * hits_k / denom) if denom > 0 else 0.0

                    trial_rows.append({
                        "model": args.model_name,
                        "cluster": cluster_id,
                        "trial": trial_id,
                        "k": k,
                        "hits_at_k": hits_k,
                        "apr_at_k": apr_k,
                        "recovery_pct": recovery_pct,
                        "recovery_denom": denom,
                        "n_candidates": len(idxs),
                        "n_seeds_used": len(seeds_present),
                        "n_relevant_hgnc_in_candidates": len(relevant_hgnc),
                        "n_ct_candidates_total": len(ct_prots),
                        "n_ct_candidates_with_graphlets": len(cand),
                    })

                skip_counts["ok_trials"] += 1

    trials_df = pd.DataFrame(trial_rows)
    out_trials = f"{args.out_prefix}__trials.csv"
    trials_df.to_csv(out_trials, index=False)

    if len(trials_df) > 0:
        summary_df = (
            trials_df
            .groupby(["model", "cluster", "k"], as_index=False)
            .agg(
                hits_mean=("hits_at_k", "mean"),
                hits_std=("hits_at_k", "std"),
                apr_mean=("apr_at_k", "mean"),
                apr_std=("apr_at_k", "std"),
                recovery_mean=("recovery_pct", "mean"),
                recovery_std=("recovery_pct", "std"),
                n_trials=("trial", "nunique"),
                n_candidates_mean=("n_candidates", "mean"),
                n_seeds_used_mean=("n_seeds_used", "mean"),
                n_relevant_hgnc_in_candidates=("n_relevant_hgnc_in_candidates", "first"),
                n_ct_candidates_total=("n_ct_candidates_total", "first"),
                n_ct_candidates_with_graphlets=("n_ct_candidates_with_graphlets", "first"),
            )
        )
    else:
        summary_df = pd.DataFrame()

    out_summary = f"{args.out_prefix}__summary.csv"
    summary_df.to_csv(out_summary, index=False)

    debug_df = pd.DataFrame(cluster_debug_rows)
    out_debug = f"{args.out_prefix}__cluster_debug.csv"
    debug_df.to_csv(out_debug, index=False)

    summary_json = {
        "model": args.model_name,
        "seedplan": args.seedplan,
        "mapping_json": args.ensp_json,
        "pd_gene_file": args.pd_gene_file,
        "ot_key": args.ot_key,
        "label_dict": args.label_dict,
        "orbits": args.orbits,
        "node_map": args.node_map,
        "device": str(device),
        "k_values": args.k_values,
        "counts": {
            "n_clusters_in_seedplan": len(clusters),
            "n_trial_rows": len(trial_rows),
            **skip_counts
        },
        "coverage": {
            "n_ct_in_label_dict": len(ct_to_proteins),
            "n_graphlet_embeddings": len(emb_dict),
            "embedding_dim": int(dim),
            "n_ot_hgnc": len(ot_hgnc),
        }
    }
    out_json = f"{args.out_prefix}__summary.json"
    with open(out_json, "w") as f:
        json.dump(summary_json, f, indent=2)

    print(f"[DONE] trials -> {out_trials}")
    print(f"[DONE] summary -> {out_summary}")
    print(f"[DONE] cluster debug -> {out_debug}")
    print(f"[DONE] summary json -> {out_json}")
    print("[COUNTS]", json.dumps(summary_json["counts"], indent=2))


if __name__ == "__main__":
    main()
