#!/usr/bin/env python3
"""
neuro_eval_semper_ct_network.py

SEMPER-ESM / SEMPER-ESMCLIP evaluator using:
- Seeds from seedplan JSON (same as before)
- Candidates from SEMPER cell-type-specific embedding space (NOT ProTrek universe)
- Positives from an OpenTargets-style disease gene JSON (HGNC list), mapped into each CT candidate set

Example:
python neuro_eval_semper_ct_network.py \
  --seedplan neuro_seedplan.protrek_intersection.json \
  --ensp_json /n/holylfs06/LABS/mzitnik_lab/Lab/adong/TransposeNetv3_data/TransposeNet_v3/ENSP_to_pref.json \
  --pd_gene_file /n/holylfs06/LABS/mzitnik_lab/Lab/adong/PINNACLE_newTransposeNet_DDP_tests/PD_genelist_similarity_test/neurodegen_disease/raw_targets_EFO_0000618.json \
  --embed_path /n/holylfs06/LABS/mzitnik_lab/Lab/adong/SEMPER_v3TransposeNet_tests/.../esm_projected.pth \
  --label_dict /n/holylfs06/LABS/mzitnik_lab/Lab/adong/SEMPER_v3TransposeNet_tests/.../esm_projected_labels.txt \
  --model_name SEMPER_ESMCLIP \
  --out_prefix neuro_SEMPER_ESMCLIP_ctnet_OTpos \
  --device cuda \
  --k_values 5 10 25 \
  --skip_if_missing_seed

Notes:
- This script intentionally ignores payload["universe_ensp"] and payload["positive_ensp"].
- It uses seedplan ONLY for seed assignment (trials).
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
    s = str(x).strip().upper()
    return s


# ----------------------------
# Load OpenTargets disease genes (HGNC)
# ----------------------------

def load_ot_hgnc_set(pd_gene_file: str) -> set:
    """
    Expected JSON like:
      { "clinically_relevant_targets": ["LRRK2", "SNCA", ...], ... }

    If your key differs, pass --ot_key.
    """
    with open(pd_gene_file, "r") as f:
        d = json.load(f)
    return d


# ----------------------------
# Load SEMPER-style embeddings
# ----------------------------

def load_celltype_embeddings(embed_path: str, label_dict_path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Robust loader:
    - Uses the EXACT 'Cell Type' string from labels_dict to index the protein list
    - Handles cell types like: 'CCI_10', 'CCI_cluster:0', 'CCI_cluster:123'
    - Returns per_cell keyed by normalized cluster id: '0', '1', '10', '123', ...
    """
    embed = torch.load(embed_path, map_location="cpu")

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

    celltype_map = {i: ct for i, ct in enumerate(cci_cells_unique)}

    def normalize_cluster_id(cci_str: str) -> str:
        s = cci_str
        if s.startswith("CCI_"):
            s = s[len("CCI_"):]
        if s.startswith("cluster:"):
            s = s.split(":", 1)[1]
        return s

    per_cell: Dict[str, Dict[str, torch.Tensor]] = {}

    for celltype_key, tensor in embed.items():
        if isinstance(celltype_key, (int, np.integer)):
            if int(celltype_key) not in celltype_map:
                continue
            cci_label = celltype_map[int(celltype_key)]
        else:
            cci_label = str(celltype_key)
            if not cci_label.startswith("CCI_"):
                cci_label = f"CCI_{cci_label}"

        if cci_label in celltype_protein_dict:
            prot_list_key = cci_label
        else:
            stripped = cci_label[len("CCI_"):] if cci_label.startswith("CCI_") else cci_label
            if stripped in celltype_protein_dict:
                prot_list_key = stripped
            else:
                continue

        all_protein_names = celltype_protein_dict[prot_list_key]
        n = min(tensor.shape[0], len(all_protein_names))

        cluster_dict: Dict[str, torch.Tensor] = {}
        for i in range(n):
            full_id = all_protein_names[i]
            ensp_id = full_id.split(".", 1)[-1] if "." in full_id else full_id
            if ensp_id.startswith("9606."):
                ensp_id = ensp_id[5:]
            ensp_id = norm_ensp(ensp_id)
            if not ensp_id:
                continue
            v = tensor[i].detach().cpu()
            if v.ndim != 1:
                v = v.reshape(-1)
            cluster_dict[ensp_id] = v.float()

        cluster_id = normalize_cluster_id(cci_label)
        per_cell[cluster_id] = cluster_dict

    if len(per_cell) == 0:
        raise RuntimeError(f"No celltype embeddings loaded from {embed_path} with labels {label_dict_path}")
    return per_cell


# ----------------------------
# Ranking + metrics
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


def safe_mean(x: List[float]) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate SEMPER-ESM/ESMCLIP using CT-network candidates + OT positives, seeds from seedplan.")

    ap.add_argument("--seedplan", required=True, help="Seedplan JSON used ONLY for seed assignment (trials).")
    ap.add_argument("--ensp_json", required=True, help="ENSP->HGNC mapping json")
    ap.add_argument("--pd_gene_file", required=True, help="OpenTargets-style disease gene JSON (HGNC list)")

    ap.add_argument("--ot_key", default="clinically_relevant_targets",
                    help="Key in --pd_gene_file that contains the HGNC list (default: clinically_relevant_targets).")

    ap.add_argument("--embed_path", required=True, help="SEMPER-ESM embed .pth (celltype dict)")
    ap.add_argument("--label_dict", required=True, help="labels_dict for that embed")

    ap.add_argument("--model_name", default="SEMPER_ESM", help="Name used in output CSVs")

    ap.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 25])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out_prefix", default="neuro_eval_SEMPER_ctnet_OTpos")

    ap.add_argument("--skip_if_missing_seed", action="store_true",
                    help="If set, skip a trial if any seed protein missing from this CT's embeddings.")
    ap.add_argument("--max_candidates", type=int, default=0,
                    help="Optional cap on candidate pool size per cluster (0 = no cap).")

    args = ap.parse_args()

    # seedplan (seeds only)
    with open(args.seedplan, "r") as f:
        seedplan = json.load(f)
    clusters = seedplan["clusters"]

    # mapping ENSP->HGNC
    with open(args.ensp_json, "r") as f:
        ensp_to_hgnc_raw = json.load(f)
    ensp_to_hgnc = {norm_ensp(k): norm_hgnc(v) for k, v in ensp_to_hgnc_raw.items() if v}

    # OT genes (HGNC)
    ot_json = load_ot_hgnc_set(args.pd_gene_file)
    if args.ot_key not in ot_json:
        raise SystemExit(f"[ERROR] Key '{args.ot_key}' not found in {args.pd_gene_file}. Keys: {list(ot_json.keys())[:20]}")
    ot_hgnc = set(norm_hgnc(g) for g in ot_json[args.ot_key] if g)

    print(f"[INFO] OT HGNC loaded: n={len(ot_hgnc)} (key={args.ot_key})")

    # SEMPER embeddings
    print("[INFO] Loading SEMPER CT embeddings...")
    per_cell = load_celltype_embeddings(args.embed_path, args.label_dict)
    print(f"[INFO] Loaded CTs: {len(per_cell)} from embed.")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device={device}")

    trial_rows = []
    skip_counts = {
        "missing_cluster_in_model": 0,
        "missing_candidate_embeddings": 0,
        "missing_seed_embeddings": 0,
        "ok_trials": 0
    }

    # helpful debug per cluster
    cluster_debug_rows = []

    with torch.inference_mode():
        for cluster_id, payload in clusters.items():
            trials = payload.get("trials", [])
            if cluster_id not in per_cell:
                skip_counts["missing_cluster_in_model"] += len(trials)
                continue

            emb_dict = per_cell[cluster_id]

            # Candidates = all proteins in this CT embedding (SEMPER CT network)
            cand = list(emb_dict.keys())
            if args.max_candidates and len(cand) > args.max_candidates:
                cand = cand[: args.max_candidates]

            if len(cand) == 0:
                skip_counts["missing_candidate_embeddings"] += len(trials)
                continue

            # Candidate tensor
            E_all = torch.stack([emb_dict[e] for e in cand]).to(device)

            # Relevant HGNC set = OT genes that are present in THIS CT candidate set
            relevant_hgnc = set()
            for e in cand:
                h = ensp_to_hgnc.get(e)
                if h and h in ot_hgnc:
                    relevant_hgnc.add(h)

            cand_hgnc = [ensp_to_hgnc.get(e, None) for e in cand]

            cluster_debug_rows.append({
                "cluster": cluster_id,
                "n_candidates": len(cand),
                "n_relevant_hgnc_in_candidates": len(relevant_hgnc),
                "n_trials": len(trials),
            })

            for tr in trials:
                trial_id = tr.get("trial")
                seed_ensp = [norm_ensp(x) for x in tr.get("seed_ensp", []) if norm_ensp(x)]

                # Seed existence check in CT embedding
                seeds_present = [e for e in seed_ensp if e in emb_dict]
                if args.skip_if_missing_seed and len(seeds_present) != len(seed_ensp):
                    skip_counts["missing_seed_embeddings"] += 1
                    continue
                if len(seeds_present) == 0:
                    skip_counts["missing_seed_embeddings"] += 1
                    continue

                seed_vecs = torch.stack([emb_dict[e] for e in seeds_present]).to(device)

                scores = cosine_rank_candidates(E_all, seed_vecs)

                # Exclude seeds from ranking
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

                    # optional recovery_pct definition (consistent across universes):
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
                    })

                skip_counts["ok_trials"] += 1

    # write outputs
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
        "embed_path": args.embed_path,
        "label_dict": args.label_dict,
        "device": str(device),
        "k_values": args.k_values,
        "counts": {
            "n_clusters_in_seedplan": len(clusters),
            "n_trial_rows": len(trial_rows),
            **skip_counts
        },
        "coverage": {
            "n_ct_in_embed": len(per_cell),
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
