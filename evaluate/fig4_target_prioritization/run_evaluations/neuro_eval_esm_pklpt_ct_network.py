#!/usr/bin/env python3
import gzip
import argparse
import json
import pickle
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

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

def load_ot_json(pd_gene_file: str) -> dict:
    with open(pd_gene_file, "r") as f:
        return json.load(f)

def load_ct_protein_lists_from_label_dict(label_dict_path: str) -> Dict[str, List[str]]:
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

    # Build CCI ordering
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

    for idx, cci_label in idx_to_cci.items():
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
            ensp = full_id.split(".", 1)[-1] if "." in str(full_id) else str(full_id)
            ensp = norm_ensp(ensp)
            if ensp:
                ensp_list.append(ensp)

        out[normalize_cluster_id(cci_label)] = ensp_list

    if len(out) == 0:
        raise RuntimeError(f"Could not parse any CT protein lists from {label_dict_path}")
    return out

def load_esm_from_pkl_pt(esm_pickle: str, esm_embeddings: str) -> Dict[str, torch.Tensor]:
    with open(esm_pickle, "rb") as f:
        protein_ids = pickle.load(f)  # {prot_str: idx}

    emb = torch.load(esm_embeddings, map_location="cpu")
    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb)
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb)

    if isinstance(protein_ids, dict):
        index_to_protein = {int(v): k for k, v in protein_ids.items()}
    else:
        raise RuntimeError("esm_pickle must be a dict {prot_str: idx_int}")

    out: Dict[str, torch.Tensor] = {}
    n = int(emb.shape[0])
    for i in range(n):
        prot = index_to_protein.get(i)
        if prot is None:
            continue
        prot = norm_ensp(prot)
        if not prot:
            continue
        vec = emb[i]
        if vec.ndim != 1:
            vec = vec.reshape(-1)
        out[prot] = vec.detach().cpu().float()

    if len(out) == 0:
        raise RuntimeError("No embeddings loaded. Check that indices in pkl match rows in pt.")
    return out

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

def main():
    ap = argparse.ArgumentParser(description="Evaluate global ESM pkl+pt on CT-network candidates, seeds from seedplan, positives from OT HGNC.")

    ap.add_argument("--seedplan", required=True)
    ap.add_argument("--ensp_json", required=True)

    ap.add_argument("--pd_gene_file", required=True)
    ap.add_argument("--ot_key", default="clinically_relevant_targets")

    ap.add_argument("--label_dict", required=True, help="labels_dict used to define CT network proteins (candidate universe)")
    ap.add_argument("--esm_pickle", required=True)
    ap.add_argument("--esm_embeddings", required=True)
    ap.add_argument("--model_name", default="ESM")

    ap.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 25])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out_prefix", default="neuro_eval_ESM_ctnet_OTpos")

    ap.add_argument("--skip_if_missing_seed", action="store_true")
    ap.add_argument("--max_candidates", type=int, default=0)

    args = ap.parse_args()

    # seedplan (seeds only)
    with (gzip.open(args.seedplan, "rt") if args.seedplan.endswith(".gz") else open(args.seedplan, "r")) as f:
        seedplan = json.load(f)
    clusters = seedplan["clusters"]

    # mapping ENSP->HGNC
    with open(args.ensp_json, "r") as f:
        ensp_to_hgnc_raw = json.load(f)
    ensp_to_hgnc = {norm_ensp(k): norm_hgnc(v) for k, v in ensp_to_hgnc_raw.items() if v}

    ot_json = load_ot_json(args.pd_gene_file)
    if args.ot_key not in ot_json:
        raise SystemExit(f"[ERROR] Key '{args.ot_key}' not found in {args.pd_gene_file}. Keys: {list(ot_json.keys())[:20]}")
    ot_hgnc = set(norm_hgnc(g) for g in ot_json[args.ot_key] if g)
    print(f"[INFO] OT HGNC loaded: n={len(ot_hgnc)} (key={args.ot_key})")

    print("[INFO] Loading CT protein lists from label_dict...")
    ct_to_proteins = load_ct_protein_lists_from_label_dict(args.label_dict)
    print(f"[INFO] CTs found in label_dict: {len(ct_to_proteins)}")

    print("[INFO] Loading ESM embeddings (pkl+pt)...")
    esm_dict = load_esm_from_pkl_pt(args.esm_pickle, args.esm_embeddings)
    dim = next(iter(esm_dict.values())).numel()
    print(f"[INFO] Loaded {len(esm_dict)} ESM vectors; dim={dim}")

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
            cand = [e for e in ct_prots if e in esm_dict]
            if args.max_candidates and len(cand) > args.max_candidates:
                cand = cand[: args.max_candidates]

            if len(cand) == 0:
                skip_counts["missing_candidate_embeddings"] += len(trials)
                continue

            E_all = torch.stack([esm_dict[e] for e in cand]).to(device)

            relevant_hgnc = set()
            for e in cand:
                h = ensp_to_hgnc.get(e)
                if h and h in ot_hgnc:
                    relevant_hgnc.add(h)

            cand_hgnc = [ensp_to_hgnc.get(e, None) for e in cand]

            cluster_debug_rows.append({
                "cluster": cluster_id,
                "n_candidates_ct_total": len(ct_prots),
                "n_candidates_with_esm": len(cand),
                "n_relevant_hgnc_in_candidates": len(relevant_hgnc),
                "n_trials": len(trials),
            })

            for tr in trials:
                trial_id = tr.get("trial")
                seed_ensp = [norm_ensp(x) for x in tr.get("seed_ensp", []) if norm_ensp(x)]

                seeds_present = [e for e in seed_ensp if e in esm_dict]
                if args.skip_if_missing_seed and len(seeds_present) != len(seed_ensp):
                    skip_counts["missing_seed_embeddings"] += 1
                    continue
                if len(seeds_present) == 0:
                    skip_counts["missing_seed_embeddings"] += 1
                    continue

                seed_vecs = torch.stack([esm_dict[e] for e in seeds_present]).to(device)

                scores = cosine_rank_candidates(E_all, seed_vecs)

                # exclude seeds themselves
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
                        "n_ct_candidates_with_esm": len(cand),
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
                n_ct_candidates_with_esm=("n_ct_candidates_with_esm", "first"),
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
        "esm_pickle": args.esm_pickle,
        "esm_embeddings": args.esm_embeddings,
        "device": str(device),
        "k_values": args.k_values,
        "counts": {
            "n_clusters_in_seedplan": len(clusters),
            "n_trial_rows": len(trial_rows),
            **skip_counts
        },
        "coverage": {
            "n_ct_in_label_dict": len(ct_to_proteins),
            "n_esm_embeddings": len(esm_dict),
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
