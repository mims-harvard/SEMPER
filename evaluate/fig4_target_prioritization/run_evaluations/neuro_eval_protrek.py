import os, json, argparse, gzip, pickle
from typing import Dict, List, Any

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
    if s.startswith("9606."):
        s = s[len("9606."):]
    return s


def load_dict_pickle_gz(path: str) -> dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def strip_9606_prefix(d: dict) -> dict:
    out = {}
    collisions = 0
    for k, v in d.items():
        kk = k
        if isinstance(kk, str) and kk.startswith("9606."):
            kk = kk.replace("9606.", "", 1)
        if kk in out and kk != k:
            collisions += 1
        out[kk] = v
    if collisions:
        print(f"[WARN] {collisions} key collisions after stripping '9606.' (last one wins).")
    return out


def to_torch_dict(d: dict) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in d.items():
        pid = norm_ensp(k)
        if not pid:
            continue

        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().float()
        elif isinstance(v, np.ndarray):
            t = torch.from_numpy(v).float().cpu()
        else:
            # list/tuple or scalar-ish
            t = torch.tensor(v, dtype=torch.float32).cpu()

        if t.ndim != 1:
            t = t.reshape(-1)
        out[pid] = t
    if len(out) == 0:
        raise RuntimeError("No ProTrek embeddings parsed after normalization.")
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
    ap = argparse.ArgumentParser()

    ap.add_argument("--seedplan", required=True, help="neuro_seedplan.protrek_intersection.json")
    ap.add_argument("--ensp_json", required=True, help="ENSP->HGNC mapping json")

    # ProTrek input
    ap.add_argument("--protrek_gz", required=True, help="ensp_id2vec_mean.pkl.gz (or structure variant)")
    ap.add_argument("--model_name", default="ProTrek", help="Name used in output CSVs")

    ap.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 25])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out_prefix", default="neuro_eval_ProTrek_protrekU")

    ap.add_argument("--skip_if_missing_seed", action="store_true",
                    help="If set, skip a trial if any seed protein missing from embeddings.")
    ap.add_argument("--max_candidates", type=int, default=0,
                    help="Optional cap on candidate universe size per cluster (0 = no cap).")

    args = ap.parse_args()
    
    with open(args.seedplan, "r") as f:
        seedplan = json.load(f)
    clusters = seedplan["clusters"]

    # Load mapping
    with open(args.ensp_json, "r") as f:
        ensp_to_hgnc_raw = json.load(f)
    ensp_to_hgnc = {
        norm_ensp(k): str(v).strip().upper()
        for k, v in ensp_to_hgnc_raw.items() if v
    }

    # Load ProTrek embeddings
    print("[INFO] Loading ProTrek embeddings...")
    raw = load_dict_pickle_gz(args.protrek_gz)
    raw = strip_9606_prefix(raw)
    emb_dict = to_torch_dict(raw)
    print(f"[INFO] Loaded {len(emb_dict)} ProTrek vectors.")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    trial_rows = []
    skip_counts = {
        "missing_candidate_embeddings": 0,
        "missing_seed_embeddings": 0,
        "ok_trials": 0
    }

    # Instead of per-cluster universes, we use ALL loaded proteins
    global_cand = list(emb_dict.keys())
    print(f"[INFO] Building global candidate tensor with {len(global_cand)} proteins...")
    
    E_all_global = torch.stack([emb_dict[e] for e in global_cand]).to(device)
    cand_hgnc_global = [ensp_to_hgnc.get(e, None) for e in global_cand]

    with torch.inference_mode():
        for cluster_id, payload in clusters.items():
            positives = [norm_ensp(x) for x in payload["positive_ensp"]]
            trials = payload["trials"]

            relevant_hgnc = set()
            for e in positives:
                h = ensp_to_hgnc.get(e)
                if h:
                    relevant_hgnc.add(h)

            for tr in trials:
                trial_id = tr["trial"]
                seed_ensp = [norm_ensp(x) for x in tr["seed_ensp"]]

                seeds_present = [e for e in seed_ensp if e in emb_dict]
                if args.skip_if_missing_seed and len(seeds_present) != len(seed_ensp):
                    skip_counts["missing_seed_embeddings"] += 1
                    continue
                if len(seeds_present) == 0:
                    skip_counts["missing_seed_embeddings"] += 1
                    continue

                seed_vecs = torch.stack([emb_dict[e] for e in seeds_present]).to(device)

                scores = cosine_rank_candidates(E_all_global, seed_vecs)

                # Exclude the seeds themselves from ranking
                seed_set = set(seed_ensp)
                idxs = [i for i, e in enumerate(global_cand) if e not in seed_set]
                
                if len(idxs) == 0:
                    skip_counts["missing_candidate_embeddings"] += 1
                    continue

                idxs_t = torch.tensor(idxs, device=device)
                scores_sub = scores[idxs_t]
                
                # Rank the remaining global candidates
                order = torch.argsort(scores_sub, descending=True).tolist()
                ranked_idxs = [idxs[i] for i in order]

                ranked_hgnc = []
                for i in ranked_idxs:
                    h = cand_hgnc_global[i]
                    if h:
                        ranked_hgnc.append(h)

                for k in args.k_values:
                    top_k = ranked_hgnc[:k]
                    hits_k = sum(1 for g in top_k if g in relevant_hgnc)
                    apr_k = apr_at_k(ranked_hgnc, relevant_hgnc, k)

                    trial_rows.append({
                        "model": args.model_name,
                        "cluster": cluster_id,
                        "trial": trial_id,
                        "k": k,
                        "hits_at_k": hits_k,
                        "apr_at_k": apr_k,
                        "n_candidates": len(idxs),
                        "n_seeds_used": len(seeds_present),
                        "n_relevant_hgnc": len(relevant_hgnc),
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
                n_trials=("trial", "nunique"),
                n_candidates_mean=("n_candidates", "mean"),
                n_seeds_used_mean=("n_seeds_used", "mean"),
                n_relevant_hgnc=("n_relevant_hgnc", "first"),
            )
        )
    else:
        summary_df = pd.DataFrame()

    out_summary = f"{args.out_prefix}__summary.csv"
    summary_df.to_csv(out_summary, index=False)

    summary_json = {
        "model": args.model_name,
        "seedplan": args.seedplan,
        "protrek_gz": args.protrek_gz,
        "device": str(device),
        "k_values": args.k_values,
        "counts": {
            "n_clusters_in_seedplan": len(clusters),
            "n_trial_rows": len(trial_rows),
            **skip_counts
        },
        "embedding_coverage": {
            "n_protrek_embeddings": len(emb_dict),
        }
    }
    out_json = f"{args.out_prefix}__summary.json"
    with open(out_json, "w") as f:
        json.dump(summary_json, f, indent=2)

    print(f"[DONE] trials -> {out_trials}")
    print(f"[DONE] summary -> {out_summary}")
    print(f"[DONE] summary json -> {out_json}")
    print("[COUNTS]", json.dumps(summary_json["counts"], indent=2))


if __name__ == "__main__":
    main()
