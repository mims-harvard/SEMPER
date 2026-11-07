#!/usr/bin/env python3
"""Cell type-aware pointwise LLM judge ("SEMPER+LLM" in Fig 5f-j).

For every prompt, the judge receives SEMPER's top-100 candidates (from
semper_retrieval.py) for one disease and one cell type, together with a
description of that cell type (name, class, supercluster, neurotransmitter
identity, top brain regions, marker genes) and all 15 seed targets. All 100
candidates go in ONE request. The judge scores each candidate 1-5 for relevance
to the disease in that cell type, and candidates are reranked by

    combined_score = 0.5 * similarity + 0.5 * (rationale_score - 1) / 4

The candidate pool is SEMPER's; only the order changes.

    python llm_pointwise_context_judge.py \
        --input_json    inputs/prompts/generated_pd_messages_celltypecluster_zeroshotprompt_10prompt_t010.json \
        --cache_dir     outputs/retrieval/PD \
        --output_dir    outputs/context_judge/PD \
        --disease       pd

Writes one JSON per prompt (same file name as the retrieval cache) with
protein, similarity, rationale_score, combined_score, rationale.
"""

import argparse
import ast
import json
import os
import re

from common import (PAPER_MODEL, format_celltype_context, load_celltype_metadata,
                    make_client, resolve_disease, sanitize)

HERE = os.path.dirname(os.path.abspath(__file__))
API_VERSION = "2025-03-01-preview"

cluster_pat = re.compile(r"celltype (cluster:\w+)", re.IGNORECASE)
seed_pat = re.compile(r"Given these proteins (.+?) that are")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_json", required=True, help="SEMPER prompts (seeds + 'celltype cluster:N').")
    ap.add_argument("--cache_dir", required=True, help="Output directory of semper_retrieval.py.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--disease", required=True,
                    help="pd, ad, ms, als or hd (uses the paper's disease names), or a free-text name.")
    ap.add_argument("--celltype_meta", default=os.path.join(HERE, "inputs", "celltype_metadata_top10.csv"))
    ap.add_argument("--model", default=PAPER_MODEL)
    ap.add_argument("--api_version", default=API_VERSION)
    args = ap.parse_args()

    disease_name = resolve_disease(args.disease)
    meta = load_celltype_metadata(args.celltype_meta)
    client = make_client(args.api_version)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_json) as f:
        prompts = json.load(f)
    print(f"{disease_name}: {len(prompts)} prompts -> {args.output_dir}")

    for idx, prompt in enumerate(prompts):
        if any(f.endswith(f"_{idx}.json") for f in os.listdir(args.output_dir)):
            continue
        content = prompt["content"]
        sm = seed_pat.search(content)
        seeds = [p.strip() for p in sm.group(1).split(",")] if sm else []
        cm = cluster_pat.search(content)
        cid = cm.group(1) if cm else None
        ctx = format_celltype_context(meta[cid]) if cid in meta else ""

        matching = [f for f in os.listdir(args.cache_dir) if f.endswith(f"_{idx}.json")]
        if not matching:
            print(f"No cached retrieval for idx {idx}")
            continue
        cached_file = matching[0]
        with open(os.path.join(args.cache_dir, cached_file)) as f:
            protein_list = json.load(f)

        try:
            judge_prompt = [
                {"role": "system",
                 "content": "You are a biomedical expert ranking proteins for relevance to a "
                            "disease in a SPECIFIC brain cell type."},
                {"role": "user",
                 "content": f"""
Given these proteins {', '.join(seeds)} that are known targets for {disease_name} in the following human brain cell type:
{ctx}

Below is a list of proteins retrieved with their similarity scores to the seed proteins. For each protein, score its relevance to {disease_name} IN THIS SPECIFIC CELL TYPE -- i.e. taking the cell type's class, neurotransmitter identity, brain regions, and marker genes into account, not generic disease relevance.

{json.dumps(protein_list[:100], indent=2)}

For each protein, return a JSON object like:
[
  {{
    "protein": "PROTEIN_NAME",
    "rationale": "1-2 sentences on relevance to {disease_name} in this cell type, if any",
    "rationale_score": 1-5
  }},
  ...
]

Your output must use only plain ASCII characters.
"""}]

            resp = sanitize(client.chat.completions.create(
                model=args.model, messages=judge_prompt).choices[0].message.content)
            try:
                rationale_list = json.loads(resp)
            except json.JSONDecodeError:
                rationale_list = ast.literal_eval(resp)

            sim_lookup = {p["protein"]: p["similarity"] for p in protein_list}
            reranked = []
            for r in rationale_list:
                name = r["protein"]
                rs = r.get("rationale_score", 1)
                sim = sim_lookup.get(name, 0.0)
                reranked.append({
                    "protein": name, "similarity": sim, "rationale_score": rs,
                    "combined_score": 0.5 * sim + 0.5 * ((rs - 1) / 4),
                    "rationale": r.get("rationale", ""),
                })
            reranked.sort(key=lambda x: x["combined_score"], reverse=True)

            with open(os.path.join(args.output_dir, cached_file), "w") as out_f:
                json.dump(reranked, out_f, indent=2)
            print(f"idx {idx} ({cid}): saved {cached_file}")

        except Exception as e:
            print(f"Error on prompt {idx}: {e}")


if __name__ == "__main__":
    main()
