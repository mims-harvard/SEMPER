#!/usr/bin/env python3
"""Pointwise LLM judge for dopamine-neuron essential genes ("SEMPER+LLM" in Fig 5d-e).

For every prompt (20 seed essential genes):
  1. SEMPER ranks the FULL candidate pool of one cell type by cosine similarity
     to the mean seed representation (semper_retrieval.SemperRetriever).
  2. The pool is sent to the LLM in batches of 100, in similarity order. Each
     request lists the first 5 seeds and asks for a 1-5 relevance score per protein.
     A batch whose reply cannot be parsed keeps its proteins with score 1
     (they then rank by similarity alone).
  3. Rerank by combined_score = 0.5 * similarity + 0.5 * (rationale_score - 1) / 4.

--celltype must be the Siletti cluster ID, e.g. cluster:395 (Splat_395, the
dopaminergic cell type). It is NOT the integer embedding key. The paper's run
passed the embedding key 319 (which is the key of cluster:395) as if it were a
cluster ID, so its candidate pools came from cluster:319. Use
--celltype cluster:319 to rerun what was published, cluster:395 for the intended
cell type.

    python llm_pointwise_da_judge.py \
        --input_json  inputs/prompts/generated_prompts_395_top25_DA_llm_DAfull_minus_DAtop100_75.json \
        --celltype    cluster:395 \
        --embed_path  ../data/semper_embed.pth \
        --label_path  ../data/semper_labels.txt \
        --output_dir  outputs/da_judge/cluster395

Prompts can be split across a Slurm job array: task t of N handles prompts with
idx % N == t (SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT).
"""

import argparse
import ast
import json
import os
from math import ceil

from common import PAPER_MODEL, make_client, sanitize
from semper_retrieval import DEFAULT_ENSP_TO_GENE, SemperRetriever, normalize_celltype, parse_seeds

API_VERSION = "2025-04-01-preview"
BATCH_SIZE = 100


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_json", required=True,
                    help="DA prompts; seeds are parsed from 'Given these proteins ... that are known'.")
    ap.add_argument("--celltype", required=True,
                    help="Siletti cluster ID, e.g. cluster:395 (not the embedding index).")
    ap.add_argument("--embed_path", required=True, help="semper_embed.pth")
    ap.add_argument("--label_path", required=True, help="semper_labels.txt")
    ap.add_argument("--ensp_to_gene", default=DEFAULT_ENSP_TO_GENE)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default=PAPER_MODEL)
    ap.add_argument("--api_version", default=API_VERSION)
    args = ap.parse_args()

    celltype = normalize_celltype(args.celltype)
    celltype_tag = celltype.replace(":", "")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.input_json) as f:
        prompts = json.load(f)

    shard = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    nshard = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))

    client = make_client(args.api_version)
    retriever = SemperRetriever(args.embed_path, args.label_path, args.ensp_to_gene)

    for idx, prompt in enumerate(prompts):
        if nshard > 1 and idx % nshard != shard:
            continue
        if any(f.endswith(f"_{idx}_LLMJUDGE.json") for f in os.listdir(args.output_dir)):
            continue
        print(f"Processing prompt {idx+1}/{len(prompts)}")

        try:
            all_seeds = parse_seeds(prompt["content"])
            protein_list = retriever.retrieve(celltype, all_seeds)
            seeds = all_seeds[:5]
            seed_str = "_".join(gene.replace("/", "-") for gene in seeds)
            output_path = os.path.join(
                args.output_dir, f"tool_top_{celltype_tag}_{seed_str}_{idx}_LLMJUDGE.json")

            similarity_lookup = {p["protein"]: p["similarity"] for p in protein_list}
            protein_names = list(similarity_lookup.keys())

            full_rationale_list = []
            for batch_idx in range(ceil(len(protein_names) / BATCH_SIZE)):
                batch_proteins = protein_names[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
                batch_data = [{"protein": name, "similarity": similarity_lookup[name]}
                              for name in batch_proteins]

                judge_prompt = [
                    {"role": "system",
                     "content": "You are a biomedical expert ranking proteins for their relevance to a disease context."},
                    {"role": "user",
                     "content": f"""
Given these proteins {', '.join(seeds)} that are known essential genes in hPSC-derived dopamine neurons (i.e., genes for dopaminergic neuron survival),
below is a list of proteins retrieved with their similarity scores to the seed proteins:

{json.dumps(batch_data, indent=2)}

For each protein, return a JSON object like:
[
  {{
    "protein": "PROTEIN_NAME",
    "rationale": "A string with 1-2 sentences that explain the protein's relevance, if any, to the given context",
    "rationale_score": 1-5
  }},
  ...
]

Your output must use only plain ASCII characters.
"""}
                ]

                try:
                    judge_response = client.chat.completions.create(model=args.model, messages=judge_prompt)
                    content = sanitize(judge_response.choices[0].message.content)
                    try:
                        rationale_list = json.loads(content)
                    except json.JSONDecodeError:
                        rationale_list = ast.literal_eval(content)
                    full_rationale_list.extend(rationale_list)
                except Exception as be:
                    print(f"  batch {batch_idx} unparseable ({be}); similarity-only fallback "
                          f"for {len(batch_proteins)} proteins")
                    full_rationale_list.extend(
                        {"protein": name, "rationale": "", "rationale_score": 1} for name in batch_proteins)

            reranked_list = []
            for r in full_rationale_list:
                name = r["protein"]
                rationale_score = r.get("rationale_score", 1)
                similarity = similarity_lookup.get(name, 0.0)
                reranked_list.append({
                    "protein": name,
                    "similarity": similarity,
                    "rationale_score": rationale_score,
                    "combined_score": 0.5 * similarity + 0.5 * ((rationale_score - 1) / 4),
                    "rationale": r.get("rationale", ""),
                })
            reranked_list.sort(key=lambda x: x["combined_score"], reverse=True)
            with open(output_path, "w") as out_f:
                json.dump(reranked_list, out_f, indent=2)
            print(f"Saved {output_path}")

        except Exception as e:
            print(f"Error on prompt {idx+1}: {e}")


if __name__ == "__main__":
    main()
