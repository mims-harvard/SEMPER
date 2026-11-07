#!/usr/bin/env python3
"""LLM-only baseline ("LLM-only" in Fig 5d-e, 5f-j and 6). No SEMPER retrieval.

Two calls per prompt, in one conversation:
  1. the prompt as is            -> free-form answer   (llm_<tag>_<seeds>_<idx>.json)
  2. "Output only the N proteins ..." -> bare list    (llm_listonly_<tag>_<seeds>_<idx>.json)
The list-only file is the ranked prediction that is scored.

Presets (prompt wording, N and API version as run for the paper):
  --preset disease   Fig 5f-j / Fig 6. Input = cell type-described prompts from
                     make_llmonly_prompts.py. N = 25, ranked list. <tag> = cell
                     type name with '_' -> '-' (e.g. Splat-367).
  --preset da        Fig 5d-e. Input = DA essential-gene prompts asking for 75
                     proteins. N = 75. <tag> = DA.

    python llm_only_baseline.py --preset disease \
        --input_json inputs/prompts/generated_pd_messages_llmonly_zeroshotprompt_10prompt_t010.json \
        --output_dir outputs/llm_only/PD

    python llm_only_baseline.py --preset da \
        --input_json inputs/prompts/generated_prompts_395_top25_DA_llm_DAfull_minus_DAtop100_75.json \
        --output_dir outputs/llm_only/DA
"""

import argparse
import json
import os
import re

from common import PAPER_MODEL, make_client

PRESETS = {
    "disease": {
        "api_version": "2025-03-01-preview",
        "followup": "Output only the 25 proteins as a ranked list, in this format "
                    "'A, B, C, D', without extra words.",
    },
    "da": {
        "api_version": "2025-04-01-preview",
        "followup": "Output only the 75 proteins as a list, in this format "
                    "'A, B, C, D', without extra words.",
    },
}

gene_pattern = re.compile(r"Given these proteins (.+?) that are known")
celltype_pattern = re.compile(r"Cell type: (\S+_\d+)")   # e.g. "Splat_367"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default=PAPER_MODEL)
    ap.add_argument("--api_version", default=None, help="default: the preset's version")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    client = make_client(args.api_version or preset["api_version"])
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.input_json) as f:
        prompts = json.load(f)

    for idx, prompt in enumerate(prompts):
        if any(f.startswith("llm_listonly_") and f.endswith(f"_{idx}.json")
               for f in os.listdir(args.output_dir)):
            continue

        content = prompt["content"]
        gene_match = gene_pattern.search(content)
        if not gene_match:
            print(f"idx {idx}: no seed genes found; skipping")
            continue
        seeds = [g.strip() for g in gene_match.group(1).split(",")]
        seed_str = "_".join(g.replace("/", "-") for g in seeds[:5])
        if args.preset == "da":
            tag = "DA"
        else:
            ct_match = celltype_pattern.search(content)
            tag = ct_match.group(1).replace("_", "-") if ct_match else "NA"

        try:
            messages = [dict(prompt)]

            r1 = client.chat.completions.create(model=args.model, messages=messages)
            free_msg = r1.choices[0].message
            with open(os.path.join(args.output_dir, f"llm_{tag}_{seed_str}_{idx}.json"),
                      "w", encoding="utf-8") as out_f:
                out_f.write((free_msg.content or "").strip())

            messages.append({"role": "assistant", "content": free_msg.content or ""})
            messages.append({"role": "user", "content": preset["followup"]})
            r2 = client.chat.completions.create(model=args.model, messages=messages)
            list_msg = r2.choices[0].message
            with open(os.path.join(args.output_dir, f"llm_listonly_{tag}_{seed_str}_{idx}.json"),
                      "w", encoding="utf-8") as out_f:
                out_f.write((list_msg.content or "").strip())
            print(f"idx {idx}: {tag}")

        except Exception as e:
            print(f"idx {idx}: {e}")


if __name__ == "__main__":
    main()
