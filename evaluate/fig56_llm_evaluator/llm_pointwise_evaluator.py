import openai
import subprocess
from openai import AzureOpenAI
import os
from tooluniverse import ContextualProteinRetrieval
import json
import re
import time
from math import ceil
import ast
import argparse as _ap
_parser = _ap.ArgumentParser()
_parser.add_argument("--input_json",  required=True,
                     help="Path to generated prompts JSON (e.g. generated_DA_messages_*.json)")
_parser.add_argument("--output_dir",  required=True,
                     help="Directory to write per-prompt reranked JSON outputs")
_parser.add_argument("--azure_endpoint", default="https://api.openai.com",
                     help="Azure OpenAI endpoint (default: OpenAI public API)")
_parser.add_argument("--api_version",    default="2025-03-01-preview")
_parser.add_argument("--model",          default="o3-mini",
                     help="Model to use for both tool call and judge steps")
_args = _parser.parse_args()
input_json  = _args.input_json
output_dir  = _args.output_dir


TOOL_REGISTRY = {
    "contextual_protein_token_retrieval": ContextualProteinRetrieval,
    # "other_tool_name": OtherToolClass,
}

function_schema =[ {
    "type":"function",
"function":{
    "name": "contextual_protein_token_retrieval",
    "description": "Tool to retrieve relevant multimodal context-aware protein representations by computing pairwise vector similarity (via similarity metric defined by user) between user-specified seed proteins and all others. Sort the resulting vector similarities in descending order (from most similar to least similar). Apply the user-specified threshold to obtain the top k most relevant proteins",
    "parameters": {
        "type": "object",
        "properties": {
            "tokens": {
                "type": "string",
                "description": "Path to .pth file containing contextual protein embeddings."
            },
            "tokenlabel": {
                "type": "string",
                "description": "Path to labels corresponding to the tokens."
            },
            "celltype": {
                "type": "string",
                "description": "Cell type context to filter embeddings by (e.g. 'cluster:100')."
            },
            "seeds": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of seed protein names (e.g., ['IL6', 'TNF'])"
            },
            "threshold": {
                "type": "integer",
                "description": "Number of top similar tokens to retrieve."
            },
            "similarity": {
                "type": "string",
                "enum": ["cosine"],
                "description": "Similarity metric to use."
            }
	},
	"required": ["tokens", "tokenlabel", "seeds", "threshold", "similarity", "celltype"]
    }
}
}]

with open(input_json, "r") as f:
    prompts = json.load(f)


def sanitize_quotes(s):
    return (
        s.replace("“", '"')
         .replace("”", '"')
         .replace("‘", "'")
         .replace("’", "'")
         .replace("‐", "-")
    )

def sanitize_unicode_quotes_and_dashes(s):
    return (
        s.replace("“", '"')
         .replace("”", '"')
         .replace("‘", "'")
         .replace("’", "'")
         .replace("–", "-")
         .replace("—", "-")
         .replace("‐", "-")
    )

def normalize_celltype(celltype_raw: str):
    s = str(celltype_raw).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        return s, re.sub(r"[^A-Za-z0-9._-]+", "_", s)  # fallback, still filename-safe

    cid = m.group(1)
    celltype_for_tool = f"cluster:{cid}"
    celltype_for_fname = f"cluster{cid}"
    return celltype_for_tool, celltype_for_fname


for idx, prompt in enumerate(prompts):
    print(f"🔄 Processing prompt {idx+1}/{len(prompts)}")
    messages = [prompt]

    try:
        openai = AzureOpenAI(
            azure_endpoint=_args.azure_endpoint,
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=_args.api_version
        )

        response = openai.chat.completions.create(
            model=_args.model,
            messages=messages,
            tools=function_schema
        )

        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name


            tool_args = json.loads(sanitize_quotes(tool_call.function.arguments))
            print(f"\n GPT selected tool: {tool_name}")
            print("Arguments:", tool_args)

            if tool_name not in TOOL_REGISTRY:
                raise ValueError(f"Unknown tool: {tool_name}")

            tool_cls = TOOL_REGISTRY[tool_name]
            tool_instance = tool_cls(tool_config={})
            tool_result = tool_instance.run(tool_args)


            threshold = tool_args.get("threshold", "top")
            celltype_raw = tool_args.get("celltype", "na")

            celltype_tool, celltype_tag = normalize_celltype(celltype_raw)
            tool_args["celltype"] = celltype_tool  # <- ensures tool always sees "cluster:<id>"

            seeds = tool_args.get("seeds", [])[:5]
            seed_str = "_".join(gene.replace("/", "-") for gene in seeds)

            filename = f"tool_{threshold}_{celltype_tag}_{seed_str}_{idx}_LLMJUDGE.json"

            output_path = os.path.join(output_dir, filename)

            print(tool_result)
            if isinstance(tool_result, dict) and "results" in tool_result:
                protein_list = tool_result["results"]
            elif isinstance(tool_result, list):
                protein_list = tool_result
            else:
                raise ValueError("tool_result has unexpected structure:", type(tool_result))

            full_rationale_list = []
            similarity_lookup = {p["protein"]: p["similarity"] for p in protein_list}
            protein_names = list(similarity_lookup.keys())

            batch_size = 100
            num_batches = ceil(len(protein_names) / batch_size)

            for batch_idx in range(num_batches):
                batch_proteins = protein_names[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                batch_data = [{"protein": name, "similarity": similarity_lookup[name]} for name in batch_proteins]
    
                judge_prompt = [
    {
        "role": "system",
        "content": "You are a biomedical expert ranking proteins for their relevance to a disease context."
    },
    {
        "role": "user",
        "content": f"""
Given these proteins {', '.join(seeds)} that are known essential genes in hPSC-derived dopamine neurons (i.e., genes for dopaminergic neuron survival),
below is a list of proteins retrieved with their similarity scores to the seed proteins:

{json.dumps(batch_data, indent=2)}

For each protein, return a JSON object like:
[
  {{
    "protein": "PROTEIN_NAME",
    "rationale": "A string with 1-2 sentences that explain the protein’s relevance, if any, to the given context",
    "rationale_score": 1-5
  }},
  ...
]

Your output must use only plain ASCII characters.
"""
    }
                ]

                judge_response = openai.chat.completions.create(
                    model=_args.model,
                    messages=judge_prompt
                )
                content = sanitize_unicode_quotes_and_dashes(judge_response.choices[0].message.content)

                try:
                    rationale_list = json.loads(content)

                except json.JSONDecodeError:
                    rationale_list = ast.literal_eval(content)

                full_rationale_list.extend(rationale_list)

            reranked_list = []

            for r in full_rationale_list:
                name = r["protein"]
                rationale = r.get("rationale", "")
                rationale_score = r.get("rationale_score", 1)
                similarity = similarity_lookup.get(name, 0.0)
                combined_score = 0.5 * similarity + 0.5 * ((rationale_score - 1) / 4)

                reranked_list.append({
                    "protein": name,
                    "similarity": similarity,
                    "rationale_score": rationale_score,
                    "combined_score": combined_score,
                    "rationale": rationale
                })

            reranked_list.sort(key=lambda x: x["combined_score"], reverse=True)

            with open(output_path, "w") as out_f:
                json.dump(reranked_list, out_f, indent=2)

            print(f"✅ Saved reranked list to: {output_path}")

    except Exception as e:
        print(f"❌ Error on prompt {idx+1}: {e}")
