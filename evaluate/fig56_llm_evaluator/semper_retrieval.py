#!/usr/bin/env python3
"""SEMPER contextual protein retrieval (no LLM).

Ranks every protein in one cell type by cosine similarity to the mean SEMPER
representation of the seed proteins. Seeds absent from that cell type's network
are dropped, and the remaining seeds are excluded from the candidates. The full
ranked list is returned (no top-k cut).

This is the same computation as ToolUniverse's ContextualProteinRetrieval tool
(tooluniverse/targetnom_tool.py) that produced the candidate pools in the paper,
written out here so the pipeline does not depend on that package or its
hard-coded mapping paths.

Cell types are addressed by their Siletti et al. (2023) cluster ID, written
'cluster:395' or '395'. This is NOT the integer key of semper_embed.pth. The
integer key is the position of 'CCI_cluster:N' in semper_labels.txt and is
looked up here. (The Fig 5d prompts passed the key 319 where cluster 395 was
meant, so that run searched cluster:319. See README.)

Batch mode, one cached list per prompt (Stage 2 of Fig 5f-j and Fig 6):

    python semper_retrieval.py \
        --prompts_json inputs/prompts/generated_pd_messages_celltypecluster_zeroshotprompt_10prompt_t010.json \
        --embed_path   ../data/semper_embed.pth \
        --label_path   ../data/semper_labels.txt \
        --output_dir   outputs/retrieval/PD

Each output is tool_top_<celltype>_<first 5 seeds>_<idx>.json, a list of
{"protein": HGNC, "similarity": float} sorted by similarity.
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENSP_TO_GENE = os.path.join(HERE, "..", "data", "ensp_to_gene.json")

SEED_PATTERN = re.compile(r"Given these proteins (.+?) that are known")
CLUSTER_PATTERN = re.compile(r"celltype (cluster:\w+)", re.IGNORECASE)


def normalize_celltype(value):
    """'395' / 'cluster:395' / 'CCI_cluster:395' -> 'cluster:395'."""
    ctx = str(value).strip()
    if ctx.startswith("CCI_"):
        ctx = ctx.split("CCI_", 1)[1]
    if ctx.isdigit():
        ctx = f"cluster:{ctx}"
    return ctx


def parse_seeds(content):
    m = SEED_PATTERN.search(content)
    return [g.strip() for g in m.group(1).split(",")] if m else None


class SemperRetriever:
    def __init__(self, embed_path, label_path, ensp_to_gene_path=DEFAULT_ENSP_TO_GENE):
        with open(label_path) as f:
            labels = json.loads(f.read().replace("'", '"'))

        # 'cluster:N' -> integer key of the embedding file (order of CCI rows)
        cci = [c for c in labels["Cell Type"] if c.startswith("CCI")]
        self.celltype_to_key = {ct.split("CCI_")[1]: i for i, ct in enumerate(cci)}
        key_to_celltype = {v: k for k, v in self.celltype_to_key.items()}

        # protein rows per cell type, in file order (row i <-> embedding row i)
        names = defaultdict(list)
        for c, p in zip(labels["Cell Type"], labels["Name"]):
            if c.startswith("BTO") or c.startswith("CCI") or c.startswith("Sanity"):
                continue
            names[str(c)].append(str(p))

        print(f"Loading SEMPER representations from {embed_path} ...")
        self.embed = torch.load(embed_path, map_location="cpu")
        self.names = {k: names[key_to_celltype[k]] for k in self.embed}
        self._cache = {}

        with open(ensp_to_gene_path) as f:
            raw = json.load(f)
        self.ensp_to_hgnc = {k.replace("9606.", ""): v.strip().upper() for k, v in raw.items()}
        self.hgnc_to_ensp = {v: k for k, v in self.ensp_to_hgnc.items()}

    def _token_dict(self, key):
        if key not in self._cache:
            tensor = self.embed[key]
            d = {}
            for i in range(tensor.shape[0]):
                full_id = self.names[key][i]
                d[full_id.split(".", 1)[1] if "." in full_id else full_id] = tensor[i]
            self._cache = {key: d}          # keep one cell type in memory
        return self._cache[key]

    def retrieve(self, celltype, seeds):
        ctx = normalize_celltype(celltype)
        if ctx not in self.celltype_to_key:
            raise ValueError(f"Unknown cell type '{celltype}'. Expected a Siletti cluster ID "
                             f"such as 'cluster:395' or '395'.")
        token_dict = self._token_dict(self.celltype_to_key[ctx])

        missing = [s for s in seeds if s.strip().upper() not in self.hgnc_to_ensp]
        if missing:
            raise KeyError(f"Seed symbols not in the ENSP->gene map: {missing}")
        seed_ensp = [self.hgnc_to_ensp[s.strip().upper()] for s in seeds]
        seed_ensp = [p for p in seed_ensp if p in token_dict]
        if not seed_ensp:
            raise ValueError(f"None of the seeds are in the {ctx} network.")

        mean_seed = torch.stack([token_dict[p] for p in seed_ensp]).mean(dim=0).unsqueeze(0)
        candidates = [k for k in token_dict if k not in seed_ensp]
        vecs = torch.stack([token_dict[k] for k in candidates])
        sim = torch.nn.functional.cosine_similarity(vecs, mean_seed, dim=1).detach().numpy()

        order = np.argsort(sim)[::-1]
        return [{"protein": self.ensp_to_hgnc[candidates[i]], "similarity": float(sim[i])}
                for i in order]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts_json", required=True,
                    help="Prompt list; seeds are parsed from 'Given these proteins ... that are known'.")
    ap.add_argument("--embed_path", required=True, help="semper_embed.pth")
    ap.add_argument("--label_path", required=True, help="semper_labels.txt")
    ap.add_argument("--ensp_to_gene", default=DEFAULT_ENSP_TO_GENE,
                    help="ENSP -> HGNC map (default: evaluate/data/ensp_to_gene.json)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--celltype", default=None,
                    help="Siletti cluster ID (e.g. cluster:395) for prompts that do not name "
                         "one. Overrides 'celltype cluster:N' in the prompt text.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.prompts_json) as f:
        prompts = json.load(f)
    retriever = SemperRetriever(args.embed_path, args.label_path, args.ensp_to_gene)

    for idx, entry in enumerate(prompts):
        if any(f.endswith(f"_{idx}.json") for f in os.listdir(args.output_dir)):
            continue
        content = entry.get("content", "")
        seeds = parse_seeds(content)
        cm = CLUSTER_PATTERN.search(content)
        celltype = normalize_celltype(args.celltype) if args.celltype else (cm.group(1) if cm else None)
        if not seeds or not celltype:
            print(f"Skipping prompt {idx}: no seeds or no cell type")
            continue

        ranked = retriever.retrieve(celltype, seeds)
        seed_str = "_".join(g.replace("/", "-") for g in seeds[:5])
        out = os.path.join(args.output_dir, f"tool_top_{celltype}_{seed_str}_{idx}.json")
        with open(out, "w") as f:
            json.dump(ranked, f, indent=2)
        print(f"[{idx}] {celltype}: {len(ranked)} candidates -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
