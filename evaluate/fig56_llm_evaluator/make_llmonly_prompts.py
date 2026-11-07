#!/usr/bin/env python3
"""Build the LLM-only prompts for Fig 5f-j / Fig 6 from the SEMPER prompts.

Each SEMPER prompt names its cell type only by an opaque Siletti cluster ID.
For the LLM-only arm the same seeds and the same cell types are kept, and the
cluster ID plus retrieval instruction are replaced by a readable description
of the cell type (name, class, supercluster, neurotransmitter identity, brain
regions, marker genes). Disease x cell type x seeds stay identical across arms.

    python make_llmonly_prompts.py \
        --semper_prompts inputs/prompts/generated_pd_messages_celltypecluster_zeroshotprompt_10prompt_t010.json \
        --disease pd \
        --out inputs/prompts/generated_pd_messages_llmonly_zeroshotprompt_10prompt_t010.json
"""

import argparse
import csv
import json
import os
import re

from common import resolve_disease

HERE = os.path.dirname(os.path.abspath(__file__))
N_TARGETS = 25   # ranked candidates requested per prompt

# glosses so the annotation codes read as biology
CLASS_GLOSS = {
    "NEUR": "neuronal",
    "OLIGO": "oligodendrocyte-lineage",
    "CHRP EPEN NEUR": "choroid-plexus / ependymal / neuronal",
    "": "unspecified",
}
NT_GLOSS = {
    "VGLUT1": "glutamatergic (excitatory)",
    "VGLUT2": "glutamatergic (excitatory)",
    "VGLUT3": "glutamatergic (excitatory)",
    "VGLUT1 VGLUT2": "glutamatergic (excitatory)",
    "VGLUT2 VGLUT3": "glutamatergic (excitatory)",
    "GABA": "GABAergic (inhibitory)",
    "GLY": "glycinergic (inhibitory)",
    "CHOL VGLUT3": "cholinergic / glutamatergic",
    "HDC VGLUT2": "histaminergic / glutamatergic",
    "": "unspecified",
}

gene_pattern = re.compile(r"Given these proteins (.+?) that are known")
cluster_pattern = re.compile(r"celltype (cluster:\w+)", re.IGNORECASE)


def gloss(mapping, raw):
    raw = (raw or "").strip()
    g = mapping.get(raw)
    if g is None:
        return raw if raw else "unspecified"
    return f"{raw} ({g})" if raw else g


def load_metadata(csv_path):
    meta = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            cid = f"cluster:{row['cluster_id'].strip()}"
            if cid in meta:
                continue
            meta[cid] = {
                "name": row["Cluster name"].strip(),
                "cls": gloss(CLASS_GLOSS, row["Class auto-annotation"]),
                "supercluster": row["Supercluster"].strip(),
                "nt": gloss(NT_GLOSS, row["Neurotransmitter auto-annotation"]),
                "regions": row["Top three regions"].strip(),
                "markers": row["Top Enriched Genes"].strip(),
            }
    return meta


def build_prompt(disease, seeds_str, m):
    return (
        f"Given these proteins {seeds_str} that are known targets for {disease} "
        f"in the following human brain cell type:\n"
        f"- Cell type: {m['name']}\n"
        f"- Cell class: {m['cls']}\n"
        f"- Supercluster: {m['supercluster']}\n"
        f"- Neurotransmitter identity: {m['nt']}\n"
        f"- Predominant brain regions: {m['regions']}\n"
        f"- Marker (top enriched) genes for this cell type: {m['markers']}\n\n"
        f"Taking this specific cellular and regional context into account, what are "
        f"{N_TARGETS} other proteins (excluding the seed proteins listed above) that are "
        f"also probable therapeutic targets for {disease} in this cell type? "
        f"Rank them from most to least likely."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--semper_prompts", required=True)
    ap.add_argument("--disease", required=True,
                    help="pd, ad, ms, als or hd (uses the paper's disease names), or a free-text name.")
    ap.add_argument("--celltype_meta", default=os.path.join(HERE, "inputs", "celltype_metadata_top10.csv"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    disease = resolve_disease(args.disease)
    meta = load_metadata(args.celltype_meta)
    with open(args.semper_prompts) as f:
        semper_prompts = json.load(f)

    out, missing = [], set()
    for entry in semper_prompts:
        content = entry.get("content", "")
        gm = gene_pattern.search(content)
        cm = cluster_pattern.search(content)
        if not (gm and cm):
            continue
        if cm.group(1) not in meta:
            missing.add(cm.group(1))
            continue
        out.append({"role": "user", "content": build_prompt(disease, gm.group(1).strip(), meta[cm.group(1)])})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"{len(out)} LLM-only prompts -> {args.out}")
    if missing:
        print(f"WARNING: no cell type metadata for {sorted(missing)}")


if __name__ == "__main__":
    main()
