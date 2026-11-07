#!/usr/bin/env python3
"""
Build the small, shareable demo dataset in tutorial/demo_data/.

The real SEMPER inputs (442 cell type networks from the adult human brain atlas)
are not redistributed. The demo instead builds four synthetic "cell type" networks
from the public STRING interactome (v12.0, human physical interactions, CC BY 4.0),
so every step of the pipeline can be run end to end in minutes. The demo networks
are not biological cell types, and results on them are not comparable to the paper.

Inputs (from https://string-db.org/cgi/download?species_text=Homo+sapiens)
  9606.protein.physical.links.v12.0.txt.gz   (--background)
  9606.protein.sequences.v12.0.fa.gz         (--fasta)

Construction
  1. Keep STRING edges with combined score >= --min_weight (0.4 is STRING's
     medium confidence).
  2. Start from the Parkinson's disease targets present in the network and grow a
     protein pool by breadth-first search to --pool_size proteins.
  3. Each demo cell type keeps a random --keep_frac of the pool; its network is the
     largest connected component of the induced subgraph.
  4. The metagraph links the cell types to two mock brain regions under a root.
"""

import argparse
import gzip
import json
import os
import random
from collections import deque

import networkx as nx
import pandas as pd
from Bio import SeqIO


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="Build the SEMPER tutorial demo dataset.")
    ap.add_argument("--background", required=True,
                    help="STRING links file (protein1 protein2 combined_score, space separated, may be gzipped) "
                         "or a CSV with columns protein1, protein2, weight in [0, 1].")
    ap.add_argument("--fasta", required=True, help="FASTA with sequences for the background proteins (ENSP IDs, with or without the 9606. prefix; may be gzipped).")
    ap.add_argument("--targets_csv", default=os.path.join(repo, "data/benchmarks/clinical_targets/PD_clinical_targets.csv"))
    ap.add_argument("--ensp_json", default=os.path.join(repo, "evaluate/data/ensp_to_gene.json"))
    ap.add_argument("--out_dir", default=os.path.join(here, "demo_data"))
    ap.add_argument("--min_weight", type=float, default=0.4)
    ap.add_argument("--pool_size", type=int, default=500)
    ap.add_argument("--n_celltypes", type=int, default=4)
    ap.add_argument("--keep_frac", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    if args.background.endswith((".txt", ".txt.gz")):  # STRING links file
        bg = pd.read_csv(args.background, sep=" ")
        bg["weight"] = bg.pop("combined_score") / 1000
    else:
        bg = pd.read_csv(args.background)
    bg = bg[bg["weight"] >= args.min_weight][["protein1", "protein2", "weight"]]
    G = nx.from_pandas_edgelist(bg, "protein1", "protein2")
    G.remove_edges_from(nx.selfloop_edges(G))

    with open(args.ensp_json) as f:
        ensp_to_gene = {k.replace("9606.", ""): v for k, v in json.load(f).items()}
    targets = set(pd.read_csv(args.targets_csv)["targetId_genename"].str.upper())
    target_nodes = sorted(n for n in G if (ensp_to_gene.get(n.replace("9606.", "")) or "").upper() in targets)
    if not target_nodes:
        raise SystemExit("No targets found in the background network; check --ensp_json and --targets_csv.")

    # Breadth-first growth from the targets, visiting neighbors in random order
    pool, queue = set(target_nodes), deque(target_nodes)
    while queue and len(pool) < args.pool_size:
        nbrs = list(G.neighbors(queue.popleft()))
        rng.shuffle(nbrs)
        for n in nbrs:
            if n not in pool:
                pool.add(n)
                queue.append(n)
                if len(pool) >= args.pool_size:
                    break
    pool = sorted(pool)

    ppi_dir = os.path.join(args.out_dir, "networks", "ppi_edgelists")
    os.makedirs(ppi_dir, exist_ok=True)
    used = set()
    for c in range(args.n_celltypes):
        keep = rng.sample(pool, int(args.keep_frac * len(pool)))
        sub = G.subgraph(keep)
        lcc = sub.subgraph(max(nx.connected_components(sub), key=len))
        nx.write_edgelist(lcc, os.path.join(ppi_dir, f"{c}_edgelist.txt"), data=False)
        used |= set(lcc.nodes)
        n_t = sum(n in set(target_nodes) for n in lcc)
        print(f"cell type {c}: {lcc.number_of_nodes()} proteins, {lcc.number_of_edges()} edges, {n_t} PD targets")

    # Metagraph: root -> two mock regions -> cell types (tab separated, as in the real data)
    regions = ["BTO:DemoRegionA", "BTO:DemoRegionB"]
    with open(os.path.join(args.out_dir, "networks", "mg_edgelist.txt"), "w") as f:
        for r in regions:
            f.write(f"BTO:DemoBrain\t{r}\n")
        for c in range(args.n_celltypes):
            f.write(f"{regions[c % len(regions)]}\tcluster:{c}\n")

    # Background restricted to proteins used in the demo networks
    bg_demo = bg[bg["protein1"].isin(used) & bg["protein2"].isin(used)]
    bg_demo.to_csv(os.path.join(args.out_dir, "human_background.csv"), index=False)

    # Sequences for the demo proteins
    wanted = {n.replace("9606.", "") for n in used}
    opener = gzip.open if args.fasta.endswith(".gz") else open
    with opener(args.fasta, "rt") as f:
        records = []
        for r in SeqIO.parse(f, "fasta"):
            r.id = r.id.replace("9606.", "")
            r.description = ""
            if r.id in wanted:
                records.append(r)
    SeqIO.write(records, os.path.join(args.out_dir, "protein_sequences.fasta"), "fasta")
    print(f"{len(used)} proteins, {len(bg_demo)} background edges, {len(records)} sequences -> {args.out_dir}")


if __name__ == "__main__":
    main()
