#!/usr/bin/env python3
"""
Build the final cell type-resolved SEMPER representations.

After contrastive alignment, each protein p has one sequence projection u_p
(identical in every cell type) and one network projection v_{p,c} per cell type
c whose network contains it. Both live in the same aligned 512-d space. SEMPER
combines them with a fixed weight t (Methods, "Cell type-resolved
representations"):

    z_{p,c} = normalize( (1 - t) * u_p + t * v_{p,c} ),   t = 0.10

No training is involved. The output is a drop-in per-cluster embedding file
({ct_idx -> tensor[n_proteins, 512]}) used by every analysis in evaluate/.
"""

import argparse
import os
import shutil

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine aligned sequence and network projections into SEMPER representations."
    )
    parser.add_argument("--contrastive_dir", required=True,
                        help="Output directory of train_contrastive_MLP.py "
                             "(must contain esm_projected.pth, ppi_projected.pth, esm_projected_labels.txt).")
    parser.add_argument("--out_dir", required=True,
                        help="Where to write semper_embed.pth and semper_labels.txt.")
    parser.add_argument("--t", type=float, default=0.10,
                        help="Weight on the cell type-specific network projection (default: 0.10).")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.t <= 1.0:
        raise ValueError(f"--t must be in [0, 1], got {args.t}")
    os.makedirs(args.out_dir, exist_ok=True)

    esm = torch.load(os.path.join(args.contrastive_dir, "esm_projected.pth"), map_location="cpu")
    ppi = torch.load(os.path.join(args.contrastive_dir, "ppi_projected.pth"), map_location="cpu")
    if set(esm) != set(ppi):
        raise RuntimeError("Cell type keys differ between esm_projected.pth and ppi_projected.pth.")

    z = {}
    for k in esm:
        if esm[k].shape != ppi[k].shape:
            raise RuntimeError(f"Shape mismatch at cell type {k}: {tuple(esm[k].shape)} vs {tuple(ppi[k].shape)}")
        u = F.normalize(esm[k].float(), dim=1)  # unit rows, so t is a clean mixing weight
        v = F.normalize(ppi[k].float(), dim=1)
        z[k] = F.normalize((1.0 - args.t) * u + args.t * v, dim=1)

    torch.save(z, os.path.join(args.out_dir, "semper_embed.pth"))
    shutil.copy(os.path.join(args.contrastive_dir, "esm_projected_labels.txt"),
                os.path.join(args.out_dir, "semper_labels.txt"))

    any_ct = next(iter(z))
    print(f"Wrote SEMPER representations for {len(z)} cell types (t = {args.t}); "
          f"example cell type {any_ct}: {tuple(z[any_ct].shape)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
