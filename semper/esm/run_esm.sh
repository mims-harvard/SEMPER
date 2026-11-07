#!/bin/bash
# Frozen ESM-C 600M inference (Methods, "The protein sequence encoder").
# Run inside the `semper-esm` conda environment (environment_esm.yml). Paths are relative to the repository root.
set -e
cd "$(dirname "$0")/../.."

FASTA="${FASTA:-data/protein_sequences.fasta}"
OUT_DIR="${OUT_DIR:-outputs/esm}"

mkdir -p "$OUT_DIR"
python semper/esm/generate_ESMC_embeds.py \
    --fasta      "$FASTA" \
    --out_embeds "$OUT_DIR/protein_esmc_600m.pt" \
    --out_index  "$OUT_DIR/protein_esmc_600m.pkl"
