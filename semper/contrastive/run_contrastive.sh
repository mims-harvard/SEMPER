#!/bin/bash
# Contrastive alignment of ESM-C and PPI-GT embeddings, then the final SEMPER
# representations (Methods, "Multimodal contrastive alignment").
# Paths are relative to the repository root.
set -e
cd "$(dirname "$0")/../.."

# Outputs from the PPI-GT encoder (run_ppi_gt.sh)
EMBED_PATH="${EMBED_PATH:-outputs/ppi_gt/_protein_embed.pth}"
LABEL_PATH="${LABEL_PATH:-outputs/ppi_gt/_labels_dict.txt}"

# Outputs from generate_ESMC_embeds.py (run_esm.sh)
ESM_PICKLE="${ESM_PICKLE:-outputs/esm/protein_esmc_600m.pkl}"
ESM_EMBEDS="${ESM_EMBEDS:-outputs/esm/protein_esmc_600m.pt}"

SAVE_DIR="${SAVE_DIR:-outputs/contrastive/}"
SEMPER_DIR="${SEMPER_DIR:-outputs/semper/}"

PROJ_DIM=512
HIDDEN_ESM="1024 512"
HIDDEN_PPI="512 512"
BATCH_SIZE=64
EPOCHS="${EPOCHS:-200}"
LR=1e-4
TEMPERATURE=0.07
SEED=42
T=0.10   # weight on the cell type-specific network projection

python semper/contrastive/train_contrastive_MLP.py \
    --embed_path  $EMBED_PATH \
    --label_path  $LABEL_PATH \
    --esm_pickle  $ESM_PICKLE \
    --esm_embeds  $ESM_EMBEDS \
    --save_dir    $SAVE_DIR \
    --proj_dim    $PROJ_DIM \
    --hidden_esm  $HIDDEN_ESM \
    --hidden_ppi  $HIDDEN_PPI \
    --batch_size  $BATCH_SIZE \
    --epochs      $EPOCHS \
    --lr          $LR \
    --temperature $TEMPERATURE \
    --seed        $SEED

python semper/contrastive/build_semper_embeddings.py \
    --contrastive_dir $SAVE_DIR \
    --out_dir         $SEMPER_DIR \
    --t               $T
