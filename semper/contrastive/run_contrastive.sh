#!/bin/bash
set -e

# Outputs from the PPI-GT encoder (train.py --save_prefix)
EMBED_PATH="outputs/ppi_gt/_protein_embed.pth"
LABEL_PATH="outputs/ppi_gt/_labels_dict.txt"

# Outputs from generate_ESMC_embeds.py
ESM_PICKLE="outputs/esm/protein_esmc_600m.pkl"
ESM_EMBEDS="outputs/esm/protein_esmc_600m.pt"

# Where contrastive outputs will be saved
SAVE_DIR="outputs/contrastive/"

PROJ_DIM=512
HIDDEN_ESM="1024 512"
HIDDEN_PPI="512 512"
BATCH_SIZE=64
EPOCHS=200
LR=1e-4
TEMPERATURE=0.07
SEED=42

python train_contrastive_MLP.py \
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
