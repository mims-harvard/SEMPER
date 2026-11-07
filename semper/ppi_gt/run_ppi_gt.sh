#!/bin/bash
# Pretrain the PPI graph transformer (Methods, "Pretraining the multi-scale network encoder").
# Paths are relative to the repository root and can be overridden, e.g.
#   NUM_GPUS=1 EPOCHS=5 DATA_DIR=tutorial/demo_data bash semper/ppi_gt/run_ppi_gt.sh
set -e
cd "$(dirname "$0")/../.."

DATA_DIR="${DATA_DIR:-data}"
G_F="${G_F:-$DATA_DIR/human_background.csv}"
PPI_DIR="${PPI_DIR:-$DATA_DIR/networks/ppi_edgelists/}"
MG_F="${MG_F:-$DATA_DIR/networks/mg_edgelist.txt}"
SAVE_PREFIX="${SAVE_PREFIX:-outputs/ppi_gt/}"

NUM_GPUS="${NUM_GPUS:-4}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29400}"

FEAT_MAT=512
BATCH_SIZE=32
EPOCHS="${EPOCHS:-300}"

mkdir -p "$SAVE_PREFIX"
cd semper/ppi_gt
torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
        --G_f "../../$G_F" \
        --ppi_dir "../../$PPI_DIR" \
        --mg_f "../../$MG_F" \
        --feat_mat=$FEAT_MAT \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --save_prefix "../../$SAVE_PREFIX"
