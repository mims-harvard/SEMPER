#!/bin/bash

G_F="data/human_background.csv"
PPI_DIR="data/networks/ppi_edgelists_all_threshold4000/"
MG_F="data/networks/mg_edgelist_brainregion_all_threshold4000.txt"

NUM_GPUS=4
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29400}"

FEAT_MAT=512
BATCH_SIZE=32
EPOCHS=300

torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
        --G_f $G_F \
        --ppi_dir $PPI_DIR \
        --mg_f $MG_F \
        --feat_mat=$FEAT_MAT \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS
