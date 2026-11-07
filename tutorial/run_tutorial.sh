#!/bin/bash
# End-to-end SEMPER tutorial on the demo dataset (see tutorial/README.md).
# Requires the `semper` and `semper-esm` conda environments and one NVIDIA GPU.
# Run from the `semper` environment; step 1 switches to `semper-esm` itself.
set -e
cd "$(dirname "$0")/.."

export WANDB_MODE="${WANDB_MODE:-disabled}"
OUT=outputs/tutorial
DEMO=tutorial/demo_data

echo "== 1/6  ESM-C 600M sequence embeddings"
if [ -n "$SKIP_ESM" ]; then  # embeddings already in $OUT/esm
    echo "skipped (SKIP_ESM is set)"
else
    ESM_ENV="${ESM_ENV:-semper-esm}"  # environment name, or a path to an environment prefix
    if [[ "$ESM_ENV" == */* ]]; then ENV_FLAG=-p; else ENV_FLAG=-n; fi
    FASTA=$DEMO/protein_sequences.fasta OUT_DIR=$OUT/esm \
        conda run --no-capture-output $ENV_FLAG "$ESM_ENV" bash semper/esm/run_esm.sh
fi

echo "== 2/6  PPI graph transformer pretraining (1 GPU)"
NUM_GPUS=1 EPOCHS="${PPI_EPOCHS:-20}" DATA_DIR=$DEMO SAVE_PREFIX=$OUT/ppi_gt/ \
    bash semper/ppi_gt/run_ppi_gt.sh

echo "== 3/6  Contrastive alignment and SEMPER representations"
EMBED_PATH=$OUT/ppi_gt/_protein_embed.pth LABEL_PATH=$OUT/ppi_gt/_labels_dict.txt \
ESM_PICKLE=$OUT/esm/protein_esmc_600m.pkl ESM_EMBEDS=$OUT/esm/protein_esmc_600m.pt \
SAVE_DIR=$OUT/contrastive/ SEMPER_DIR=$OUT/semper/ EPOCHS="${CLIP_EPOCHS:-50}" \
    bash semper/contrastive/run_contrastive.sh

echo "== 4/6  Seed plan from the Parkinson's disease targets used in the paper"
python evaluate/fig4_target_prioritization/make_seedplan.py \
    --disease_json data/benchmarks/clinical_targets/analysis_target_lists/PD_targets_MONDO_0005180.json \
    --ppi_dir      $DEMO/networks/ppi_edgelists/ \
    --ensp_json    evaluate/data/ensp_to_gene.json \
    --k_seed 15 --n_trials 20 \
    --out_prefix   $OUT/seedplans/PD

echo "== 5/6  Zero-shot target retrieval (Hits@k)"
mkdir -p $OUT/retrieval
for MODEL in SEMPER Sequence_only; do
    if [ "$MODEL" = SEMPER ]; then
        EMB=$OUT/semper/semper_embed.pth;          LAB=$OUT/semper/semper_labels.txt
    else  # sequence projection alone (t = 0), for comparison
        EMB=$OUT/contrastive/esm_projected.pth;    LAB=$OUT/contrastive/esm_projected_labels.txt
    fi
    python evaluate/fig4_target_prioritization/run_evaluations/neuro_eval_semper_ct_network.py \
        --seedplan     $OUT/seedplans/PD_seedplan.json \
        --pd_gene_file $OUT/seedplans/PD_targets.json \
        --ensp_json    evaluate/data/ensp_to_gene.json \
        --embed_path   $EMB \
        --label_dict   $LAB \
        --model_name   $MODEL \
        --device       cpu \
        --k_values 5 10 25 \
        --out_prefix   $OUT/retrieval/PD_$MODEL > $OUT/retrieval/PD_$MODEL.log
done

python - <<'EOF'
import pandas as pd
df = pd.concat(pd.read_csv(f"outputs/tutorial/retrieval/PD_{m}__trials.csv") for m in ["SEMPER", "Sequence_only"])
print("\nMean Hits@k over demo cell types and trials (demo data, not comparable to the paper):")
print(df.pivot_table(index="model", columns="k", values="hits_at_k", aggfunc="mean").round(2).to_string())
EOF
echo "== 6/6  Paper-style figures on the demo data"
bash tutorial/run_figures.sh
echo "Done. Outputs are in $OUT/ and figures in $OUT/figures/"
