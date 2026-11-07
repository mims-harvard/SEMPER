#!/bin/bash
# Paper-style figures on the demo data, made with the same scripts as the paper figures
# (see tutorial/README.md). Run after tutorial/run_tutorial.sh steps 1-5.
# The demo cell types are random subsets of one network neighbourhood, so these figures
# check that the analyses run; they are not expected to show the paper's biology.
set -e
cd "$(dirname "$0")/.."

OUT=outputs/tutorial
FIG=$OUT/figures
EMB=$OUT/semper/semper_embed.pth
LAB=$OUT/semper/semper_labels.txt
mkdir -p $FIG

echo "-- Fig. 2a analogue: within vs. across cell type similarity"
python evaluate/fig2_embedding_space/fig2a_pairwise_similarity.py \
    --embed-pt $EMB --labels $LAB \
    --supercluster-xlsx tutorial/demo_data/demo_superclusters.csv \
    --n-pairs 20000 \
    --out-prefix $FIG/fig2a_demo > $FIG/fig2a_demo.log

echo "-- Fig. 2b-d analogue: Gene Ontology term compactness vs. random null"
python evaluate/fig2_embedding_space/fig2bcd_go_dispersion.py \
    --ensp2go evaluate/data/ensp2go.json.gz \
    --go2ontology evaluate/data/go2ontology.json.gz \
    --embed-pt $EMB --labels $LAB \
    --out-prefix $FIG/fig2bcd_demo > $FIG/fig2bcd_demo.log

echo "-- Fig. 2e analogue: UMAP and Leiden modules of demo cell type 0"
python evaluate/fig2_embedding_space/fig2e_cluster_and_enrich.py \
    --embed-pt $EMB --labels $LAB --target-id 0 \
    --ensp2lowestgo evaluate/data/ensp2lowestgo.json.gz \
    --go2ontology evaluate/data/go2ontology.json.gz \
    --n-neighbors 15 \
    --out-prefix $FIG/fig2e_demo > $FIG/fig2e_demo.log

echo "-- Fig. 3b analogue: complex identification (co-expression definition)"
B=$FIG/fig3_demo
python evaluate/fig3_protein_complexes/prepare_complexes/prepare_coexpressed.py \
    --complexes data/benchmarks/protein_complexes/cns_complexes_547.csv \
    --members-col subunits_ensp --keep-cols complex_id,complex_name \
    --ppi-dir tutorial/demo_data/networks/ppi_edgelists/ \
    --avoid-pos-members --dedupe-globally \
    --out-dir $B > $FIG/fig3_demo.log
for MODEL in SEMPER Sequence_only; do
    if [ "$MODEL" = SEMPER ]; then E=$EMB; L=$LAB
    else E=$OUT/contrastive/esm_projected.pth; L=$OUT/contrastive/esm_projected_labels.txt; fi
    python evaluate/fig3_protein_complexes/evaluate_models/score_semper.py \
        --eval-csv $B/posneg.csv --model-name $MODEL --model-pt $E --model-labels $L \
        --out-scores-csv $B/${MODEL}_scores.csv --out-metrics-json $B/${MODEL}_metrics.json >> $FIG/fig3_demo.log
done
python evaluate/fig3_protein_complexes/evaluate_models/plot_auprc.py \
    --group-by cluster --min-n 50 \
    --model SEMPER=$B/SEMPER_scores.csv --model Sequence_only=$B/Sequence_only_scores.csv \
    --title "Demo cell types: co-expressed complexes" \
    --out-prefix $FIG/fig3_demo_auprc >> $FIG/fig3_demo.log

echo "-- Fig. 4 analogue: Parkinson's disease target retrieval per cell type"
python evaluate/fig4_target_prioritization/plot_results/plot_hits_at_k.py \
    --trials_csv $OUT/retrieval/PD_*__trials.csv \
    --global_models Sequence_only \
    --k_values 10 25 \
    --out_dir $FIG/fig4_demo > $FIG/fig4_demo.log

echo "Figures written to $FIG/"
