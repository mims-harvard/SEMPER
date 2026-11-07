# Figure 4 — Zero-shot retrieval of candidate therapeutic targets

This directory reproduces Figure 4: SEMPER ranks all proteins in each CNS cell
type by cosine similarity to a seed set of clinical trial candidate genes, and
we measure how many held-out candidates appear in the top-K (Hits@K) across five
neurodegenerative diseases.

The pipeline has two parts:

1. **`run_evaluations/`** — run each model against each disease to produce
   `*__trials.csv` files (one per model × disease combination)

2. **`plot_results/`** — aggregate trials CSVs and plot Hits@K distributions,
   plus statistical significance testing

---

## Data requirements

```
data/
├── semper_embed.pth / semper_labels.txt         # SEMPER (semper/contrastive/build_semper_embeddings.py)
├── ppi_gt_embed.pth / ppi_gt_labels.txt           # PPI-GT
├── pinnacle_embed.pth / pinnacle_labels.txt        # PINNACLE
├── protein_esmc_600m.pkl / protein_esmc_600m.pt   # ESM-C
├── ensp_to_gene.json                              # ENSP → gene symbol
├── ensp_id2vec_mean.pkl.gz                        # ProTrek Seq embeddings
├── ensp_id2vec_structure_mean.pkl.gz              # ProTrek Struct embeddings
├── transposenet_orbits.txt                        # ORCA global orbits
├── transposenet_nodes.json                        # ORCA node map
└── ppi_graphlets_orca/                            # ORCA-CT per-cell-type orbits
    └── <N>_edgelist_nodes.json / <N>_edgelist_orbits.txt
```

### Seed plans and disease target lists

The target lists used in the paper are in
[`data/benchmarks/clinical_targets/analysis_target_lists/`](../../data/benchmarks/clinical_targets/analysis_target_lists),
one JSON per disease (e.g. `PD_targets_MONDO_0005180.json`). Pass these as
`--pd_gene_file` below to score against the same targets as the paper.

A seed plan fixes the seed and held-out targets of every trial so all models are
scored on identical trials. The seed plans used in the paper are in [`seedplans/`](seedplans)
(`<D>_seedplan.json.gz` for PD, AD, MS, HD and ALS; the evaluation scripts read the
gzipped files directly). Use them, with the target lists above, to reproduce Fig. 4 and
Supplementary Table 7. In these plans, the candidates in each cell type are the cell type's network
proteins that are also in ProTrek's protein set (`make_seedplan.py` below does not apply the
ProTrek filter).

To build seed plans for your own networks with `make_seedplan.py` (paper settings: 15 seeds,
100 trials):

```bash
python make_seedplan.py \
    --disease_json ../../data/benchmarks/clinical_targets/analysis_target_lists/PD_targets_MONDO_0005180.json \
    --ppi_dir      ../../data/networks/ppi_edgelists/ \
    --ensp_json    ../data/ensp_to_gene.json \
    --k_seed 15 --n_trials 100 \
    --out_prefix   my_seedplans/PD
```

This writes `my_seedplans/PD_seedplan.json` and `my_seedplans/PD_targets.json` (the
`--pd_gene_file` below). Repeat for AD, MS, HD and ALS.

---

## Part 1 — Run evaluations

Run each script once per disease × model combination. Each run produces a
`*__trials.csv` file in the specified output directory.

### SEMPER, PPI-GT, PINNACLE (cell type-aware embedding models)

```bash
# SEMPER — run once per disease
python run_evaluations/neuro_eval_semper_ct_network.py \
    --seedplan      seedplans/PD_seedplan.json.gz \
    --pd_gene_file  ../../data/benchmarks/clinical_targets/analysis_target_lists/PD_targets_MONDO_0005180.json \
    --ensp_json     data/ensp_to_gene.json \
    --embed_path    data/semper_embed.pth \
    --label_dict    data/semper_labels.txt \
    --model_name    SEMPER_ESMCLIP \
    --out_prefix    trials/neuro_eval_SEMPER_ESMCLIP__ct_network_PD \
    --device        cuda \
    --k_values 5 10 25

# PPI-GT — same script, swap embed_path, label_dict, model_name
python run_evaluations/neuro_eval_semper_ct_network.py \
    --seedplan      seedplans/PD_seedplan.json.gz \
    --pd_gene_file  ../../data/benchmarks/clinical_targets/analysis_target_lists/PD_targets_MONDO_0005180.json \
    --ensp_json     data/ensp_to_gene.json \
    --embed_path    data/ppi_gt_embed.pth \
    --label_dict    data/ppi_gt_labels.txt \
    --model_name    PPI_GT_Encoder \
    --out_prefix    trials/neuro_eval_PPI_GT__ct_network_PD \
    --device        cuda \
    --k_values 5 10 25

# PINNACLE — same script, swap embed_path, label_dict, model_name
python run_evaluations/neuro_eval_semper_ct_network.py \
    --seedplan      seedplans/PD_seedplan.json.gz \
    --pd_gene_file  ../../data/benchmarks/clinical_targets/analysis_target_lists/PD_targets_MONDO_0005180.json \
    --ensp_json     data/ensp_to_gene.json \
    --embed_path    data/pinnacle_embed.pth \
    --label_dict    data/pinnacle_labels.txt \
    --model_name    GNN_PINNACLE \
    --out_prefix    trials/neuro_eval_PINNACLE__ct_network_PD \
    --device        cuda \
    --k_values 5 10 25

# Repeat for AD, MS, HD, ALS by swapping --seedplan, --pd_gene_file, --out_prefix
```

### ESM-C (global embedding model)

```bash
python run_evaluations/neuro_eval_esm_pklpt_ct_network.py \
    --seedplan       seedplans/PD_seedplan.json.gz \
    --pd_gene_file   ../../data/benchmarks/clinical_targets/analysis_target_lists/PD_targets_MONDO_0005180.json \
    --ensp_json      data/ensp_to_gene.json \
    --label_dict     data/pinnacle_labels.txt \
    --esm_pickle     data/protein_esmc_600m.pkl \
    --esm_embeddings data/protein_esmc_600m.pt \
    --model_name     ESMC_600M \
    --out_prefix     trials/neuro_eval_ESMC600M__ct_network_PD \
    --device         cuda \
    --k_values 5 10 25 \
    --skip_if_missing_seed

# Repeat for AD, MS, HD, ALS
```

### ORCA (global graphlets)

```bash
python run_evaluations/neuro_eval_graphlets_global_ct_network.py \
    --seedplan      seedplans/PD_seedplan.json.gz \
    --pd_gene_file  ../../data/benchmarks/clinical_targets/analysis_target_lists/PD_targets_MONDO_0005180.json \
    --ensp_json     data/ensp_to_gene.json \
    --label_dict    data/pinnacle_labels.txt \
    --node-map      data/transposenet_nodes.json \
    --orbits        data/transposenet_orbits.txt \
    --model_name    Graphlets_ORCA \
    --out_prefix    trials/neuro_eval_ORCA__ct_network_PD \
    --device        cpu \
    --k_values 5 10 25 \
    --skip_if_missing_seed

# Repeat for AD, MS, HD, ALS
```

### ORCA-CT (cell type-specific graphlets)

```bash
python run_evaluations/neuro_eval_graphlets_ct_orbits_ct_network.py \
    --seedplan      seedplans/PD_seedplan.json.gz \
    --pd_gene_file  ../../data/benchmarks/clinical_targets/analysis_target_lists/PD_targets_MONDO_0005180.json \
    --ensp_json     data/ensp_to_gene.json \
    --label_dict    data/pinnacle_labels.txt \
    --ct-orbits-dir data/ppi_graphlets_orca/ \
    --ct-suffix     _edgelist \
    --model_name    Graphlets_ORCA_CT \
    --out_prefix    trials/neuro_eval_ORCA_CT__ct_network_PD \
    --device        cpu \
    --k_values 5 10 25 \
    --skip_if_missing_seed

# Repeat for AD, MS, HD, ALS
```

### ProTrek Seq and ProTrek Struct (global embedding models)

ProTrek scores against its own global protein universe rather than
the per-cell-type SEMPER candidate space. Run once per model × disease:

```bash
# ProTrek Seq
python run_evaluations/neuro_eval_protrek.py \
    --seedplan      seedplans/PD_seedplan.json.gz \
    --ensp_json     data/ensp_to_gene.json \
    --protrek_gz    data/ensp_id2vec_mean.pkl.gz \
    --model_name    ProTrek_seq_mean \
    --out_prefix    trials/neuro_eval_ProTrek_seq__PD \
    --device        cuda \
    --k_values 5 10 25 \
    --skip_if_missing_seed

# ProTrek Struct — swap --protrek_gz and --model_name
python run_evaluations/neuro_eval_protrek.py \
    --seedplan      seedplans/PD_seedplan.json.gz \
    --ensp_json     data/ensp_to_gene.json \
    --protrek_gz    data/ensp_id2vec_structure_mean.pkl.gz \
    --model_name    ProTrek_struct_mean \
    --out_prefix    trials/neuro_eval_ProTrek_struct__PD \
    --device        cuda \
    --k_values 5 10 25 \
    --skip_if_missing_seed

# Repeat for AD, MS, HD, ALS
```

---

## Part 2 — Plot results

### Plot Hits@K across models (Figures 4c–g)

Run once per disease, pointing to all trials CSVs for that disease:

```bash
# Parkinson's disease (panel c)
python plot_results/plot_hits_at_k.py \
    --trials_csv trials/*PD*__trials.csv \
    --out_dir    figures/PD/ \
    --k_values   5 10 25

# Repeat for AD, MS, HD, ALS
```

### Statistical significance vs. baselines (Supplementary Table 7)

Run once per disease, so that Hits@k is averaged over the trials of one disease in
each cell type. The test compares SEMPER with the cell type-aware baselines
(paired, one-sided Wilcoxon signed-rank test across cell types):

```bash
for D in PD AD MS HD ALS; do
  python plot_results/calc_hits_significance.py \
      --trials_csv    "trials/*${D}*__trials.csv" \
      --semper_model  SEMPER \
      --baselines     PINNACLE PPI-GT ORCA-CT \
      --k_values      10
done
```
