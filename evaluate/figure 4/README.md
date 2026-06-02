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
├── esm_projected.pth / esm_projected_labels.txt   # SEMPER (Figshare)
├── ppi_gt_embed.pth / ppi_gt_labels.txt           # PPI-GT (Figshare)
├── pinnacle_embed.pth / pinnacle_labels.txt        # PINNACLE (Figshare)
├── protein_esmc_600m.pkl / protein_esmc_600m.pt   # ESM-C (Figshare)
├── ensp_to_gene.json                              # ENSP → gene symbol (Figshare)
├── transposenet_orbits.txt                        # ORCA global orbits (Figshare)
├── transposenet_nodes.json                        # ORCA node map (Figshare)
└── ppi_graphlets_orca/                            # ORCA-CT per-cell-type orbits (Figshare)
    └── <N>_edgelist_nodes.json / <N>_edgelist_orbits.txt
```

### Seedplans and disease gene lists

The seedplan JSONs (which control seed/held-out trial splits per cell type) and
the disease gene lists (clinical trial candidate targets for each disease) are
required to run the evaluations but are not included in this repository.

To reproduce the paper results, please contact the authors to request access to:
- `neuro_seedplan_<DISEASE>.json` — one per disease (PD, AZ, MS, HD, ALS)
- `raw_targets_<DISEASE>.json` — clinical trial candidate gene lists per disease

---

## Part 1 — Run evaluations

Run each script once per disease × model combination. Each run produces a
`*__trials.csv` file in the specified output directory.

### SEMPER, PPI-GT, PINNACLE (cell type-aware embedding models)

```bash
# SEMPER — run once per disease
python run_evaluations/neuro_eval_semper_ct_network.py \
    --seedplan      data/neuro_seedplan_PD.json \
    --pd_gene_file  data/raw_targets_PD.json \
    --ensp_json     data/ensp_to_gene.json \
    --embed_path    data/esm_projected.pth \
    --label_dict    data/esm_projected_labels.txt \
    --model_name    SEMPER_ESMCLIP \
    --out_prefix    trials/neuro_eval_SEMPER_ESMCLIP__ct_network_PD \
    --device        cuda \
    --k_values 5 10 25

# PPI-GT — same script, swap embed_path, label_dict, model_name
python run_evaluations/neuro_eval_semper_ct_network.py \
    --seedplan      data/neuro_seedplan_PD.json \
    --pd_gene_file  data/raw_targets_PD.json \
    --ensp_json     data/ensp_to_gene.json \
    --embed_path    data/ppi_gt_embed.pth \
    --label_dict    data/ppi_gt_labels.txt \
    --model_name    PPI_GT_Encoder \
    --out_prefix    trials/neuro_eval_PPI_GT__ct_network_PD \
    --device        cuda \
    --k_values 5 10 25

# PINNACLE — same script, swap embed_path, label_dict, model_name
python run_evaluations/neuro_eval_semper_ct_network.py \
    --seedplan      data/neuro_seedplan_PD.json \
    --pd_gene_file  data/raw_targets_PD.json \
    --ensp_json     data/ensp_to_gene.json \
    --embed_path    data/pinnacle_embed.pth \
    --label_dict    data/pinnacle_labels.txt \
    --model_name    GNN_PINNACLE \
    --out_prefix    trials/neuro_eval_PINNACLE__ct_network_PD \
    --device        cuda \
    --k_values 5 10 25

# Repeat for AZ, MS, HD, ALS by swapping --seedplan, --pd_gene_file, --out_prefix
```

### ESM-C (global embedding model)

```bash
python run_evaluations/neuro_eval_esm_pklpt_ct_network.py \
    --seedplan       data/neuro_seedplan_PD.json \
    --pd_gene_file   data/raw_targets_PD.json \
    --ensp_json      data/ensp_to_gene.json \
    --label_dict     data/pinnacle_labels.txt \
    --esm_pickle     data/protein_esmc_600m.pkl \
    --esm_embeddings data/protein_esmc_600m.pt \
    --model_name     ESMC_600M \
    --out_prefix     trials/neuro_eval_ESMC600M__ct_network_PD \
    --device         cuda \
    --k_values 5 10 25 \
    --skip_if_missing_seed

# Repeat for AZ, MS, HD, ALS
```

### ORCA (global graphlets)

```bash
python run_evaluations/neuro_eval_graphlets_global_ct_network.py \
    --seedplan      data/neuro_seedplan_PD.json \
    --pd_gene_file  data/raw_targets_PD.json \
    --ensp_json     data/ensp_to_gene.json \
    --label_dict    data/pinnacle_labels.txt \
    --node-map      data/transposenet_nodes.json \
    --orbits        data/transposenet_orbits.txt \
    --model_name    Graphlets_ORCA \
    --out_prefix    trials/neuro_eval_ORCA__ct_network_PD \
    --device        cpu \
    --k_values 5 10 25 \
    --skip_if_missing_seed

# Repeat for AZ, MS, HD, ALS
```

### ORCA-CT (cell type-specific graphlets)

```bash
python run_evaluations/neuro_eval_graphlets_ct_orbits_ct_network.py \
    --seedplan      data/neuro_seedplan_PD.json \
    --pd_gene_file  data/raw_targets_PD.json \
    --ensp_json     data/ensp_to_gene.json \
    --label_dict    data/pinnacle_labels.txt \
    --ct-orbits-dir data/ppi_graphlets_orca/ \
    --ct-suffix     _edgelist \
    --model_name    Graphlets_ORCA_CT \
    --out_prefix    trials/neuro_eval_ORCA_CT__ct_network_PD \
    --device        cpu \
    --k_values 5 10 25 \
    --skip_if_missing_seed

# Repeat for AZ, MS, HD, ALS
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

# Repeat for AZ, MS, HD, ALS
```

### Statistical significance vs. baselines (Figure 4 statistics)

```bash
python plot_results/calc_hits_significance.py \
    --trials_csv    "trials/*__trials.csv" \
    --semper_model  SEMPER \
    --baselines     PINNACLE PPI-GT ORCA-CT ESM-C ORCA \
    --k_values      5 10 25
```

---

## Summary

| Script | What it does |
|---|---|
| `run_evaluations/neuro_eval_semper_ct_network.py` | Hits@K for SEMPER, PPI-GT, PINNACLE |
| `run_evaluations/neuro_eval_esm_pklpt_ct_network.py` | Hits@K for ESM-C |
| `run_evaluations/neuro_eval_graphlets_global_ct_network.py` | Hits@K for ORCA (global graphlets) |
| `run_evaluations/neuro_eval_graphlets_ct_orbits_ct_network.py` | Hits@K for ORCA-CT |
| `plot_results/plot_hits_at_k.py` | Per-disease Hits@K strip plots (Figs 4c–g) |
| `plot_results/calc_hits_significance.py` | Paired Wilcoxon significance vs. baselines 
