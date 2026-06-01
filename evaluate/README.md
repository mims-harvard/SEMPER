# Evaluating SEMPER

All scripts operate on `esm_projected.pth` and its companion
label dictionary.

**Required files** (download from [Figshare](#)):
- `esm_projected.pth` — SEMPER embeddings `{cluster_idx → tensor [n_proteins, 512]}`
- `esm_projected_labels.txt` — label dictionary
- `ensp_to_gene.json` — ENSP → HGNC gene symbol map
- `ensp2lowestgo.json.gz` — ENSP → lowest-level GO term annotations
- `go2ontology.json.gz` — GO term → ontology namespace (BP/MF/CC)

---

## 1. Cell type cluster analysis: UMAP + Leiden + GO enrichment (Figure 2e)

Single unified script. Loads one specific cluster, runs Leiden community detection,
produces a Nature-style UMAP colored by module, and tests GO term enrichment per module.

```bash
python cluster_and_enrich.py \
    --embed-pt      outputs/contrastive/esm_projected.pth \
    --labels        outputs/contrastive/esm_projected_labels.txt \
    --target-id     395 \
    --ensp2lowestgo data/ensp2lowestgo.json.gz \
    --go2ontology   data/go2ontology.json.gz \
    --resolution    1.0 \
    --n-neighbors   20 \
    --ontology      BP \
    --out-prefix    outputs/evaluate/DANeuron_c395
```

| Argument | Description | Default |
|---|---|---|
| `--target-id` | Numerical cluster ID (e.g. `395` for dopaminergic neurons) | required |
| `--resolution` | Leiden resolution — higher yields more, smaller modules | `1.0` |
| `--ontology` | GO namespace: `BP`, `MF`, `CC`, or `ALL` | `BP` |

**Outputs:** `<out-prefix>_umap.pdf`, `<out-prefix>_umap.png`, `<out-prefix>_cluster_assignments.csv`, `<out-prefix>_enrichment.csv`

---

## 2. Target prioritization — compute Hits@K (Figure 4, Step 1)

Runs the Hits@K evaluation for SEMPER (or any embedding-based model) against a
disease gene list. For each cell type in the embedding, seeds are sampled from the
disease gene list, all proteins are ranked by cosine similarity to the seed centroid,
and held-out positives are counted in the top-K. Produces a `*__trials.csv` per run.

Run once per disease and once per model being benchmarked:

```bash
# SEMPER — Parkinson's disease
python run_target_prioritization.py \
    --seedplan      data/neuro_seedplan_PD.json \
    --pd_gene_file  data/disease_genes/raw_targets_PD.json \
    --mapping_json  data/ensp_to_gene.json \
    --embed_path    outputs/contrastive/esm_projected.pth \
    --label_dict    outputs/contrastive/esm_projected_labels.txt \
    --model_name    SEMPER_ESMCLIP \
    --out_prefix    outputs/evaluate/trials/neuro_eval_SEMPER_ESMCLIP__ct_network_PD \
    --device        cuda \
    --k_values 5 10 25

# Repeat for AZ, MS, ALS, HD by swapping --seedplan, --pd_gene_file, --out_prefix
# Repeat for baseline models (GNN_PINNACLE, PPI_GT_Encoder, etc.) by swapping
# --embed_path, --label_dict, and --model_name
```

**Output:** `<out_prefix>__trials.csv` — one row per (model, cluster, trial, k)

---

## 3. Plot Hits@K across models (Figure 4c–g, Step 2)

Aggregates trials CSVs across models and plots per-cluster Hits@K distributions
for all benchmarked models side by side.

```bash
python plot_hits_at_k.py \
    --trials_csv \
        outputs/evaluate/trials/*PD*__trials.csv \
        outputs/evaluate/trials/*PD*protrek*__trials.csv \
    --out_dir figures/hits_at_k_PD \
    --k_values 5 10 25

# Repeat for AZ, MS, ALS, HD
```

---

## 4. Statistical significance vs. baselines (Figure 4, Step 3)

One-sided paired Wilcoxon test comparing SEMPER's mean Hits@K to each baseline
across matched cell type clusters.

```bash
python calc_hits_significance.py \
    --trials_csv    "outputs/evaluate/trials/*__trials.csv" \
    --semper_model  SEMPER \
    --baselines     PINNACLE PPI-GT ORCA-CT ESM-C \
    --k_values      5 10 25
```

---

## Summary

| Script | Figure | What it does |
|---|---|---|
| `cluster_and_enrich.py` | Fig 2e | Leiden + UMAP + GO enrichment for a specific cell type cluster |
| `run_target_prioritization.py` | Fig 4 (Step 1) | Compute Hits@K trials CSV per disease and model |
| `plot_hits_at_k.py` | Fig 4c–g (Step 2) | Multi-model Hits@K comparison plot |
| `calc_hits_significance.py` | Fig 4 stats (Step 3) | Paired Wilcoxon significance vs. baselines |
