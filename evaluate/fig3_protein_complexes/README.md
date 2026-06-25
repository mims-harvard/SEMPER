# Figure 3 — Zero-shot identification of cell type-specific protein complexes

This directory reproduces Figure 3: SEMPER identifies cell type-specific protein
complexes zero-shot by scoring mean pairwise cosine similarity across complex members,
benchmarked against all baselines under four definitions of "cell type-specific complex."

The pipeline has two parts for each panel:

1. **`prepare_complexes/`** — assign CORUM complexes to cell types and generate
   matched positives + negatives. One script per complex definition (panels c–f).

2. **`evaluate_models/`** — score all models on any `posneg.csv` and plot AUPRC
   distributions. The same scripts run identically for all four panels.

---

## Data requirements

```
data/
├── filtered_complexes_min3.csv           # CORUM complexes with ≥3 ENSP members
│                                         # https://mips.helmholtz-muenchen.de/corum/
├── ppi_edgelists/                        # Cell type-specific PPI edgelists
│   └── <N>_edgelist.txt                  # 442 files, one per CNS cell type
├── ensp_id2vec_mean.pkl.gz               # ProTrek ENSP→vec map
│                                         # Defines the allowed protein universe
├── subcellular_location_data_ensp.tsv    # [panel e] HPA subcellular localization
│                                         # https://www.proteinatlas.org/about/download
├── cohorts_combined_brain_tumor_avg_outer_prob.csv  # [panel f] Laman Trip brain scores
│                                         # https://doi.org/10.1038/s41467-022-34544-2
│   # Embeddings:
├── esm_projected.pth / esm_projected_labels.txt   # SEMPER
├── ppi_gt_embed.pth / ppi_gt_labels.txt           # PPI-GT
├── pinnacle_embed.pth / pinnacle_labels.txt        # PINNACLE
├── protein_esmc_600m.pkl / protein_esmc_600m.pt   # ESM-C
├── ensp_to_gene.json                              # ENSP → gene symbol map
│   # Graphlet features:
├── transposenet_orbits.txt / transposenet_nodes.json  # ORCA global
└── ppi_graphlets_orca/                            # ORCA-CT (per cell type)
    └── <N>_edgelist_nodes.json / <N>_edgelist_orbits.txt
```

---

## Part 1 — Prepare complexes

### Panel c — Co-expressed AND physically interact in PPI network

```bash
python prepare_complexes/prepare_ppi_connected.py \
    --complexes          data/filtered_complexes_min3.csv \
    --ppi-dir            data/ppi_edgelists/ \
    --protrek-pkl        data/ensp_id2vec_mean.pkl.gz \
    --keep-cols          complex_name,subunits_ensp \
    --neg-per-pos        10 \
    --avoid-pos-members \
    --dedupe-globally \
    --out-dir            data/prepared/ppi_connected/
```

### Panel d — Co-expressed in cell type (no PPI connectivity requirement)

```bash
python prepare_complexes/prepare_coexpressed.py \
    --complexes          data/filtered_complexes_min3.csv \
    --ppi-dir            data/ppi_edgelists/ \
    --protrek-pkl        data/ensp_id2vec_mean.pkl.gz \
    --keep-cols          complex_name,subunits_ensp \
    --neg-per-pos        10 \
    --avoid-pos-members \
    --dedupe-globally \
    --out-dir            data/prepared/coexpressed/
```

### Panel e — Co-expressed AND HPA co-localized

```bash
python prepare_complexes/prepare_hpa_colocalized.py \
    --complexes          data/filtered_complexes_min3.csv \
    --hpa                data/subcellular_location_data_ensp.tsv \
    --ppi-dir            data/ppi_edgelists/ \
    --protrek-pkl        data/ensp_id2vec_mean.pkl.gz \
    --keep-cols          ComplexID,Complex_name \
    --neg-per-pos        10 \
    --avoid-pos-members \
    --dedupe-globally \
    --out-dir            data/prepared/hpa_colocalized/
```

### Panel f — Co-expressed AND brain co-abundant (Laman Trip et al.)

Panel f starts from the panel d `posneg.csv` and filters positives by brain
protein co-abundance score.

```bash
python prepare_complexes/prepare_coabundance.py \
    --posneg-csv   data/prepared/coexpressed/posneg.csv \
    --assoc-brain  data/cohorts_combined_brain_tumor_avg_outer_prob.csv \
    --ensp-json    data/ensp_to_gene.json \
    --out-dir      data/prepared/coabundance/ \
    --thresholds   0.5 0.8
```

Each script writes two files to `--out-dir`:
- `assignments.csv` — (complex, cell type) positive pairs
- `posneg.csv` — positives + matched negatives, ready for scoring

---

## Part 2 — Evaluate models

Run all scorers on the `posneg.csv` produced above.
Replace `<POSNEG>` with the path for the panel you are reproducing.

```bash
POSNEG=data/prepared/ppi_connected/posneg.csv   # change per panel

# SEMPER
python evaluate_models/score_semper_or_ppigt.py \
    --eval-csv         $POSNEG \
    --model-name       SEMPER \
    --model-pt         data/esm_projected.pth \
    --model-labels     data/esm_projected_labels.txt \
    --out-scores-csv   out_scores/semper_scores.csv \
    --out-metrics-json out_scores/semper_metrics.json

# PPI-GT
python evaluate_models/score_semper_or_ppigt.py \
    --eval-csv         $POSNEG \
    --model-name       PPI-GT \
    --model-pt         data/ppi_gt_embed.pth \
    --model-labels     data/ppi_gt_labels.txt \
    --out-scores-csv   out_scores/ppigt_scores.csv \
    --out-metrics-json out_scores/ppigt_metrics.json

# PINNACLE
python evaluate_models/score_semper_or_ppigt.py \
    --eval-csv         $POSNEG \
    --model-name       PINNACLE \
    --model-pt         data/pinnacle_embed.pth \
    --model-labels     data/pinnacle_labels.txt \
    --out-scores-csv   out_scores/pinnacle_scores.csv \
    --out-metrics-json out_scores/pinnacle_metrics.json

# ProTrek Seq
python evaluate_models/score_protrek.py \
    --eval-csv         $POSNEG \
    --model-name       "ProTrek Seq" \
    --protrek-pkl-gz   data/ensp_id2vec_mean.pkl.gz \
    --out-scores-csv   out_scores/protrek_seq_scores.csv \
    --out-metrics-json out_scores/protrek_seq_metrics.json

# ProTrek Struct (use separate pkl if different from Seq)
python evaluate_models/score_protrek.py \
    --eval-csv         $POSNEG \
    --model-name       "ProTrek Struct" \
    --protrek-pkl-gz   data/ensp_id2vec_struct_mean.pkl.gz \
    --out-scores-csv   out_scores/protrek_struct_scores.csv \
    --out-metrics-json out_scores/protrek_struct_metrics.json

# ESM-C
python evaluate_models/score_esm.py \
    --eval-csv         $POSNEG \
    --model-name       ESM-C \
    --esm-pickle       data/protein_esmc_600m.pkl \
    --esm-pt           data/protein_esmc_600m.pt \
    --out-scores-csv   out_scores/esm_scores.csv \
    --out-metrics-json out_scores/esm_metrics.json

# ORCA (global graphlets)
python evaluate_models/score_orca_global.py \
    --eval-csv         $POSNEG \
    --orbits           data/transposenet_orbits.txt \
    --node-map         data/transposenet_nodes.json \
    --model-name       ORCA \
    --out-scores-csv   out_scores/orca_scores.csv \
    --out-metrics-json out_scores/orca_metrics.json

# ORCA-CT (cell type-specific graphlets)
python evaluate_models/score_orca_ct.py \
    --eval-csv           $POSNEG \
    --ct-orbits-dir      data/ppi_graphlets_orca/ \
    --model-name         ORCA-CT \
    --neg-missing-action use_row_ct \
    --out-scores-csv     out_scores/orca_ct_scores.csv \
    --out-metrics-json   out_scores/orca_ct_metrics.json
```

### Plot AUPRC distributions

```bash
python evaluate_models/plot_auprc.py \
    --group-by        cluster \
    --model           SEMPER=out_scores/semper_scores.csv \
    --model           PINNACLE=out_scores/pinnacle_scores.csv \
    --model           PPI-GT=out_scores/ppigt_scores.csv \
    --model           "ProTrek Seq=out_scores/protrek_seq_scores.csv" \
    --model           "ProTrek Struct=out_scores/protrek_struct_scores.csv" \
    --model           ESM-C=out_scores/esm_scores.csv \
    --model           ORCA=out_scores/orca_scores.csv \
    --model           ORCA-CT=out_scores/orca_ct_scores.csv \
    --min-n           50 \
    --aggregate-models "ProTrek Seq,ProTrek Struct,ESM-C,ORCA" \
    --out-prefix      figures/fig3c \
    --title           "Members co-expressed and physically interact in cell type"
# Repeat with different --out-prefix and --title for panels d, e, f
```
