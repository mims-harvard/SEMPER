# Figure 3 — Zero-shot identification of cell type-specific protein complexes

This directory reproduces Figure 3b–e and Supplementary Table 5. SEMPER identifies
cell type-specific protein complexes zero-shot by scoring the mean pairwise cosine
similarity of a complex's subunits in a cell type, and is benchmarked against all
baselines under four definitions of a cell type-specific complex.

| Panel | Definition (file prefix in `data/benchmarks/protein_complexes/`) | Min rows per cell type (`--min-n`) | Cell types evaluated |
|---|---|---|---|
| b | `transcriptional_coexpression` | 50 | 349 |
| c | `structural_connectivity` | 50 | 328 |
| d | `protein_coabundance` | 30 | 213 |
| e | `spatial_colocalization` | 30 | 79 |

Each present complex has ten size-matched decoys, so 50 and 30 rows mean at least five
and three present complexes. These are the cell types with `evaluated = True` in the
released files.

The directory has two parts:

1. **`evaluate_models/`** — score every model on a benchmark file, plot per-cell-type
   AUPRC (Fig. 3b–e) and test SEMPER against the cell type-specific baselines
   (Supplementary Table 5). The same commands run for all four panels.

2. **`prepare_complexes/`** — how the benchmark files were built (assign CORUM
   complexes to cell types and draw decoys). You do not need to run these to reproduce
   the figure. The released `*_by_celltype.csv` files are the inputs used in the paper.

---

## Data requirements

Run all commands from `evaluate/fig3_protein_complexes/`.

```
../../data/benchmarks/protein_complexes/
└── <definition>_by_celltype.csv             # released benchmark inputs (one per panel)

data/
│   # Embeddings:
├── semper_embed.pth / semper_labels.txt     # SEMPER (semper/contrastive/build_semper_embeddings.py, t = 0.10)
├── ppi_gt_embed.pth / ppi_gt_labels.txt     # PPI-GT
├── pinnacle_embed.pth / pinnacle_labels.txt # PINNACLE
├── protein_esmc_600m.pkl / protein_esmc_600m.pt   # ESM-C
├── ensp_id2vec_mean.pkl.gz                  # ProTrek sequence ENSP→vec map
├── ensp_id2vec_structure_mean.pkl.gz        # ProTrek structure ENSP→vec map
│   # Graphlet features:
├── transposenet_orbits.txt / transposenet_nodes.json   # ORCA global
└── ppi_graphlets_orca/                      # ORCA-CT (per cell type)
    └── <N>_edgelist_nodes.json / <N>_edgelist_orbits.txt
```

---

## Part 1 — Score models

The scorers default to the column names of the lab files (`pos_key`, `cell_type`,
`members_ensp`, `k`). For the released files, pass the overrides in `$COLS`. Rows are
grouped by cell type (`cell_type_id`). The key column is only copied to the output and
does not affect any metric.

```bash
BENCH=../../data/benchmarks/protein_complexes
DEF=structural_connectivity   # transcriptional_coexpression | structural_connectivity | protein_coabundance | spatial_colocalization
EVAL=$BENCH/${DEF}_by_celltype.csv
OUT=out_scores/$DEF; mkdir -p $OUT
COLS="--key-col matched_complex_id --celltype-col cell_type_id --members-col subunit_ensp --k-col n_subunits"

# SEMPER
python evaluate_models/score_semper.py --eval-csv $EVAL $COLS \
    --model-name       SEMPER \
    --model-pt         data/semper_embed.pth \
    --model-labels     data/semper_labels.txt \
    --out-scores-csv   $OUT/semper_scores.csv \
    --out-metrics-json $OUT/semper_metrics.json

# PPI-GT (same scorer, different embedding)
python evaluate_models/score_semper.py --eval-csv $EVAL $COLS \
    --model-name       PPI-GT \
    --model-pt         data/ppi_gt_embed.pth \
    --model-labels     data/ppi_gt_labels.txt \
    --out-scores-csv   $OUT/ppigt_scores.csv \
    --out-metrics-json $OUT/ppigt_metrics.json

# PINNACLE (same scorer, different embedding)
python evaluate_models/score_semper.py --eval-csv $EVAL $COLS \
    --model-name       PINNACLE \
    --model-pt         data/pinnacle_embed.pth \
    --model-labels     data/pinnacle_labels.txt \
    --out-scores-csv   $OUT/pinnacle_scores.csv \
    --out-metrics-json $OUT/pinnacle_metrics.json

# ProTrek Seq
python evaluate_models/score_protrek.py --eval-csv $EVAL $COLS \
    --model-name       "ProTrek Seq" \
    --protrek-pkl-gz   data/ensp_id2vec_mean.pkl.gz \
    --out-scores-csv   $OUT/protrek_seq_scores.csv \
    --out-metrics-json $OUT/protrek_seq_metrics.json

# ProTrek Struct
python evaluate_models/score_protrek.py --eval-csv $EVAL $COLS \
    --model-name       "ProTrek Struct" \
    --protrek-pkl-gz   data/ensp_id2vec_structure_mean.pkl.gz \
    --out-scores-csv   $OUT/protrek_struct_scores.csv \
    --out-metrics-json $OUT/protrek_struct_metrics.json

# ESM-C
python evaluate_models/score_esm.py --eval-csv $EVAL $COLS \
    --model-name       ESM-C \
    --esm-pickle       data/protein_esmc_600m.pkl \
    --esm-pt           data/protein_esmc_600m.pt \
    --out-scores-csv   $OUT/esm_scores.csv \
    --out-metrics-json $OUT/esm_metrics.json

# ORCA (global graphlets)
python evaluate_models/score_orca_global.py --eval-csv $EVAL $COLS \
    --orbits           data/transposenet_orbits.txt \
    --node-map         data/transposenet_nodes.json \
    --model-name       ORCA \
    --out-scores-csv   $OUT/orca_scores.csv \
    --out-metrics-json $OUT/orca_metrics.json

# ORCA-CT (cell type-specific graphlets; decoys use the orbits of the row's cell type)
python evaluate_models/score_orca_ct.py --eval-csv $EVAL $COLS \
    --ct-orbits-dir      data/ppi_graphlets_orca/ \
    --model-name         ORCA-CT \
    --neg-missing-action use_row_ct \
    --out-scores-csv     $OUT/orca_ct_scores.csv \
    --out-metrics-json   $OUT/orca_ct_metrics.json
```

Each scorer writes one row per scored complex or decoy (`label`, `cluster_label`,
`score_model`) and prints the pooled AUROC and AUPRC. Scoring all four definitions with
SEMPER takes under two minutes and about 3 GB of memory.

### Plot AUPRC distributions (Fig. 3b–e)

Use the `--min-n` of the panel (table above).

```bash
MIN_N=50   # 50 for transcriptional_coexpression and structural_connectivity, 30 otherwise

python evaluate_models/plot_auprc.py \
    --group-by        cluster \
    --model           SEMPER=$OUT/semper_scores.csv \
    --model           PINNACLE=$OUT/pinnacle_scores.csv \
    --model           PPI-GT=$OUT/ppigt_scores.csv \
    --model           "ProTrek Seq=$OUT/protrek_seq_scores.csv" \
    --model           "ProTrek Struct=$OUT/protrek_struct_scores.csv" \
    --model           ESM-C=$OUT/esm_scores.csv \
    --model           ORCA=$OUT/orca_scores.csv \
    --model           ORCA-CT=$OUT/orca_ct_scores.csv \
    --min-n           $MIN_N \
    --aggregate-models "ProTrek Seq,ProTrek Struct,ESM-C,ORCA" \
    --out-prefix      figures/fig3_$DEF \
    --title           "Members co-expressed in cell type and connected in PPI network"   # set per panel
```

Cell type-specific models are shown with one dot per evaluated cell type. Models without
cell type context (ProTrek, ESM-C, ORCA) are shown as one pooled AUPRC.

### Significance tests (Supplementary Table 5)

Paired one-sided Wilcoxon signed-rank test (`alternative="greater"`) of per-cell-type
AUPRC, SEMPER vs each cell type-specific baseline, over the evaluated cell types.

```bash
python evaluate_models/compare_models_wilcoxon.py \
    --semper    $OUT/semper_scores.csv \
    --baseline  PINNACLE=$OUT/pinnacle_scores.csv \
    --baseline  PPI-GT=$OUT/ppigt_scores.csv \
    --baseline  ORCA-CT=$OUT/orca_ct_scores.csv \
    --min-n     $MIN_N
```

It prints N (paired cell types), W and P for each baseline. Add `--metric auroc` for
AUROC and `--out-csv` to save the table. Expected values:

| Definition | N | P vs PINNACLE | P vs PPI-GT | P vs ORCA-CT | Median SEMPER AUPRC |
|---|---|---|---|---|---|
| Co-expression | 349 | 3.30e-59 | 7.87e-59 | 3.01e-59 | 0.697 |
| Connectivity | 328 | 1.05e-55 | 2.17e-55 | 8.24e-56 | 0.743 |
| Co-abundance | 213 | 7.62e-37 | 2.21e-36 | 7.25e-37 | 0.761 |
| Colocalization | 79 | 6.45e-15 | 1.71e-13 | 7.51e-15 | 0.612 |

---

## Part 2 — How the benchmark files were built (optional)

One script per definition assigns CORUM complexes to cell types and draws ten decoys per
present complex from the same cell type's PPI network with the same number of subunits
(seed 42). The outputs use the lab column names (`pos_key`, `cell_type`, `members_ensp`,
`k`), so the scorers above run on them without `$COLS`. The released `*_by_celltype.csv`
files were built with these steps, with cell type names, superclusters and gene symbols
added. To reproduce the paper's numbers, score the released files rather than
regenerated ones.

Additional inputs:

```
data/
├── filtered_complexes_min3.csv           # CORUM complexes with ≥3 ENSP members
├── ppi_edgelists/<N>_edgelist.txt        # cell type PPI networks (442 files)
├── ensp_id2vec_mean.pkl.gz               # ProTrek map, defines the allowed protein universe
├── subcellular_location_data_ensp.tsv    # [panel e] HPA subcellular localization
├── cohorts_combined_brain_tumor_avg_outer_prob.csv  # [panel d] Laman Trip et al. brain co-abundance
└── ENSP_to_pref.json                     # [panel d] ENSP → gene symbol
```

### Panel b — Co-expressed in cell type

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

### Panel c — Co-expressed AND connected in the cell type PPI network

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

### Panel d — Co-expressed AND brain co-abundant (Laman Trip et al.)

Starts from the panel b `posneg.csv`, keeps the positives whose subunit pairs all have
brain co-abundance ≥ 0.5, and draws this definition's own 1:10 decoys. The co-expression
decoys are not reused.

```bash
python prepare_complexes/prepare_coabundance.py \
    --posneg-csv   data/prepared/coexpressed/posneg.csv \
    --assoc-brain  data/cohorts_combined_brain_tumor_avg_outer_prob.csv \
    --ensp-json    data/ENSP_to_pref.json \
    --ppi-dir      data/ppi_edgelists/ \
    --protrek-pkl  data/ensp_id2vec_mean.pkl.gz \
    --thresholds   0.5 \
    --neg-per-pos  10 \
    --avoid-pos-members \
    --dedupe-globally \
    --seed         42 \
    --out-dir      data/prepared/coabundance/
```

This writes `posneg_coabundance_t0p5.csv` (positives and decoys) and
`coab_assignments_t0p5.csv` (positives only).

### Panel e — Co-expressed AND HPA co-localized

```bash
python prepare_complexes/prepare_hpa_colocalized.py \
    --complexes          data/filtered_complexes_min3.csv \
    --hpa                data/subcellular_location_data_ensp.tsv \
    --ppi-dir            data/ppi_edgelists/ \
    --protrek-pkl        data/ensp_id2vec_mean.pkl.gz \
    --keep-cols          complex_id,complex_name \
    --neg-per-pos        10 \
    --avoid-pos-members \
    --dedupe-globally \
    --out-dir            data/prepared/hpa_colocalized/
```

Panels b, c and e write two files to `--out-dir`:
- `assignments.csv` — (complex, cell type) positive pairs
- `posneg.csv` — positives + matched decoys, ready for scoring
