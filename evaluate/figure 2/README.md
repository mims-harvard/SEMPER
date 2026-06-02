# Figure 2 — Cell type context and biological function structure the SEMPER latent space

This directory contains scripts to reproduce all panels of Figure 2 from the SEMPER paper.
All scripts require `esm_projected.pth` and `esm_projected_labels.txt` as base inputs,
plus panel-specific data files described below.

---

## Data requirements

Before running any script, download the required files and place them in the paths shown.
The recommended layout is:

```
data/
├── esm_projected.pth                          # SEMPER embeddings (Figshare)
├── esm_projected_labels.txt                   # Label dictionary (Figshare)
├── ensp_to_gene.json                          # ENSP → gene symbol map (Figshare)
│
├── science.add7046_table_s3.xlsx              # Fig 2a: Siletti et al. CNS supercluster map
│
├── ensp2go.json.gz                            # Fig 2b–d: ENSP → all GO term annotations
├── go2ontology.json.gz                        # Fig 2b–d: GO term → ontology namespace
│
├── ensp2lowestgo.json.gz                      # Fig 2e: ENSP → most-specific GO annotations
│                                              #         (different from ensp2go above)
│
├── laman_trip_association_scores/             # Fig 2f: Laman Trip et al. co-abundance CSVs
│   ├── cohorts_combined_brain_tumor_avg_outer_prob.csv
│   ├── cohorts_combined_ovary_tumor_avg_outer_prob.csv
│   └── ...                                   # one CSV per tissue
│
└── moonlightproteins_moondb_allemf_idmap.xlsx                       # Fig 2g: MoonDB EMF gene symbols
```

| File | Source | Notes |
|---|---|---|
| `esm_projected.pth`, `esm_projected_labels.txt` | [Figshare](#) | Primary SEMPER outputs |
| `science.add7046_table_s3.xlsx` | [Siletti et al., Science 2023](https://doi.org/10.1126/science.add7046) | Supplementary Table S3 — CNS cell type supercluster annotations |
| `ensp2go.json.gz`, `go2ontology.json.gz` | Already included in `data/` | Full GO term annotations per protein; GO term → BP/MF/CC namespace |
| `ensp2lowestgo.json.gz` | Already included in `data/` | Most-specific (lowest-level) GO annotations per protein; used only for Fig 2e enrichment |
| `laman_trip_association_scores/` | [Laman Trip et al., Nat Commun 2022](https://doi.org/10.1038/s41467-022-34544-2) | Protein co-abundance association scores across 11 tissues |
| `moondb_emf_idmap.xlsx` | Already included in `data/` | MoonDB EMF proteins (238 entries, columns: symbol, alias). Source: [MoonDB](https://moondb.hb.univ-amu.fr) |

---

## Panel-by-panel

### Figure 2a — Intra- vs. inter-cell-type pairwise cosine similarity

Mean pairwise cosine similarity of protein embeddings within the same cell type
(intra-cell-type, blue) vs. across different cell types (inter-cell-type, grey),
across 30 CNS superclusters. Mann-Whitney U test with BH-FDR correction.

```bash
python fig2a_pairwise_similarity.py \
    --embed-pt           data/esm_projected.pth \
    --labels             data/esm_projected_labels.txt \
    --supercluster-xlsx  data/science.add7046_table_s3.xlsx \
    --out-prefix         outputs/fig2a \
    --n-pairs            100000
```

| Argument | Description | Default |
|---|---|---|
| `--n-pairs` | Pairs sampled per cell type | `100000` |
| `--seed` | Random seed | `42` |

**Outputs:** `_raw_scores.csv`, `_aggregate_stats.csv`, `.pdf`, `.png`

---

### Figures 2b–d — GO term embedding dispersion vs. random null

Distribution of mean pairwise squared Euclidean distance between SEMPER embeddings
of proteins sharing a GO term, for Biological Process (BP), Cellular Component (CC),
and Molecular Function (MF). Solid lines mark SEMPER mean; dashed lines mark the
random null. One-sided t-test.

```bash
python fig2bcd_go_dispersion.py \
    --ensp2go        data/ensp2go.json.gz \
    --go2ontology    data/go2ontology.json.gz \
    --embed-pt       data/esm_projected.pth \
    --labels         data/esm_projected_labels.txt \
    --out-prefix     outputs/fig2bcd \
    --min-term-size  10 \
    --max-term-size  500
```

| Argument | Description | Default |
|---|---|---|
| `--min-term-size` | Minimum proteins per GO term | `10` |
| `--max-term-size` | Maximum proteins per GO term | `500` |
| `--no-l2-normalize` | Skip L2 normalization | flag |

**Outputs:** `_go_dispersions.csv.gz`, `_BP.pdf`, `_MF.pdf`, `_CC.pdf` (+ `.png`)

---

### Figure 2e — UMAP + Leiden clustering + GO enrichment for a specific cell type

UMAP projection of SEMPER embeddings for proteins active in a single cell type
(in the paper: dopaminergic neurons, cluster ID 395, N=3,035 proteins). Colors
correspond to Leiden clusters. Enriched GO terms annotated per module.

```bash
python fig2e_cluster_and_enrich.py \
    --embed-pt       data/esm_projected.pth \
    --labels         data/esm_projected_labels.txt \
    --target-id      395 \
    --ensp2lowestgo  data/ensp2lowestgo.json.gz \
    --go2ontology    data/go2ontology.json.gz \
    --resolution     1.0 \
    --n-neighbors    20 \
    --ontology       BP \
    --out-prefix     outputs/fig2e_DANeuron_c395
```

| Argument | Description | Default |
|---|---|---|
| `--target-id` | Numerical cluster ID to analyze | required |
| `--resolution` | Leiden resolution (higher = more, smaller modules) | `1.0` |
| `--n-neighbors` | Neighborhood size for graph construction | `20` |
| `--ontology` | GO namespace: `BP`, `MF`, `CC`, or `ALL` | `BP` |

**Outputs:** `_umap.pdf`, `_umap.png`, `_cluster_assignments.csv`, `_enrichment.csv`

To reproduce the paper exactly, run with `--target-id 395` (dopaminergic neuron cluster).
To analyze a different cell type, look up its cluster ID in `science.add7046_table_s3.xlsx`.

---

### Figure 2f — SEMPER cosine similarity vs. Laman Trip protein co-abundance

SEMPER pairwise embedding cosine similarity correlates with protein co-abundance scores
from Laman Trip et al. across 11 tissues. Brain-enriched CNS cell types show the
strongest correlation. Three-step pipeline combined into one script.

```bash
python fig2f_lamantrip_correlation.py \
    --embed-pt          data/esm_projected.pth \
    --labels            data/esm_projected_labels.txt \
    --ensp-json         data/ensp_to_gene.json \
    --assoc-dir         data/laman_trip_association_scores/ \
    --out-prefix        outputs/fig2f \
    --brain-threshold   0.68 \
    --work-dir          /tmp/semper_fig2f_workdir
```

| Argument | Description | Default |
|---|---|---|
| `--brain-threshold` | Min brain score for a pair to be brain-enriched (95th percentile in paper) | `0.68` |
| `--enrich-threshold` | Min % brain-enriched pairs to keep a cluster in Step 3 | median |
| `--work-dir` | Temp directory for per-cluster `.npy` files | `<out-prefix>_workdir` |
| `--keep-work-dir` | Retain temp files (useful for re-runs) | flag |

**Outputs:**
- `_step1_per_ct_correlations.csv` — per-cluster Spearman r for each tissue
- `_step2_brain_enrichment.csv` — per-cluster brain-enriched pair fraction
- `_step3_summary.csv` — median Spearman r summary (all vs. enriched clusters)
- `_all_clusters.pdf` / `_enriched_clusters.pdf` — tissue comparison plots

---

### Figure 2g — Centroid dispersion: EMF proteins vs. background proteome

Distribution of centroid dispersion — defined as the mean L2 distance of a
protein's cell type-specific embeddings from its global centroid — for
computationally predicted extreme multifunctional (EMF) proteins (n=243, red)
vs. all other proteins in the proteome (grey). Dashed lines mark group means.
Mann-Whitney U test.

Combines two steps into one run:
1. Compute per-protein centroid dispersion from `esm_projected.pth`
2. Map the MoonDB EMF gene list to ENSP IDs, split into EMF vs. background, run Mann-Whitney U, and plot the density histogram

The EMF gene set (n=243) was derived from MoonDB (https://moondb.hb.univ-amu.fr).
The idmap file is already included at `data/moondb_emf_idmap.xlsx` (columns: `symbol`, `alias`) — no download required.

```bash
python fig2g_emf_dispersion.py \
    --embed-pt      data/esm_projected.pth \
    --labels        data/esm_projected_labels.txt \
    --ensp-json     data/ensp_to_gene.json \
    --emf-xlsx      data/moondb_emf_idmap.xlsx \
    --out-prefix    outputs/fig2g \
    --metric        disp_centroid_mean_l2 \
    --exclude-k1
```

| Argument | Description | Default |
|---|---|---|
| `--emf-xlsx` | Path to MoonDB EMF idmap xlsx | required |
| `--metric` | Dispersion metric: `disp_centroid_mean_l2` or `disp_mean_pairwise_l2` | `disp_centroid_mean_l2` |
| `--exclude-k1` | Exclude proteins active in only 1 cell type | flag |
| `--top-k` | Number of top-dispersion outliers to save | `200` |

**Outputs:** `_protein_dispersion.csv`, `_top200_outliers.csv`, `_histogram.pdf`, `_histogram.png`
