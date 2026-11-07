# SEMPER benchmark datasets

Benchmark datasets used to evaluate SEMPER in the paper. They contain no SEMPER
outputs and can be used to benchmark any protein representation model.

| Folder | Contents | Paper |
|---|---|---|
| [`protein_complexes/`](protein_complexes) | 547 CNS-relevant CORUM complexes, and for each of four complex definitions the complexes present in each cell type with the size-matched decoys used for evaluation | Fig. 3, Supplementary Tables 4–5, Supplementary Note 1 |
| [`clinical_targets/`](clinical_targets) | Clinically pursued therapeutic targets for five neurodegenerative diseases, with clinical and pharmacological metadata for each target | Fig. 4–6, Supplementary Table 6 |
| [`da_neuron_essential_genes/`](da_neuron_essential_genes) | 693 genes essential to the survival of human iPSC-derived midbrain dopaminergic neurons | Fig. 5, Methods "Prediction of essential genes in dopaminergic neurons" |

## Cell type-specific protein complexes

`cns_complexes_547.csv` lists the 547 human CORUM complexes scored as CNS-relevant
(LLM relevance score of 4 or 5, at least three distinct subunits, every subunit in the
cell type-agnostic PPI network; Supplementary Note 1). It has the columns of
Supplementary Table 4 plus four columns `n_celltypes_<definition>` giving the number of
cell types in which the complex counts as present under each definition (0 if never).

A complex counts as present in a cell type under four definitions. Each definition has two
files, `<definition>_complexes.csv` (the complexes it uses) and
`<definition>_by_celltype.csv` (the complexes present in each cell type with their decoys):

| Definition (file prefix) | A complex is present in a cell type when | Complexes | Cell types with a complex | Cell types evaluated |
|---|---|---|---|---|
| `transcriptional_coexpression` | Every subunit is in the cell type's PPI network | 144 | 440 | 349 |
| `structural_connectivity` | Co-expression, and the subunits form a connected subgraph of the cell type's PPI network | 125 | 440 | 328 |
| `protein_coabundance` | Co-expression, and every subunit pair has brain co-abundance ≥ 0.5 in Laman Trip et al. (2026) | 40 | 410 | 213 |
| `spatial_colocalization` | Co-expression, and all subunits share a Human Protein Atlas subcellular location | 36 | 326 | 79 |

All complexes in these files come from `cns_complexes_547.csv`. Each present complex has ten
decoys drawn from the same cell type's network with the same number of subunits, and no decoy
matches a CORUM complex. A cell type is evaluated (scored by AUPRC in Fig. 3 and
Supplementary Table 5) when it has at least five present complexes (co-expression,
connectivity) or at least three (co-abundance, colocalization). The scripts that built these files are in
[`evaluate/fig3_protein_complexes/prepare_complexes/`](../../evaluate/fig3_protein_complexes/prepare_complexes). `prepare_coexpressed.py` builds `transcriptional_coexpression`,
`prepare_ppi_connected.py` builds `structural_connectivity`, `prepare_coabundance.py`
builds `protein_coabundance` and `prepare_hpa_colocalized.py` builds
`spatial_colocalization`. The [Figure 3 README](../../evaluate/fig3_protein_complexes/README.md)
gives the commands and inputs. These scripts need the 442 cell type networks, which are not
released, so the files here are provided ready to use.

`<definition>_complexes.csv`, one row per complex:

| Column | Description |
|---|---|
| `complex_id`, `complex_name` | CORUM identifier and name |
| `n_subunits` | Number of proteins |
| `subunits_gene_name`, `subunits_ensp` | Semicolon-delimited gene symbols and Ensembl protein IDs |
| `n_celltypes_present` | Number of cell types in which the complex is present under this definition |
| `n_celltypes_evaluated` | Of those, the number of evaluated cell types |

`<definition>_by_celltype.csv`, one row per complex or decoy in a cell type:

| Column | Description |
|---|---|
| `cell_type_id` | Cluster ID in the Siletti et al. (2023) adult human brain atlas |
| `cell_type_name`, `supercluster` | Cluster name and supercluster from Siletti et al. (2023) |
| `evaluated` | Whether the cell type meets the minimum number of present complexes |
| `label` | 1 for a present complex, 0 for a decoy |
| `complex_id`, `complex_name` | CORUM identifier and name (present complexes only) |
| `matched_complex_id` | CORUM identifier of the present complex that the row belongs to (each decoy is matched to one) |
| `n_subunits` | Number of proteins |
| `subunit_genes`, `subunit_ensp` | Semicolon-delimited gene symbols and Ensembl protein IDs |

## Clinically pursued therapeutic targets

One CSV per disease, one row per target gene:

| File | Disease | Targets |
|---|---|---|
| `AD_clinical_targets.csv` | Alzheimer's disease | 130 |
| `PD_clinical_targets.csv` | Parkinson's disease | 97 |
| `MS_clinical_targets.csv` | Multiple sclerosis | 166 |
| `HD_clinical_targets.csv` | Huntington's disease | 86 |
| `ALS_clinical_targets.csv` | Amyotrophic lateral sclerosis | 98 |

Targets were retrieved from the [Open Targets Platform](https://platform.opentargets.org)
(ChEMBL evidence, June 2025 release) and restricted to protein-coding genes targeted by
drugs or compounds in completed Phase 2 trials or in Phase 3 or 4 trials of any status. The code that builds these lists from
the Open Targets release is in [`data_prep/parse_opentargets/`](../../data_prep/parse_opentargets).

| Column | Description |
|---|---|
| `diseaseId` | MONDO/EFO disease identifier(s) |
| `targetId_genename` | HGNC gene symbol |
| `targetId` | Ensembl gene ID (ENSG) |
| `targetFromSourceId` | UniProt accession, or the Ensembl gene ID for the two antisense oligonucleotide targets (*HTT*, *SOD1*) |
| `Max Completed Clinical Phase` | Highest clinical phase reached by any associated drug or compound (Phase 2 counts only if completed, Phase 3 and 4 at any trial status) |
| `Associated drugId(s)` | ChEMBL IDs of associated drugs and compounds |
| `drugbank` | Corresponding DrugBank IDs, where available |
| `Target/Enzyme/Transporter` | DrugBank role of the protein, where annotated |
| `Pharmacological action` | Mechanism of the drug–target interaction, where annotated |
| `Therapeutic Intent` | Manually curated label: disease-modifying, symptomatic, or both |
| `Notes` | Free-text curation notes |

### Target lists used in the analyses

[`analysis_target_lists/`](clinical_targets/analysis_target_lists) holds the exact target
lists that the evaluation scripts read for Figs. 4–6 (`--pd_gene_file`), one JSON per
disease with the targets under the key `clinically_relevant_targets`. They were produced
from the same Open Targets release and contain the targets in the CSVs.

| File | Disease | Targets | Not in the CSV |
|---|---|---|---|
| `AD_targets_MONDO_0004975.json` | Alzheimer's disease | 133 | ADRA2A, ADRA2B, GLP1R |
| `PD_targets_MONDO_0005180.json` | Parkinson's disease | 98 | SCN7A |
| `MS_targets_MONDO_0005301.json` | Multiple sclerosis | 168 | HRH2, KCND3 |
| `HD_targets_MONDO_0007739.json` | Huntington's disease | 86 | |
| `ALS_targets_MONDO_0004976.json` | Amyotrophic lateral sclerosis | 99 | SCN7A |

Use these files to reproduce the paper's results. The JSONs also list
`all_associated_genes` and `all_druggable_targets` from the same Open Targets query.

## Dopaminergic neuron essential genes

`da_neuron_essential_genes.csv` has one column, `gene_symbol`, listing the 693 genes
identified as essential to the survival of human iPSC-derived midbrain dopaminergic neurons
in the genome-wide CRISPR screen of Nazeen et al., *Cell Genomics* (2026). SEMPER retrieves these genes
from the dopaminergic neuron representation (cell type 395) given seed sets of 20 essential
genes (Methods). Please cite Nazeen et al. when using this list.