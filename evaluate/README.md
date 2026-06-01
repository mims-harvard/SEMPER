<<<<<<< Updated upstream
# SEMPER

**SEMPER** (**S**ingle-cell **E**mbeddings from **M**ultimodal **P**rotein **E**ncoder **R**epresentations) is a multimodal deep learning framework for generating cell type-aware protein representations by jointly modeling protein interaction network topology and amino acid sequence.

SEMPER learns a shared latent space that bridges a protein's intrinsic evolutionary sequence identity (via a frozen ESM-Cambrian language model) with its dynamic, cell type-specific topological context (via a multi-scale PPI Graph Transformer), yielding embeddings optimized for downstream biomedical tasks across diverse CNS cell types and tissues.

---

## Overview

SEMPER consists of three sequential components:

1. PPI-GT Encoder: Multi-scale graph transformer trained on cell type-specific PPI networks + BTO metagraph
2. Frozen ESM-Cambrian (600M) sequence embeddings extracted from canonical amino acid sequences
3. Symmetric InfoNCE alignment of both modalities into a shared cell type-aware latent space

---

## Repository Structure

```
SEMPER/
├── environment.yml              # Conda environment
│
├── ppi_gt/                      # Component 1: PPI Graph Transformer encoder
│   ├── train.py                 # Main training loop (DDP)
│   ├── model.py                 # PPIGT model architecture
│   ├── conv.py                  # PCTConv and PPIConv graph convolutions
│   ├── loss.py                  # Link prediction + center loss
│   ├── center_loss.py           # CenterLoss module
│   ├── minibatch_utils.py       # GraphSAINT batch generation
│   ├── generate_input.py        # Data loading
│   ├── utils.py                 # Metrics, plotting, embedding utilities
│   ├── parse_args.py            # CLI arguments
│   └── run_ppi_gt.sh            # Example training script
│
├── esm/                         # Component 2: ESM-C sequence encoder
│   ├── fetch_sequences.py       # Sequence acquisition pipeline (Ensembl + UniProt)
│   ├── generate_ESMC_embeds.py  # ESM-Cambrian inference + embedding extraction
│   └── run_esm.sh               # Example inference script
│
└── contrastive/                 # Component 3: Multimodal contrastive alignment
    ├── train_contrastive_MLP.py # Projection head training (symmetric InfoNCE)
    └── run_contrastive.sh       # Example training script
```

---

## Installation

**Requirements:** Linux, CUDA 11.8, conda.

```bash
# 1. Clone the repository
git clone https://github.com/mims-harvard/SEMPER.git
cd SEMPER

# 2. Create and activate the conda environment
conda env create -f environment.yml
conda activate semper

# 3. Install PyTorch Geometric wheels matched to PyTorch 2.1.0 + CUDA 11.8
pip install torch-scatter torch-sparse torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

---

## Data

SEMPER was trained on the following inputs:

| Input | Description |
|---|---|
| Background interactome | Global human reference PPI network |
| Cell type-specific PPI networks | One edgelist per CNS cell type |
| BTO metagraph | Cell type / tissue hierarchy edgelist |
| Protein sequences | Canonical amino acid sequences (FASTA) retrieved via Ensembl and UniProt |

---

## Usage

Components 1 and 2 can be run in parallel. Component 3 requires outputs from both.

### Component 1 — PPI Graph Transformer

Trains the multi-scale graph transformer encoder on cell type-specific PPI networks and the BTO metagraph using GraphSAINT minibatch sampling and DDP across 4 GPUs.

```bash
cd ppi_gt/
bash run_ppi_gt.sh
```

Key arguments:

| Argument | Description | Default |
|---|---|---|
| `--G_f` | Path to global background PPI edgelist | — |
| `--ppi_dir` | Directory of cell type-specific PPI edgelists | — |
| `--mg_f` | Path to BTO metagraph edgelist | — |
| `--feat_mat` | Initial node feature dimension | `512` |
| `--hidden` | Hidden layer dimension | `16` |
| `--output` | Output embedding dimension | `8` |
| `--n_heads` | Number of attention heads | `8` |
| `--batch_size` | GraphSAINT minibatch size | `32` |
| `--epochs` | Training epochs | `300` |
| `--theta` | PPI vs. metagraph loss mixing coefficient | `0.1` |
| `--lmbda` | Center loss weight λ | `0.01` |
| `--lr` | Learning rate | `0.001` |
| `--lr_cent` | Center loss learning rate | `0.01` |
| `--dropout` | Dropout rate | `0.5` |
| `--save_prefix` | Output directory for embeddings | — |

**Outputs** (written to `--save_prefix`):
- `_protein_embed.pth` — per-cluster protein embeddings `{ct_idx -> tensor[n_proteins, D]}`
- `_labels_dict.txt` — label dictionary mapping cluster indices to cell type names

---

### Component 2 — ESM-C Sequence Encoder

**Step 1:** Fetch canonical amino acid sequences from Ensembl and UniProt.

```bash
cd esm/
python fetch_sequences.py \
    --protein_list  data/unique_proteins.txt \
    --out_fasta     data/protein_sequences.fasta \
    --out_failed    data/failed_proteins.txt
```

The pipeline stratifies protein IDs into ENSP (Ensembl Protein) and UniProt Entry Name identifiers, retrieves sequences via the respective REST APIs, applies an archive-based fallback for deprecated ENSP IDs, and strips stop-codon characters (`*`) to prevent tokenization errors.

**Step 2:** Run frozen ESM-Cambrian (600M) inference to extract sequence embeddings.

```bash
bash run_esm.sh
```

Key arguments for `generate_ESMC_embeds.py`:

| Argument | Description | Default |
|---|---|---|
| `--fasta` | Path to input FASTA file | — |
| `--out_embeds` | Output path for embedding tensor (.pt) | — |
| `--out_index` | Output path for protein index (.pkl) | — |
| `--batch_size` | Sequences per inference batch | `16` |

**Outputs:**
- `protein_esmc_600m.pt` — embedding tensor `[N_proteins, 1152]`
- `protein_esmc_600m.pkl` — index mapping `{protein_id -> tensor_index}`

---

### Component 3 — Contrastive Alignment

Trains two lightweight MLP projection heads that align PPI graph embeddings and ESM-C sequence embeddings into a shared latent space via symmetric InfoNCE loss. Requires outputs from both Component 1 and Component 2.

```bash
cd contrastive/
bash run_contrastive.sh
```

Key arguments:

| Argument | Description | Default |
|---|---|---|
| `--embed_path` | PPI-GT protein embeddings (.pth) | — |
| `--label_path` | PPI-GT label dictionary (.txt) | — |
| `--esm_pickle` | ESM index file (.pkl) | — |
| `--esm_embeds` | ESM embedding tensor (.pt) | — |
| `--save_dir` | Output directory | — |
| `--proj_dim` | Shared latent space dimensionality | `512` |
| `--hidden_esm` | ESM MLP hidden layer sizes | `1024 512` |
| `--hidden_ppi` | PPI MLP hidden layer sizes | `512 512` |
| `--batch_size` | Training batch size | `64` |
| `--epochs` | Training epochs | `200` |
| `--lr` | Learning rate | `1e-4` |
| `--temperature` | InfoNCE temperature τ | `0.07` |

**Outputs** (written to `--save_dir`):
- `clip_mlps.ckpt` — projection head checkpoint (weights + config)
- `ppi_projected.pth` — projected PPI embeddings per cluster `{ct_idx -> tensor[n, 512]}`
- `esm_projected.pth` — projected ESM embeddings per cluster `{ct_idx -> tensor[n, 512]}`
- `joint_projected.pth` — concatenated joint embeddings per cluster `{ct_idx -> tensor[n, 1024]}`
- `esm_proj_embeddings.pt` — flat projected ESM tensor `[N_proteins, 512]`
- `esm_proj_ids.json` — ordered protein ID list corresponding to the flat tensor

---

## Experimental Tracking

SEMPER uses [Weights & Biases](https://wandb.ai) for experiment tracking and hyperparameter search. Before running Component 1, log in with:

```bash
wandb login
```

Set your W&B entity in `ppi_gt/train.py` or disable tracking by removing the `wandb` calls if preferred.

---

## Hardware

SEMPER was trained on the [Kempner Institute](https://kempner.harvard.edu) computing cluster.

| Component | Hardware | Notes |
|---|---|---|
| PPI-GT encoder | 4× NVIDIA H100 (80GB) | Distributed Data Parallel (DDP) |
| ESM-C inference | 1× NVIDIA H100 (80GB) | Frozen inference only |
| Contrastive alignment | 1× NVIDIA H100 (80GB) | Projection heads only |

---

## Citation

If you use SEMPER in your research, please cite:

```bibtex
@article{semper2025,
  title   = {},
  author  = {},
  journal = {},
  year    = {2025}
}
```

---

## License

[MIT License](LICENSE)

---

## Acknowledgements

SEMPER builds on [PINNACLE](https://github.com/mims-harvard/PINNACLE) and the [ESM](https://github.com/evolutionaryscale/esm) protein language model family. We thank the Kempner Institute for computational resources.
=======
# Evaluating SEMPER

This directory contains scripts to reproduce the core evaluation analyses from
the SEMPER paper. All scripts operate on `esm_projected.pth` and its companion
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
This is how Figure 2e was generated — on the dopaminergic neuron cluster (ID 395).

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
>>>>>>> Stashed changes
