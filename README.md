# SEMPER: SystEm-scale Multimodal ProtEin Representations

[![Project Page](https://img.shields.io/badge/Project_Page-mims--harvard-blue)](https://zitniklab.hms.harvard.edu/SEMPER/)
[![Paper](https://img.shields.io/badge/Paper-SEMPER-green)](#)
[![Code](https://img.shields.io/badge/Code-SEMPER-purple)](https://github.com/mims-harvard/SEMPER)
[![Model](https://img.shields.io/badge/HuggingFace-SEMPER-yellow)](https://huggingface.co/mims-harvard/SEMPER)
[![License](https://img.shields.io/badge/License-MIT-lightgreen)]([https://huggingface.co/mims-harvard/SEMPER](https://opensource.org/license/MIT))

Proteins carry out cellular functions largely through their interactions with other proteins. Because the set of proteins expressed differs from one cell type to the next, a given protein can engage different partners depending on its cellular context. This context dependence is well documented but difficult to represent computationally, and it is particularly relevant in the central nervous system (CNS), whose diversity of neuronal and glial cell types has been linked to the cell type specific vulnerability observed across neurodegenerative diseases. Yet existing protein representation methods capture only part of the picture: sequence-based models assign each protein a single, context-free embedding, while network-based models incorporate cellular context but set sequence information aside.

**SEMPER** (**S**yst**E**m-scale **M**ultimodal **P**rot**E**in **R**epresentations) is a multimodal deep learning model that learns cell type-aware protein embeddings that combine both. A frozen hybrid graph transformer encodes each protein's context within cell type specific protein–protein interaction (PPI) networks and a metagraph of cellular and tissue organization, and a frozen protein sequence encoder represents its amino acid sequence; a symmetric InfoNCE objective aligns the two into a shared latent space. Trained across 442 cell types of the human CNS, SEMPER assigns the same protein a distinct embedding in each cellular context, so that its representation reflects where in the brain it acts.

Across our analyses, this embedding space recovers cellular and functional organization: proteins cluster by cell type and by shared function, multifunctional proteins spread across contexts as expected, and cell type specific protein complexes are recovered more accurately than by the baselines we tested. Building on these representations, SEMPER can prioritize candidate therapeutic targets for five neurodegenerative diseases in a zero-shot setting, and its embedding-derived signals can be combined with large language model reasoning to refine those rankings. These predictions are associative and intended to support hypothesis generation rather than replace experimental validation.

---

## Overview

SEMPER integrates three inputs derived from an adult human CNS single-cell transcriptomic atlas: 1D protein sequences, 442 cell type specific PPI networks, and a hierarchical metagraph of cellular and tissue organization. A frozen protein sequence encoder and a frozen hybrid graph transformer encode each input independently; trainable adaptors project both into a shared latent space aligned by a symmetric InfoNCE contrastive loss.

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

**Requirements:** Linux, NVIDIA GPU with CUDA 11.8, conda.

```bash
# 1. Clone the repository
git clone https://github.com/mims-harvard/SEMPER.git
cd SEMPER

# 2. Create and activate the environment (installs PyTorch, PyG, and all
#    dependencies for data prep, training, and figures 2-4)
conda env create -f environment.yml
conda activate semper
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
@article{semper2026,
  title   = {SEMPER: System-Scale Multimodal Protein Embedding Representations},
  author  = {Dong, Amy and Li, Michelle M. and {[co-authors]} and Zitnik, Marinka},
  journal = {},
  year    = {2026},
  url     = {}
}
```

---

## License

[MIT License](LICENSE)

---

## Acknowledgements

SEMPER builds on [PINNACLE](https://github.com/mims-harvard/PINNACLE) and the [ESM](https://github.com/evolutionaryscale/esm) protein language model family. We thank the Kempner Institute for computational resources.
