# SEMPER: SystEm-scale Multimodal ProtEin Representations

[![Project Page](https://img.shields.io/badge/Project_Page-mims--harvard-blue)](https://zitniklab.hms.harvard.edu/SEMPER/)
[![Paper](https://img.shields.io/badge/Paper-SEMPER-green)](#)
[![Code](https://img.shields.io/badge/Code-SEMPER-purple)](https://github.com/mims-harvard/SEMPER)
[![Model](https://img.shields.io/badge/HuggingFace-SEMPER-yellow)](https://huggingface.co/mims-harvard/SEMPER)
[![License](https://img.shields.io/badge/License-MIT-lightgreen)](https://opensource.org/license/MIT)

## Overview

Proteins carry out cellular functions largely through their interactions with other proteins. Because the set of proteins expressed differs from one cell type to the next, a given protein can engage different partners depending on its cellular context. This context dependence is well documented but difficult to represent computationally, and it is particularly relevant in the central nervous system (CNS), whose diversity of neuronal and glial cell types has been linked to the cell type specific vulnerability observed across neurodegenerative diseases. Yet existing protein representation methods capture only part of the picture: sequence-based models assign each protein a single, context-free embedding, while network-based models incorporate cellular context but set sequence information aside.

**SEMPER** (**S**yst**E**m-scale **M**ultimodal **P**rot**E**in **R**epresentations) is a multimodal deep learning model that learns cell type-aware protein embeddings that combine both. A frozen hybrid graph transformer encodes each protein's context within cell type specific protein–protein interaction (PPI) networks and a metagraph of cellular and tissue organization, and a frozen protein sequence encoder represents its amino acid sequence; a symmetric InfoNCE objective aligns the two into a shared latent space. Trained across 442 cell types of the human CNS, SEMPER assigns the same protein a distinct embedding in each cellular context, so that its representation reflects where in the brain it acts.

Across our analyses, this embedding space recovers cellular and functional organization: proteins cluster by cell type and by shared function, multifunctional proteins spread across contexts as expected, and cell type specific protein complexes are recovered more accurately than by the baselines we tested. Building on these representations, SEMPER can prioritize candidate therapeutic targets for five neurodegenerative diseases in a zero-shot setting, and its embedding-derived signals can be combined with large language model reasoning to refine those rankings. These predictions are associative and intended to support hypothesis generation rather than replace experimental validation.

SEMPER integrates three inputs derived from an adult human CNS single-cell transcriptomic atlas: 1D protein sequences, 442 cell type specific PPI networks, and a hierarchical metagraph of cellular and tissue organization. A frozen protein sequence encoder and a frozen hybrid graph transformer encode each input independently; trainable adaptors project both into a shared latent space aligned by a symmetric InfoNCE contrastive loss.

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

## Training

Components 1 and 2 can be run in parallel. Component 3 requires outputs from both.

### PPI Graph Transformer

Trains the multi-scale graph transformer encoder on cell type-specific PPI networks and the BTO metagraph using GraphSAINT minibatch sampling and DDP across 4 GPUs.

```bash
cd ppi_gt/
bash run_ppi_gt.sh
```

<details>
<summary>Full argument list</summary>

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

</details>

**Outputs** (written to `--save_prefix`):
- `_protein_embed.pth` — per-cluster protein embeddings `{ct_idx -> tensor[n_proteins, D]}`
- `_labels_dict.txt` — label dictionary mapping cluster indices to cell type names

---

### ESM-C Sequence Encoder

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

<details>
<summary>Full argument list for `generate_ESMC_embeds.py`:</summary>

| Argument | Description | Default |
|---|---|---|
| `--fasta` | Path to input FASTA file | — |
| `--out_embeds` | Output path for embedding tensor (.pt) | — |
| `--out_index` | Output path for protein index (.pkl) | — |
| `--batch_size` | Sequences per inference batch | `16` |

</details>

**Outputs:**
- `protein_esmc_600m.pt` — embedding tensor `[N_proteins, 1152]`
- `protein_esmc_600m.pkl` — index mapping `{protein_id -> tensor_index}`

---

### Contrastive Alignment

Trains two lightweight MLP projection heads that align PPI graph embeddings and ESM-C sequence embeddings into a shared latent space via symmetric InfoNCE loss. Requires embeddings from the above ESM-C and PPI Graph Transformer.

```bash
cd contrastive/
bash run_contrastive.sh
```

<details>
<summary>Full arguments:</summary>

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

</details>

**Outputs** (written to `--save_dir`):
- `clip_mlps.ckpt` — projection head checkpoint (weights + config)
- `ppi_projected.pth` — projected PPI embeddings per cluster `{ct_idx -> tensor[n, 512]}`
- `esm_projected.pth` — projected ESM embeddings per cluster `{ct_idx -> tensor[n, 512]}`
- `joint_projected.pth` — concatenated joint embeddings per cluster `{ct_idx -> tensor[n, 1024]}`
- `esm_proj_embeddings.pt` — flat projected ESM tensor `[N_proteins, 512]`
- `esm_proj_ids.json` — ordered protein ID list corresponding to the flat tensor

---

## Experiment Tracking (optional)

Component 1 logs training and validation metrics to [Weights & Biases](https://wandb.ai). Tracking is optional.

**To enable it**, set your W&B entity (team or username) and log in once:

```bash
export WANDB_ENTITY=your-entity   # omit to use your account's default entity
wandb login                       # first run only
bash run_ppi_gt.sh
```

**To run without W&B** — no account or login required — disable it with the standard environment variable:

```bash
export WANDB_MODE=disabled
bash run_ppi_gt.sh
```

---

## Using the released weights

The weights on HuggingFace reproduce the trained PPI graph-transformer and
contrastive projection heads from the paper. They are specific to the CNS atlas
SEMPER was trained on, so they are intended
for reproducing the published model (e.g., once the source atlas is available),
**not** as a general-purpose protein embedder. To apply SEMPER to a different
organism, tissue, or cell-type atlas, retrain on your own cell type-specific PPI
networks (see *Training*).

---

## Hardware

SEMPER was trained on the [Kempner Institute](https://kempner.harvard.edu) computing cluster.

| Component | Hardware | Notes |
|---|---|---|
| PPI-GT encoder | 4× NVIDIA H100 (80GB) | Distributed Data Parallel (DDP) |
| ESM-C inference | 1× NVIDIA H100 (80GB) | Frozen inference only |
| Contrastive alignment | 1× NVIDIA H100 (80GB) | Projection heads only |

---

## Code and Data Availability

- **Code** — this repository, released under the MIT License.
- **Model weights** — available on [HuggingFace](https://huggingface.co/mims-harvard/SEMPER) (atlas-specific; see *Using the released weights*).
- **Training data** — the adult human CNS single-cell atlas, the 442 cell type-specific PPI networks, and the cellular/tissue metagraph are not publicly available. They are collaborator-controlled under a data use agreement and are available from [collaborator / contact] on reasonable request.
- **Derived embeddings** — not released, as they are derived one-to-one from and would reconstitute the restricted atlas.
- **Project website** — https://zitniklab.hms.harvard.edu/SEMPER/
  
---

## Citation

If you use SEMPER in your research, please cite:

```bibtex
@article{semper2026,
  title   = {SEMPER: Multimodal protein representations for neurological target assessment},
  author = {Dong, Amy and Li, Michelle M. and Ding, Kerr and Nazeen, Sumaiya and Noori, Ayush and Huang, Yepeng and Ritter, Dylan and Afroz, Jalwa and Henderson, Alexander B. H. and Studer, Lorenz and Luo, Yunan and Khurana, Vikram and Parmigiani, Giovanni and Zitnik, Marinka},
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
