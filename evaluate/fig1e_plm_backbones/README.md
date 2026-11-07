# Figure 1e — SEMPER across protein language model backbones

This directory reproduces Figure 1e: zero-shot candidate target retrieval
(Hits@10) for each protein language model (PLM) backbone used alone (open
squares) and for SEMPER built on that backbone (filled squares). The six
backbones are

| Label in figure | Checkpoint | Embedding script |
|---|---|---|
| ESM-2 8M   | `esm2_t6_8M_UR50D`    | `generate_plm_embeds.py` (fair-esm) |
| ESM-2 35M  | `esm2_t12_35M_UR50D`  | `generate_plm_embeds.py` (fair-esm) |
| ESM-2 150M | `esm2_t30_150M_UR50D` | `generate_plm_embeds.py` (fair-esm) |
| ESM-3      | `esm3-sm-open-v1` (1.4B) | `generate_plm_embeds.py` (EvolutionaryScale `esm`) |
| ESM-C 300M | `esmc_300m` | [`semper/esm/generate_ESMC_embeds.py`](../../semper/esm/generate_ESMC_embeds.py) `--model esmc_300m` |
| ESM-C 600M | `esmc_600m` | [`semper/esm/generate_ESMC_embeds.py`](../../semper/esm/generate_ESMC_embeds.py) (default) |

All backbones share the same frozen PPI-GT network encoder. For each backbone
only the two projection heads are trained (same hyperparameters as the main
model), and the cell type-specific representations are formed with the same
interpolation, t = 0.10.

Retrieval uses a single pooled target set, the Open Targets
`clinically_relevant_targets` of EFO_0000618 (nervous system disease), with 40
seed targets per trial, 100 trials per cell type, and 441 cell types.

```
fig1e_plm_backbones/
├── generate_plm_embeds.py                  # ESM-2 / ESM-3 mean-pooled embeddings
├── plot_fig1e.py                           # aggregate trials CSVs and draw panel e
└── data/
    ├── neuro_seedplan_EFO_0000618.json.gz  # the seed plan used for the paper (all models, identical trials)
    └── targets_EFO_0000618.json            # Open Targets target lists for EFO_0000618 (--pd_gene_file)
```

---

## Data requirements

The cell type PPI networks and the trained PPI-GT encoder are not released, so
steps 2 and 3 need representations you train yourself with the code in
[`semper/`](../../semper).

```
inputs/
├── protein_sequences.fasta     # semper/esm/fetch_sequences.py
├── ppi_gt/_protein_embed.pth   # semper/ppi_gt/run_ppi_gt.sh (frozen network encoder output)
└── ppi_gt/_labels_dict.txt     # same run; defines the per-cell type candidate proteins
../data/ensp_to_gene.json       # ENSP -> gene symbol (released, evaluate/data/)
```

The seed plan in `data/` fixes the 40 seeds of every trial. It was built with
the cell type networks intersected with the ProTrek protein universe
(`universe_definition` in the file), which `../fig4_target_prioritization/make_seedplan.py`
does not do, so use the shipped plan to score the same trials as the paper. The
evaluation scripts read only `clusters -> trials -> seed_ensp` from it. Held-out
positives are all EFO_0000618 targets in each cell type network.

All commands below are run from this directory.

---

## Step 1 — Sequence embeddings for each backbone

```bash
FASTA=inputs/protein_sequences.fasta
mkdir -p embeds

# ESM-2 (environment with fair-esm: `pip install fair-esm biopython tqdm`;
# fair-esm and the EvolutionaryScale `esm` package both install a module called `esm`)
for M in esm2_t6_8M_UR50D esm2_t12_35M_UR50D esm2_t30_150M_UR50D; do
    python generate_plm_embeds.py --model $M --fasta $FASTA \
        --out_embeds embeds/$M.pt --out_index embeds/$M.pkl
done

# ESM-3 open small (semper-esm environment, environment_esm.yml; the weights are
# gated on Hugging Face, so accept the license and `huggingface-cli login` first)
python generate_plm_embeds.py --model esm3-sm-open-v1 --fasta $FASTA \
    --out_embeds embeds/esm3-sm-open-v1.pt --out_index embeds/esm3-sm-open-v1.pkl

# ESM-C 300M and 600M (semper-esm environment)
for M in esmc_300m esmc_600m; do
    python ../../semper/esm/generate_ESMC_embeds.py --model $M --fasta $FASTA \
        --out_embeds embeds/$M.pt --out_index embeds/$M.pkl
done
```

## Step 2 — SEMPER representations for each backbone

Contrastive alignment and the t = 0.10 interpolation reuse the main model's
scripts unchanged (defaults are the paper hyperparameters: projection 512,
heads 1024-512 and 512-512, batch 64, 200 epochs, lr 1e-4, temperature 0.07,
seed 42). One GPU per backbone.

```bash
for M in esm2_t6_8M_UR50D esm2_t12_35M_UR50D esm2_t30_150M_UR50D esm3-sm-open-v1 esmc_300m esmc_600m; do
    python ../../semper/contrastive/train_contrastive_MLP.py \
        --embed_path inputs/ppi_gt/_protein_embed.pth \
        --label_path inputs/ppi_gt/_labels_dict.txt \
        --esm_pickle embeds/$M.pkl \
        --esm_embeds embeds/$M.pt \
        --save_dir   backbones/$M/contrastive/

    python ../../semper/contrastive/build_semper_embeddings.py \
        --contrastive_dir backbones/$M/contrastive/ \
        --out_dir         backbones/$M/semper/ \
        --t 0.10
done
```

For ESM-C 600M this is the main SEMPER model, so `semper_embed.pth` from
[`semper/contrastive/run_contrastive.sh`](../../semper/contrastive/run_contrastive.sh)
can be used directly.

## Step 3 — Retrieval for each backbone, alone and inside SEMPER

Both scripts are the Figure 4 evaluators in
[`../fig4_target_prioritization/run_evaluations/`](../fig4_target_prioritization/run_evaluations).
As in the paper, the SEMPER runs score every trial, and the backbone-alone runs
use `--skip_if_missing_seed`.

```bash
SEEDPLAN=data/neuro_seedplan_EFO_0000618.json.gz
TARGETS=data/targets_EFO_0000618.json
EVAL=../fig4_target_prioritization/run_evaluations
mkdir -p trials

for M in esm2_t6_8M_UR50D esm2_t12_35M_UR50D esm2_t30_150M_UR50D esm3-sm-open-v1 esmc_300m esmc_600m; do
    # SEMPER on backbone M
    python $EVAL/neuro_eval_semper_ct_network.py \
        --seedplan     $SEEDPLAN \
        --pd_gene_file $TARGETS \
        --ensp_json    ../data/ensp_to_gene.json \
        --embed_path   backbones/$M/semper/semper_embed.pth \
        --label_dict   backbones/$M/semper/semper_labels.txt \
        --model_name   SEMPER_$M \
        --out_prefix   trials/neuro_SEMPER_${M}_t010 \
        --device cuda --k_values 5 10 25

    # Backbone M alone (cosine retrieval on raw embeddings, same candidates per cell type)
    python $EVAL/neuro_eval_esm_pklpt_ct_network.py \
        --seedplan       $SEEDPLAN \
        --pd_gene_file   $TARGETS \
        --ensp_json      ../data/ensp_to_gene.json \
        --label_dict     inputs/ppi_gt/_labels_dict.txt \
        --esm_pickle     embeds/$M.pkl \
        --esm_embeddings embeds/$M.pt \
        --model_name     $M \
        --out_prefix     trials/neuro_${M}__ct_network \
        --device cuda --k_values 5 10 25 \
        --skip_if_missing_seed
done
```

## Step 4 — Plot panel e

```bash
python plot_fig1e.py \
    --pair "ESM-C 600M" trials/neuro_esmc_600m__ct_network__trials.csv           trials/neuro_SEMPER_esmc_600m_t010__trials.csv \
    --pair "ESM-C 300M" trials/neuro_esmc_300m__ct_network__trials.csv           trials/neuro_SEMPER_esmc_300m_t010__trials.csv \
    --pair "ESM-3"      trials/neuro_esm3-sm-open-v1__ct_network__trials.csv     trials/neuro_SEMPER_esm3-sm-open-v1_t010__trials.csv \
    --pair "ESM-2 150M" trials/neuro_esm2_t30_150M_UR50D__ct_network__trials.csv trials/neuro_SEMPER_esm2_t30_150M_UR50D_t010__trials.csv \
    --pair "ESM-2 35M"  trials/neuro_esm2_t12_35M_UR50D__ct_network__trials.csv  trials/neuro_SEMPER_esm2_t12_35M_UR50D_t010__trials.csv \
    --pair "ESM-2 8M"   trials/neuro_esm2_t6_8M_UR50D__ct_network__trials.csv    trials/neuro_SEMPER_esm2_t6_8M_UR50D_t010__trials.csv \
    --k 10 --out_dir figures/
```

This writes `figures/fig1e_k10_mean_of_ct_max.{png,pdf}` and the plotted values
in `figures/fig1e_values_k10_mean_of_ct_max.csv`.

**Aggregation.** The paper panel plots, for each model, the best Hits@10 over
the 100 seed sets in each cell type, averaged over the 441 cell types
(`--agg mean_of_ct_max`, the default). `--agg mean` instead averages over seed
sets first (the per-cell type summary used in Figure 4). `--common_trials`
scores both members of a pair on the (cell type, trial) pairs present in both
CSVs.

### Values in the paper panel

Recomputed with `plot_fig1e.py` from the trials CSVs behind the figure.

| Backbone | PLM alone | SEMPER | PLM alone (`--agg mean`) | SEMPER (`--agg mean`) |
|---|---|---|---|---|
| ESM-C 600M | 3.47 | 5.79 | 1.46 | 2.27 |
| ESM-C 300M | 2.99 | 5.58 | 1.06 | 2.27 |
| ESM-3      | 5.00 | 6.75 | 2.59 | 3.25 |
| ESM-2 150M | 4.79 | 6.65 | 2.47 | 3.23 |
| ESM-2 35M  | 5.22 | 6.74 | 2.64 | 3.21 |
| ESM-2 8M   | 3.16 | 6.15 | 1.37 | 2.84 |

### Notes on the paper run

- Two seed proteins in the plan (ENSP00000477713, ENSP00000478570) have no
  sequence embedding, so with `--skip_if_missing_seed` the backbone-alone runs
  score 33,961 of the 44,100 trials, while SEMPER scores all 44,100 (proteins
  without a sequence embedding get the mean projected embedding inside SEMPER).
  Restricting SEMPER to the same trials (`--common_trials`) lowers its values by
  0.12 to 0.15 and leaves every gap positive.
- The ESM-C 300M open square in the paper was computed from ESM++ small
  embeddings (a Hugging Face port of ESM-C 300M), while SEMPER ESM-C 300M was
  trained on official `esmc_300m` embeddings. The commands above use
  `esmc_300m` for both.
- The ESM-3 open square used an earlier ESM-3 embedding file covering 18,614 of
  the 18,680 proteins (identical embeddings for nearly all shared proteins).
