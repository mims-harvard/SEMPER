# Figures 5 & 6 — LLM-augmented SEMPER for target prioritization

These scripts reproduce the three arms compared in Figures 5d–j and 6.

* **SEMPER** ranks the proteins of one cell type by cosine similarity to the mean
  SEMPER representation of a set of seed targets.
* **SEMPER+LLM** reranks SEMPER's candidates with an LLM, either pointwise
  (each candidate scored 1–5) or with a pairwise Elo tournament that shows the LLM
  Gene Ontology annotations.
* **LLM-only** asks the LLM for candidates directly, with no SEMPER retrieval.

Every LLM call in these figures used the `gpt-5.6-sol` deployment on Azure OpenAI
with the default sampling temperature.

## Scripts

| Script | Reproduces | What it does |
|---|---|---|
| `semper_retrieval.py` | SEMPER candidate pools for Fig 5f–j and 6 | Full ranked list per prompt (no LLM) |
| `llm_pointwise_context_judge.py` | Fig 5f–j, SEMPER+LLM | Cell type-aware 1–5 scoring of SEMPER's top 100, all in one request |
| `llm_go_tournament.py` | Fig 6, SEMPER+LLM | Cell type-aware Elo tournament over SEMPER's top 100 with Open Targets GO evidence |
| `llm_pointwise_da_judge.py` | Fig 5d–e, SEMPER+LLM | 1–5 scoring of the full SEMPER pool in batches of 100 (dopamine-neuron essential genes) |
| `llm_only_baseline.py` | LLM-only arm of Fig 5d–e, 5f–j and 6 | Two-turn answer, then bare ranked list |
| `make_llmonly_prompts.py` | LLM-only prompts of Fig 5f–j and 6 | Replaces the cluster ID in each SEMPER prompt with a readable cell type description |
| `common.py` | — | Credentials, disease names, cell type metadata |

The SEMPER-only arm of Fig 5d–e and the Hits@K scoring and plotting of all three
figures are not in this folder.

## Setup

The scripts run in the main SEMPER environment (`torch`, `numpy`) plus the OpenAI client.
ToolUniverse is not needed.

```bash
pip install openai
```

Credentials are read from environment variables only. The scripts stop with an
error if none are set.

```bash
# Azure OpenAI (as in the paper)
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com"

# or OpenAI, used only when both Azure variables are unset
export OPENAI_API_KEY="..."
```

`--model` defaults to `gpt-5.6-sol`. Pass your own deployment or model name if it
differs. Each script keeps the API version it was run with (`2025-03-01-preview` for
the context judge and the disease LLM-only arm, `2025-04-01-preview` for the
tournament, the DA judge and the DA LLM-only arm). Override with `--api_version`.

## Inputs

| File | Description |
|---|---|
| `semper_embed.pth`, `semper_labels.txt` | SEMPER representations (t = 0.10), built with [`semper/contrastive/build_semper_embeddings.py`](../../semper/contrastive/build_semper_embeddings.py). Not released |
| `../data/ensp_to_gene.json` | ENSP → HGNC symbol (default for `--ensp_to_gene`) |
| `inputs/ensp_to_ensg.json` | ENSP → Ensembl gene ID, used to query Open Targets |
| `inputs/celltype_metadata_top10.csv` | Siletti et al. annotations (name, class, supercluster, neurotransmitter, top regions, top enriched genes) of the Fig 4 top-10 cell types per disease |
| `inputs/prompts/generated_<d>_messages_celltypecluster_zeroshotprompt_10prompt_t010.json` | SEMPER prompts used in the paper, `<d>` ∈ pd, ad, ms, als, hd. 100 per disease = top-10 cell types × 10 draws of 15 seed targets. Only the seeds and `celltype cluster:N` are read. Embedding paths in the text were replaced by `semper_embed.pth` / `semper_labels.txt` |
| `inputs/prompts/generated_<d>_messages_llmonly_zeroshotprompt_10prompt_t010.json` | LLM-only prompts used in the paper. `make_llmonly_prompts.py` regenerates them byte for byte from the SEMPER prompts |
| `inputs/prompts/generated_prompts_395_top25_DA_llm_DAfull_minus_DAtop100_75.json` | Fig 5d–e prompts, 100 draws of 20 dopamine-neuron essential genes, asking for 75 proteins (the `top25` in the name is historical). Used by the DA judge (seeds only) and the DA LLM-only arm |

### Cell type IDs

Cell types are given as Siletti et al. (2023) cluster IDs, `cluster:395` or `395`.
The integer keys of `semper_embed.pth` are something else. They are the positions
of the `CCI_cluster:N` rows in `semper_labels.txt` (for example key 319 is
`cluster:395`). `semper_retrieval.py` does that lookup, so always pass the cluster ID.

The Fig 5d–e prompts used in the paper wrote the embedding key (`celltype 319`)
where the cluster ID `cluster:395` (Splat_395, the dopaminergic cell type) was meant.
The retrieval read it as a cluster ID, so the published SEMPER+LLM pools for
Fig 5d–e came from `cluster:319`. `llm_pointwise_da_judge.py` takes the cell type
explicitly. Use `--celltype cluster:395` for the intended cell type and
`--celltype cluster:319` to rerun what was published.

## Fig 5f–j and Fig 6 (five neurodegenerative diseases)

```bash
EMB=../data/semper_embed.pth; LAB=../data/semper_labels.txt
for d in pd ad ms als hd; do
  D=$(echo $d | tr a-z A-Z)
  P=inputs/prompts/generated_${d}_messages_celltypecluster_zeroshotprompt_10prompt_t010.json

  # SEMPER (no LLM): full ranked list per prompt
  python semper_retrieval.py --prompts_json $P --embed_path $EMB --label_path $LAB \
      --output_dir outputs/retrieval/$D

  # SEMPER+LLM, Fig 5f-j: cell type-aware pointwise judge (100 requests per disease)
  python llm_pointwise_context_judge.py --input_json $P --cache_dir outputs/retrieval/$D \
      --output_dir outputs/context_judge/$D --disease $d

  # SEMPER+LLM, Fig 6: cell type-aware GO tournament (~85 requests per prompt)
  python llm_go_tournament.py --input_json $P --cache_dir outputs/retrieval/$D \
      --output_dir outputs/go_tournament/$D --disease $d

  # LLM-only (prompts are shipped; make_llmonly_prompts.py rebuilds them)
  python llm_only_baseline.py --preset disease \
      --input_json inputs/prompts/generated_${d}_messages_llmonly_zeroshotprompt_10prompt_t010.json \
      --output_dir outputs/llm_only/$D
done
```

All scripts skip prompts whose output already exists, so an interrupted run can be restarted.

**Context judge.** SEMPER's top 100 for one prompt go to the LLM in a single request,
with all 15 seeds and the cell type description. Each candidate gets a 1–5 score and
a rationale, and candidates are reranked by
`combined_score = 0.5 × similarity + 0.5 × (score − 1) / 4`. Output per prompt is
a JSON list of `protein, similarity, rationale_score, combined_score, rationale`.

**GO tournament.** Settings used in the paper (constants at the top of the script).

| Parameter | Value |
|---|---|
| Participants | SEMPER top 100 |
| Initial Elo | linear in SEMPER similarity, 1100 (lowest) to 1500 (highest) |
| Rounds | 5, Swiss style (sort by Elo with random tie order, pair neighbours) |
| Matches per request | up to 3, JSON response format |
| K factor | 32, Tie = 0.5 |
| GO evidence | Open Targets GraphQL `target.geneOntology`, truncated to 1,500 characters per protein |

GO annotations are fetched with a direct GraphQL query. The ToolUniverse Open Targets
GO tool queries a field that Open Targets has renamed, so it fails on every call.
Outputs per prompt are `tournament_GO_<file>` (final Elo order with `elo`,
`start_elo`, `matches_played`), `GOterms_*` (the GO data shown to the referee) and
`rationales_*` (winner and one-sentence critique for every match).

**LLM-only.** The prompt describes the cell type instead of naming a cluster, and asks
for 25 ranked targets. A second turn asks for the bare list
(`llm_listonly_<CellType>_<seeds>_<idx>.json`), which is the scored prediction.

## Fig 5d–e (dopamine-neuron essential genes)

```bash
DA=inputs/prompts/generated_prompts_395_top25_DA_llm_DAfull_minus_DAtop100_75.json

# SEMPER+LLM: full SEMPER pool scored in batches of 100
python llm_pointwise_da_judge.py --input_json $DA --celltype cluster:395 \
    --embed_path ../data/semper_embed.pth --label_path ../data/semper_labels.txt \
    --output_dir outputs/da_judge/cluster395

# LLM-only: 75 proteins per prompt
python llm_only_baseline.py --preset da --input_json $DA --output_dir outputs/llm_only/DA
```

The DA judge ranks the whole cell type network (minus the seeds) with SEMPER and sends
it to the LLM in batches of 100 in similarity order. As in the paper, each request
lists only the first 5 of the 20 seeds. A batch whose reply cannot be parsed keeps
its proteins with score 1. The combined score is the same as for the context judge.
The script can be split over a Slurm job array (`SLURM_ARRAY_TASK_ID` /
`SLURM_ARRAY_TASK_COUNT`).

## Reproducibility

* LLM calls use the default temperature and are not deterministic, and the
  tournament also shuffles tied ratings at random. Reruns reproduce the published
  results statistically, not exactly. Hosted model versions can also change over time.
* The raw LLM outputs behind Fig 5f–j and Fig 6 (judge scores, tournament results,
  LLM-only answers) and the cached SEMPER retrieval lists are not available, so the
  published numbers cannot be recomputed from saved outputs.
* `semper_retrieval.py` is deterministic and computes the same ranking as the
  ToolUniverse `ContextualProteinRetrieval` tool that produced the paper's candidate
  pools.

## Differences from the scripts that were run

The prompts, model, batching, scoring and Elo logic are unchanged. The changes are

* paths, keys and the endpoint come from arguments and environment variables;
* the SEMPER retrieval is computed in `semper_retrieval.py` instead of through
  ToolUniverse (same computation);
* in the DA judge the original asked the LLM to fill in the retrieval-tool arguments
  from the prompt text. Here the seeds are parsed from the prompt and the cell type
  comes from `--celltype`;
* DA judge output files are named `tool_top_cluster<N>_<seeds>_<idx>_LLMJUDGE.json`
  (originally `tool_75_...`; the 75 was never applied, the full pool was scored);
* debugging output and unused code were removed.
