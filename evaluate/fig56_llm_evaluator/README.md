# Figures 5 & 6 — LLM-augmented SEMPER for target prioritization

This directory contains scripts for the two LLM evaluator pipelines introduced in
Figures 5 and 6. Both pipelines take SEMPER's top-100 candidate proteins as input
and rerank them using an LLM evaluator, but differ in how the LLM scores candidates.

Both scripts call the OpenAI API (or any Azure OpenAI-compatible endpoint). The figure 5-6 LLM evaluator needs two additional packages that are not part of
the core environment:

```bash
pip install openai tooluniverse
```

Set your API key before running:

```bash
export OPENAI_API_KEY="your-key-here"
```

---

## Figure 5 — Pointwise LLM evaluator (`llm_pointwise_evaluator.py`)

**Pipeline (panel c in Figure 5):**

1. SEMPER retrieves the top-100 candidate proteins for a given cell type and seed
   gene set (run via `fig4_target_prioritization/` scripts)
2. The LLM evaluator scores each candidate individually on a 1–5 biological
   relevance scale and generates a rationale
3. The final ranking combines the normalized LLM score with SEMPER's embedding
   similarity score (equal weight: `0.5 × similarity + 0.5 × (score−1)/4`)

**Input:** A prompts JSON file (`generated_*_messages_*.json`) where each entry is
a pre-formatted message payload containing the cell type context, seed genes, and
the SEMPER top-100 candidate list. One entry per cell type per disease.

```bash
python llm_pointwise_evaluator.py \
    --input_json  generated_DA_messages_cluster395_zeroshotprompt.json \
    --output_dir  outputs/llm_judge/ \
    --model       o3-mini \
    --azure_endpoint https://api.openai.com \
    --api_version    2025-03-01-preview
```

| Argument | Description | Default |
|---|---|---|
| `--input_json` | Pre-generated prompts JSON (one entry per cell type) | required |
| `--output_dir` | Directory for per-cell-type reranked JSON outputs | required |
| `--model` | Model for both tool call and judge steps | `o3-mini` |
| `--azure_endpoint` | OpenAI or Azure endpoint | `https://api.openai.com` |
| `--api_version` | API version string | `2025-03-01-preview` |

**Output:** One JSON per cell type in `--output_dir`, each containing the reranked
protein list with fields: `protein`, `similarity`, `rationale_score`, `combined_score`,
`rationale`.

---

## Figure 6 — Tool-augmented Elo tournament (`llm_elo_tournament.py`)

**Pipeline (panel b in Figure 6):**

1. SEMPER's top-100 candidates are seeded with initial Elo ratings (1100–1500)
   proportional to their SEMPER similarity scores
2. Candidates enter a multi-round Swiss-system tournament: adjacently ranked
   candidates are paired head-to-head each round
3. For each match, an LLM evaluator fetches live Gene Ontology annotations via
   tool use (OpenTargets MCP), evaluates the pair, determines a winner, and
   generates a biological rationale
4. Elo ratings are updated dynamically after each match; the final ranking is the
   post-tournament Elo order

```bash
python llm_elo_tournament.py \
    --input_json    generated_pd_messages_celltypecluster_zeroshotprompt_10prompt.json \
    --cache_dir     outputs/semper_top100/ \
    --output_dir    outputs/elo_tournament/PD/ \
    --disease       "Parkinson's Disease" \
    --ensp_to_ensg  data/ensp_to_ensg.json \
    --ensp_to_pref  data/ensp_to_gene.json \
    --azure_endpoint https://api.openai.com \
    --api_version    2025-04-01-preview
```

| Argument | Description | Default |
|---|---|---|
| `--input_json` | Pre-generated prompts JSON | required |
| `--cache_dir` | Directory containing SEMPER top-100 JSON files (one per cell type) | required |
| `--output_dir` | Directory for tournament result JSONs | required |
| `--disease` | Disease name string used in LLM prompts | required |
| `--ensp_to_ensg` | Path to `ensp_to_ensg.json` (ENSP → Ensembl gene ID) | required |
| `--ensp_to_pref` | Path to `ensp_to_gene.json` (ENSP → HGNC symbol) | required |
| `--azure_endpoint` | OpenAI or Azure endpoint | `https://api.openai.com` |
| `--api_version` | API version string | `2025-04-01-preview` |
| `--force_rerun` | Overwrite existing outputs | flag |

**Tournament configuration** (constants at top of script):

| Parameter | Value used in paper |
|---|---|
| `BATCH_SIZE` | 3 matches per LLM call |
| `TOURNAMENT_ROUNDS` | 5 rounds |
| `K_FACTOR` | 32 |
| `ELO_SEED_MIN / MAX` | 1100 / 1500 |
| `TOP_N_PARTICIPANTS` | 100 |
| `MODEL_NAME` | `gpt-4.1` |

**Output:** One JSON per cell type in `--output_dir` with the final Elo-ranked
protein list, including fields: `protein`, `similarity`, `elo`, `start_elo`,
`matches_played`.

**Tool dependency:** `llm_elo_tournament.py` requires the `tooluniverse` package
and its OpenTargets MCP tool for live Gene Ontology lookups:

```bash
pip install tooluniverse
```
