# SEMPER — Evaluation

This directory contains all scripts to reproduce the analyses in the SEMPER paper,
organized by figure. Each subfolder has its own README with detailed usage instructions.

These scripts take SEMPER representations of the 442 brain cell types and the cell type-specific PPI networks as input. Neither is released, so the scripts can be run on representations trained with the code in [`semper/`](../semper). Zero-shot target retrieval can also be run end to end on the released targets (see the [tutorial](../tutorial)).

---

## Structure

```
evaluate/
├── fig1e_plm_backbones/         # Figure 1e — SEMPER vs. its protein language model backbone
├── fig2_embedding_space/        # Figure 2  — SEMPER embedding space analyses
├── fig3_protein_complexes/      # Figure 3  — Zero-shot protein complex identification
├── fig4_target_prioritization/  # Figure 4  — Zero-shot therapeutic target retrieval
└── fig56_llm_evaluator/         # Figures 5–6 — LLM-augmented SEMPER pipelines
```

---

## Figure-by-figure overview

| Folder | Figure | Key task |
|---|---|---|
| `fig1e_plm_backbones/` | 1e | Target retrieval with six protein language model backbones, alone and inside SEMPER |
| `fig2_embedding_space/` | 2 | Pairwise similarity, GO dispersion, UMAP + Leiden, Laman Trip correlation (plus an extra EMF analysis not in the paper) |
| `fig3_protein_complexes/` | 3 | Zero-shot CORUM complex identification across four complex definitions |
| `fig4_target_prioritization/` | 4 | Hits@K target retrieval across five neurodegenerative diseases |
| `fig56_llm_evaluator/` | 5–6 | LLM-only baseline, pointwise LLM reranking, and the Elo tournament with Gene Ontology annotations from Open Targets |
