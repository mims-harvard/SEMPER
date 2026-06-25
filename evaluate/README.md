# SEMPER — Evaluation

This directory contains all scripts to reproduce the analyses in the SEMPER paper,
organized by figure. Each subfolder has its own README with detailed usage instructions.

These scripts require SEMPER embeddings and cell type-specific PPI networks derived from a controlled-access CNS atlas (see Availability). They are provided for transparency of the evaluation methodology; exact reproduction requires access to that data.

---

## Structure

```
evaluate/
├── fig2_embedding_space/        # Figure 2  — SEMPER embedding space analyses
├── fig3_protein_complexes/      # Figure 3  — Zero-shot protein complex identification
├── fig4_target_prioritization/  # Figure 4  — Zero-shot therapeutic target retrieval
└── fig56_llm_evaluator/         # Figures 5–6 — LLM-augmented SEMPER pipelines
```

---

## Figure-by-figure overview

| Folder | Figure | Key task |
|---|---|---|
| `fig2_embedding_space/` | 2 | Pairwise similarity, GO dispersion, UMAP + Leiden, Laman Trip correlation, EMF dispersion |
| `fig3_protein_complexes/` | 3 | Zero-shot CORUM complex identification across four complex definitions |
| `fig4_target_prioritization/` | 4 | Hits@K target retrieval across five neurodegenerative diseases |
| `fig56_llm_evaluator/` | 5–6 | Pointwise LLM reranking and Elo tournament with tool-augmented GO lookup |
