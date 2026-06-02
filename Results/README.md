# `results/` — analytical artifacts

Each file here is the output of one stage of the analysis pipeline. The files are organised by analytical question, not by script.

## Authoritative analytical outputs (use these)

| File | What it is | Produced by | Used by manuscript section |
|---|---|---|---|
| `all_models_combined.parquet` | Per-condition retrieval and geometry metrics, 11 base configurations × 3 corpora × 2 query formats × layers × terciles | `scripts/paper2_layer_analysis.py` | §4.1 (trajectories), §4.7 (geometry), Table 2 |
| `anova_bert_typeII.parquet` | BERT-scale Type II ANOVA decomposition (13 layers, 8-model subset) | `scripts/paper2_layer_analysis.py` | §4.2 (variance decomposition); Multimedia Appendix 4 |
| `anova_llm_typeII.parquet` | LLM-scale Type II ANOVA decomposition (33 layers, 3-model subset, n=18 per layer caveat) | `scripts/paper2_layer_analysis.py` | §4.2 caveat; Multimedia Appendix 4 |
| `per_model_summary_CORRECTED.parquet` | Per-model LME variance components + ICC, all 13 configurations, corrected corpus-unique grouping (300 groups per model) | `scripts/revision/v1_final_refit.py` | §4.5 (LME), Methods §3.4.3; Multimedia Appendix 5 |
| `per_model_fixed_effects_CORRECTED.parquet` | Per-model LME fixed-effect estimates, SEs, z-statistics, P-values | `scripts/revision/v1_final_refit.py` | §4.5 table, matched contrasts; Multimedia Appendix 5 |
| `epsilon_sensitivity.parquet` | Corpus-only ZCA ΔMRR@10 sweep across ε ∈ {1e-7 … 1e-2} × 13 models × 6 conditions | Stream B (not yet in repo — see `scripts/revision/README.md`) | §4.10 (ε sensitivity); Multimedia Appendix 8 |
| `cross_validation.parquet` | 5-fold cross-validated corpus-only ZCA ΔMRR@10 per model, fit on 80 docs, evaluated on held-out 20 | Stream B (not yet in repo) | §4.11, Figure 4, Table 3 |
| `methodology_comparison.parquet` | Corpus-only vs transductive vs no-correction baseline ZCA across full 13 models × 6 conditions | Stream B (not yet in repo) | §3.5, Multimedia Appendix 7 |
| `lexical_overlap.parquet` | BM25-rank vs embedding-rank Spearman correlations per model × corpus × query format | Stream E (not yet in repo) | §4.8 (lexical overlap audit) |
| `geometry_full.parquet` | Full final-layer geometry diagnostics (PR, anisotropy SVD, average pairwise cosine) for all 13 configurations | Stream G (not yet in repo) | Table 2, Figures 2-3 |

## Superseded — DO NOT USE

| File | Why superseded | Replaced by |
|---|---|---|
| `per_model_summary.parquet` (without `_CORRECTED`) | Degenerate LME fit — zero variance component, non-finite log-likelihood, ICC=0 across all models. Inappropriate random-effects grouping (positional `query_idx` 0–99 reused across three corpora). | `per_model_summary_CORRECTED.parquet` |
| `per_model_fixed_effects.parquet` (without `_CORRECTED`) | Produced from the same degenerate fit. Coefficients are numerically close to the corrected values, but the diagnostic statistics (SEs, P-values) reflect the broken random-effects structure. | `per_model_fixed_effects_CORRECTED.parquet` |

These superseded files are retained only for historical traceability and should not be used for any inference. See `docs/analysis_decisions.md` for the diagnostic story.

## Compute provenance

- Embedding extraction: Google Colab Pro+, NVIDIA H100 80GB (some later analyses on RTX PRO 6000 Blackwell 102 GB)
- Analytical pipeline: Python 3.11, PyTorch 2.3, transformers 4.44, statsmodels 0.14, scipy 1.12, numpy 1.26
- LME fits: `method='bfgs'`, `maxiter=500`
- Bootstrap CIs: 1000 resamples
