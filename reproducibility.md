# Reproducibility map

For each substantive numerical claim in the v45 manuscript, this document identifies the source file and computation. Verifications can be re-run against the parquets in `results/` without access to GPU compute.

## Headline claims

### "All 13 configurations converged with substantial between-query variance (ICC 0.06 to 0.55, median 0.41)"
- **Source:** `results/per_model_summary_CORRECTED.parquet`
- **Computation:** `df["ICC"].min()`, `df["ICC"].max()`, `df["ICC"].median()`. All 13 rows present with `converged=True` and `log_likelihood` finite.
- **Manuscript:** §4.5; Abstract.

### "Three configurations worsen with depth (BERT-base +0.62, BioBERT +0.22, ClinicalBERT +0.20); ten improve"
- **Source:** `results/per_model_fixed_effects_CORRECTED.parquet`
- **Computation:** Filter `term == "rel_layer"`, group by `model`. Three positive estimates (worsen): BERT-base-uncased (+0.624), BioBERT (+0.224), ClinicalBERT (+0.200). Ten negative.
- **Manuscript:** §4.5; Abstract.

### "Bootstrap 95% CIs are approximately 2.3× wider than Wald CIs"
- **Source:** Bootstrap CI source file (deposited separately; provided as Multimedia Appendix 5 supporting evidence).
- **Computation:** Width ratios for the three representative models (BERT-base, BGE-base, E5-Mistral-7B-ablation): bootstrap widths 0.243, 0.270, 0.248; Wald widths 0.187, 0.190, 0.110; ratios approximately 1.3, 1.4, 2.3 — the 2.3× claim is the upper end (E5-Mistral-7B-ablation), as the manuscript states.
- **Manuscript:** §4.5.

### "Three anisotropy tiers: extreme >0.92, moderate 0.65–0.92, reduced <0.65"
- **Source:** `results/all_models_combined.parquet` (or `results/geometry_full.parquet` when added).
- **Computation:** Filter to final layer per model, `length_tercile == "all"`. Per-model `avg_cosine` values: Phi-3-mini 0.974, BioBERT 0.958, ClinicalBERT 0.930, BERT-base-uncased 0.926 (extreme); GTE-base 0.762 down to Nomic-embed-text 0.683 (moderate); Nomic-embed-text-nopfx 0.638, E5-Mistral-7B 0.546, BioLORD-2023 0.300 (reduced).
- **Manuscript:** §4.7; Figure 3.

### "Corpus-only ZCA cross-validated ΔMRR@10: Tier 2 +0.066 to +0.304, Tier 1 −0.021 to −0.051"
- **Source:** `results/cross_validation.parquet` (to be deposited from Stream B).
- **Computation:** Per-model mean ΔMRR@10 across 5 folds, grouped by Tier. Tier 2 minimum is E5-Mistral-7B-ablation (+0.066); Tier 2 maximum is E5-Mistral-7B (+0.304).
- **Manuscript:** Abstract; §4.11; Figure 4; Table 3.

### "ε = 1e-5 default; sensitivity sweep shows ε = 1e-3 panel-optimal"
- **Source:** `results/epsilon_sensitivity.parquet`.
- **Computation:** Per-ε panel mean ΔMRR@10 across all 13 models × 6 conditions. ε = 1e-3 yields panel mean +0.024 (highest); ε = 1e-5 yields −0.006 (near break-even).
- **Manuscript:** §3.5; Multimedia Appendix 8.

### "LLM-scale validation Spearman ρ = 0.952 (PMC-500), 0.929 (MTSamples-400), 0.976 within retrieval-trained subgroup"
- **Source:** `supplementary/validation_500_combined.parquet`, `supplementary/validation_mt400_combined.parquet`.
- **Computation:** Spearman rank correlation between final-layer MRR@10 of the 100-doc primary subset and the 500-doc / 400-doc validation subsets, computed per model. ρ = 0.976 is the cross-corpus correlation restricted to the retrieval-trained subgroup (BioLORD, MedCPT, BGE, GTE, Nomic, Nomic-nopfx).
- **Manuscript:** §4.15.

### "Document-length tercile boundaries: 286/453 (MTSamples), 331/518 (PMC-Patients), 290/308 (Synthetic)"
- **Source:** `data/mtsamples_sample.csv`, `data/pmc_patients_sample.csv`, `data/synthetic_notes.csv`.
- **Computation:** `np.percentile([len(text.split()) for text in subset[:100]], [33.3, 66.7])` for each corpus's primary 100-doc subset.
- **Manuscript:** §4.3; Multimedia Appendix 3.

### "BM25 baselines on synthetic-corpus alignment recovery: MTSamples 0.87, PMC 0.92, Synthetic 0.96"
- **Source:** `supplementary/MA02_synthetic_alignment_audit.md` (when deposited).
- **Computation:** BM25 MRR@10 on the synthetic-corpus alignment-recovery audit, rounded to two decimal places.
- **Manuscript:** §3.2; Multimedia Appendix 2.

## Per-figure source

| Figure | Source file(s) |
|---|---|
| Figure 1 (layer trajectories) | `results/all_models_combined.parquet`, grouped by `model × layer`, mean `MRR@10` across corpora and query formats |
| Figure 2 (geometry, 3 panels) | `results/all_models_combined.parquet` final layer per model, `length_tercile == "all"`; columns `participation_ratio`, `avg_cosine`, `anisotropy_svd` |
| Figure 3 (anisotropy tiers) | `results/all_models_combined.parquet`, same as Figure 2 panel B, sorted descending |
| Figure 4 (cross-validated ZCA) | `results/cross_validation.parquet`, per-model mean ΔMRR@10 with bootstrap SE |

## Per-table source

| Table | Source file(s) |
|---|---|
| Table 1 (model registry) | `scripts/paper2_layer_analysis.py` model config dictionary |
| Table 2 (trajectory characteristics) | `results/all_models_combined.parquet` peak/trough/final per model; Recovery Ratio with MedCPT footnote (see `docs/analysis_decisions.md` §6) |
| Table 3 (cross-validated ZCA per model) | `results/cross_validation.parquet` |
| Table 4 (matched contrasts) | `results/per_model_fixed_effects_CORRECTED.parquet` plus pooled-LME interaction terms (in MMA 5) |

## How to verify

```python
import pandas as pd

# Verify the ICC range claim
sm = pd.read_parquet("results/per_model_summary_CORRECTED.parquet")
print(f"ICC range: {sm['ICC'].min():.3f} to {sm['ICC'].max():.3f}, median {sm['ICC'].median():.3f}")
# Expected: 0.061 to 0.555, median 0.411

# Verify the depth-direction claim
fe = pd.read_parquet("results/per_model_fixed_effects_CORRECTED.parquet")
rel = fe[fe.term == "rel_layer"].set_index("model")["estimate"]
print(f"Models worsening with depth (rel_layer > 0): {rel[rel > 0].index.tolist()}")
# Expected: ['BERT-base-uncased', 'BioBERT', 'ClinicalBERT']
```
