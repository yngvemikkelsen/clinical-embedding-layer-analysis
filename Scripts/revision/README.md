# `scripts/revision/` — revision-era analytical scripts

This folder contains scripts added during the major revision (v17 → v45) that are not in the original three-script pipeline (`scripts/paper2_layer_analysis.py`, `paper2_supplementary.py`, `paper2_figures.py`).

## Verified scripts in this folder

### `v1_final_refit.py`
The corrected per-query LME refit. Produces:
- `../../results/per_model_summary_CORRECTED.parquet`
- `../../results/per_model_fixed_effects_CORRECTED.parquet`

Run after loading `per_query_ranks.parquet` (Stream C output). Takes 5–10 minutes on CPU for all 13 models. See `../../docs/analysis_decisions.md` §1 for the substantive story.

### `v1_run_now.py`
The diagnostic that motivated the corrected grouping in `v1_final_refit.py`. Fits both positional and corpus-unique grouping side by side on the same data so the reader can verify the rationale. Useful for reviewers or future work that wants to audit the LME specification choice.

## Stream scripts NOT yet in this folder

The major revision produced four additional analysis streams (B, C, E, G) whose scripts live in your Colab notebooks rather than the repo. The outputs of those streams (parquets) will be deposited in `../../results/` and `../../supplementary/` once consolidated. The pipeline scripts themselves are:

| Stream | Pending scripts | Outputs (in `results/` or `supplementary/`) | Manuscript usage |
|---|---|---|---|
| A (matched comparisons) | `stream_a_matched_comparisons.py` — adds BERT-base-uncased and BioMistral-7B to the panel | Per-model results for 2 added configurations | Methods (Models table); Results matched-comparison contrasts |
| B (ZCA methodology) | `stream_b_methodology_comparison.py`, `stream_b_epsilon_sensitivity.py`, `stream_b_cross_validation.py` | `methodology_comparison.parquet`, `epsilon_sensitivity.parquet`, `cross_validation.parquet` | §3.5, §4.10, §4.11, Figure 4, Table 3, MMA 7, 8 |
| C (per-query LME inputs) | `stream_c_per_query_ranks.py` — generates long-format input for `v1_final_refit.py` | `per_query_ranks.parquet` (437,400 rows) | Source data for §4.5 LME |
| E (editorial: lexical, synthetic, chunking) | `stream_e_lexical_overlap.py`, `stream_e_synthetic_audit.py`, `stream_e_chunking_sensitivity.py` | `lexical_overlap.parquet`, `synthetic_distributional_audit.parquet`, `chunking_sensitivity.parquet` | §4.8, MMA 2, MMA 10 |
| G (editorial: geometry, MedCPT, E5) | `stream_g_geometry_full.py`, `stream_g_medcpt_expanded.py`, `stream_g_e5_ablation_sweep.py`, `stream_g_anisotropy_tiers.py` | `geometry_full.parquet`, `medcpt_expanded.parquet`, `e5_mistral_ablation_sweep.parquet`, `anisotropy_three_tier.parquet` | Table 2, Figure 2, Figure 3, §4.10, MMA 9 |

These scripts will be added in a follow-up commit when the source notebooks are exported. Until then, the outputs in `results/` and `supplementary/` are the authoritative artifacts, and the manuscript Multimedia Appendices document what each contains.

## Reconstruction roadmap

For someone wishing to reproduce a Stream B/C/E/G output before the pipeline scripts are added:

1. The output parquets in `results/` and `supplementary/` are deposited and complete.
2. The Methods sections (§3.4 through §3.6) describe each analysis with enough specificity to re-implement (formulae, parameter values, sample sizes, model spec).
3. The Multimedia Appendices contain the tabulated source values.
4. Open an issue if you need the source script for a specific output before the formal release.

## Compute provenance

- Google Colab Pro+ with NVIDIA H100 80GB (initial extraction) and NVIDIA RTX PRO 6000 Blackwell 102 GB (revision compute)
- Python 3.11, PyTorch 2.3, transformers 4.44, statsmodels 0.14
- All BFGS / L-BFGS LME fits used `maxiter=500`
- Bootstrap CIs: 1000 resamples (parametric Wald CIs reported alongside)
