# `scripts/revision/` — revision-era analytical pipelines

This folder contains the Colab notebook-derived Python scripts for the four analytical streams (A, B, C, E, G) added during the major revision cycle (v17 → final), plus figure generation and validation extension.

Each script is a stand-alone Colab-compatible pipeline. Set the Hugging Face token as an environment variable (`HF_TOKEN`) before running.

## Contents

| Script | Purpose | Outputs | Runtime |
|---|---|---|---|
| `stream_a_matched_comparisons.py` | Adds BERT-base-uncased and BioMistral-7B (matched controls) to the 11-model panel; per-layer extraction, geometry, retrieval metrics, and interventions | `results/BERT-base-uncased.parquet`, `results/BioMistral-7B.parquet`, `interventions/interventions_new_models.parquet` | ~75–105 min (H100) |
| `stream_b_zca_methodology.py` | ZCA whitening methodology comparison (transductive vs corpus-only vs baseline), ε sensitivity sweep, and 5-fold cross-validation for all 13 configurations | `stream_b/results/methodology_comparison.parquet`, `epsilon_sensitivity.parquet`, `cross_validation.parquet` | ~80–105 min |
| `stream_c_per_query_ranks_lme.py` | Per-query per-layer rank extraction for LME input, plus first-pass LME fitting | `stream_c/rank_data/per_query_ranks.parquet`, `stream_c/lme_results/*.parquet` | ~60–90 min |
| `stream_eg_editorial_analyses.py` | Combined editorial-response analyses: lexical overlap audit, synthetic corpus descriptive audit, MedCPT expanded evaluation, geometry across 11 configurations, E5-Mistral intervention sweep, chunking sensitivity | `stream_eg/stream_e/*.parquet`, `stream_eg/stream_g/*.parquet` | ~90–120 min |
| `revision_figures.py` | Generates all 10 manuscript figures from cached parquet outputs of Streams A–G | `manuscript_figures/Figure1-10.png` + `figure_captions.md` | ~5 min (CPU) |
| `colab_validation_extension.py` | Extended validation on 500 PMC-Patients and 400 MTSamples for the 5 models not previously validated at 4–5× scale | `{Model}_validation500_layers.parquet`, `validation_summary_5models.parquet` | ~90–120 min |
| `prepare_500doc_corpora.py` | Prepares the 500-doc PMC-Patients and MTSamples corpora with GPT-4o query generation for the validation extension | `pmc500_docs.parquet`, `mtsamples500_docs.parquet`, `pmc500_queries.parquet`, `mtsamples500_queries.parquet` | ~30 min |

Plus the previously-deposited LME correction scripts:
- `v1_final_refit.py` — Corrected additive-model LME re-fit (produces `_CORRECTED.parquet` outputs)
- `v1_run_now.py` — Positional-vs-corpus-unique grouping diagnostic

## Requirements

- Google Colab Pro+ with NVIDIA H100 80GB (initial extraction) or NVIDIA RTX PRO 6000 Blackwell 102 GB (revision compute)
- Python 3.11, PyTorch 2.3, transformers 4.44, statsmodels 0.14, scipy 1.12, numpy 1.26, rank-bm25 0.2, huggingface_hub 0.24
- Hugging Face access token in `HF_TOKEN` environment variable (needed for gated models: BioMistral-7B, E5-Mistral-7B, some Phi-3 variants)

## Input data locations

Each script expects inputs at `/content/sample_data/` unless otherwise noted:
- `mtsamples_sample.csv` — 500+ rows, `text` column
- `metadata_queries.json` — nested `{corpus: {keyword: [], natural_language: []}}` with 100 queries per corpus (per_v17 v17 canonical set)
- `synthetic_notes.csv` — 100+ synthetic clinical notes

PMC-Patients is loaded from HuggingFace `zhengyun21/PMC-Patients` at runtime with `seed=42` for the primary 100 and `seed=123` for the 500-doc validation extension.

## Run order

1. `stream_a_matched_comparisons.py` — adds the two matched-control models
2. `stream_b_zca_methodology.py` — ZCA experiments across all 13 configurations
3. `stream_c_per_query_ranks_lme.py` — per-query ranks for LME
4. `v1_final_refit.py` — corrected LME re-fit on `per_query_ranks.parquet`
5. `stream_eg_editorial_analyses.py` — editorial-response analyses
6. `colab_validation_extension.py` (uses `prepare_500doc_corpora.py` first) — 5× scale validation
7. `revision_figures.py` — final figure generation from cached parquets

## LME correction

The v1 refit (`v1_final_refit.py`) applies a corrected corpus-unique random-effects grouping (`corpus_query_idx`) rather than the positional grouping (`query_idx`) that produced a degenerate solution in an earlier fit. See `docs/analysis_decisions.md` §1 for the diagnostic story. All files ending `_CORRECTED.parquet` are authoritative; predecessor files without the suffix should not be used.

## Compute provenance

- Extraction: Google Colab Pro+, NVIDIA H100 80GB and RTX PRO 6000 Blackwell 102 GB
- LME analysis: Python 3.11, statsmodels 0.14, `method='bfgs'`, `maxiter=500`
- Bootstrap CIs: 50 cluster-bootstrap iterations for the primary analysis (see `bootstrap_rerun_additive.py`), 1000 resamples where indicated in individual scripts

## License

CC BY 4.0 (matches root repository LICENSE).
