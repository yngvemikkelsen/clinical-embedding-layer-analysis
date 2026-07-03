# Layer-Level Analysis of Embedding Degradation in Clinical Document Retrieval: Effects of Model Choice, Corpus Context, and Post-Hoc Correction

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.20412005-blue)](https://doi.org/10.5281/zenodo.20412005)

Code, data, and results for the paper:

> Mikkelsen Y. Layer-Level Analysis of Embedding Degradation in Clinical Document Retrieval: Effects of Model Choice, Corpus Context, and Post-Hoc Correction. *JMIR Medical Informatics* 2026 (in revision, MS#99639).

## Status

Manuscript is currently in major revision at *JMIR Medical Informatics*. This repository corresponds to the **revised submission** dated 28 May 2026. Major changes from the initial submission include a corrected per-query mixed-effects analysis and updated anisotropy tier boundaries — see [`docs/analysis_decisions.md`](docs/analysis_decisions.md) for the full record.

## Overview

Layer-wise embeddings were extracted from 11 transformer models (13 configurations including ablations) across 3 clinical corpora to characterize where embedding degradation occurs across depth, separate training-objective from training-domain effects, and quantify whether post-hoc geometric correction can recover retrieval performance without retraining.

**Key findings:**
- Layer-wise retrieval trajectories vary by category; three non-retrieval-trained encoders (BERT-base-uncased, BioBERT, ClinicalBERT) worsen with depth and the remaining 10 configurations improve.
- Final-layer document-embedding geometry partitions the panel into three anisotropy tiers (extreme >0.92, moderate 0.65–0.92, reduced <0.65).
- Corpus-only ZCA whitening shows a two-tier pattern under 5-fold cross-validation: Tier 2 (geometrically uncalibrated) gains ΔMRR@10 +0.066 to +0.304; Tier 1 (retrieval-calibrated) incurs small losses (−0.021 to −0.051).
- The per-query random-slope LME converged for all 13 configurations with substantial between-query variance (ICC 0.06–0.55, median 0.41).
- Model rankings replicate at LLM-scale validation (PMC-500 ρ=0.952, MTSamples-400 ρ=0.929; cross-corpus ρ=0.976 within the retrieval-trained subgroup).

## Repository structure

```
├── README.md
├── LICENSE                              # CC BY 4.0
├── requirements.txt
│
├── scripts/
│   ├── paper2_layer_analysis.py         # Main extraction + metrics + ZCA (initial pipeline)
│   ├── paper2_supplementary.py          # Whitening variants + validation corpora
│   ├── paper2_figures.py                # Figure generation (initial)
│   └── revision/                        # Revision-era additions (v17 → v45)
│       ├── README.md                    # Status and pending items
│       ├── v1_final_refit.py            # Corrected per-query LME
│       ├── v1_run_now.py                # Grouping diagnostic
│       └── chunking_sensitivity_corrected.py   # Token-level chunking re-run
│
├── results/                             # Analytical outputs
│   ├── README.md                        # File-by-file provenance + supersession notice
│   ├── all_models_combined.parquet
│   ├── anova_bert_typeII.parquet
│   ├── anova_llm_typeII.parquet
│   ├── per_model_summary_CORRECTED.parquet
│   ├── per_model_fixed_effects_CORRECTED.parquet
│   └── chunking_sensitivity_CORRECTED.parquet
│
├── supplementary/                       # Supporting outputs
│   ├── whitening_variants.parquet
│   ├── validation_500_combined.parquet
│   ├── validation_mt400_combined.parquet
│   └── layer_selection.parquet
│
├── figures/                             # Manuscript figures (4 in v45)
│   ├── Figure_1_layer_trajectories.png
│   ├── Figure_2_geometry_diagnostics.png
│   ├── Figure_3_anisotropy_tiers.png
│   └── Figure_4_cross_validated_zca.png
│
└── docs/
    ├── analysis_decisions.md            # Corrected LME, tier reconciliation, etc.
    └── reproducibility.md               # Manuscript claim → source file mapping
```

## Models

11 base models with full layer-wise extraction, plus 2 ablation configurations, give **13 configurations** in the analysis panel.

| Model | HuggingFace ID | Category | Hidden layers |
|---|---|---|---|
| BERT-base-uncased | google-bert/bert-base-uncased | General encoder | 12 |
| BioBERT | dmis-lab/biobert-v1.1 | Domain encoder | 12 |
| ClinicalBERT | medicalai/ClinicalBERT | Domain encoder | 6 |
| BGE-base | BAAI/bge-base-en-v1.5 | General embedding | 12 |
| GTE-base | thenlper/gte-base | General embedding | 12 |
| Nomic-embed-text | nomic-ai/nomic-embed-text-v1.5 | General embedding | 12 |
| BioLORD-2023 | FremyCompany/BioLORD-2023 | Biomedical retriever | 12 |
| MedCPT | ncbi/MedCPT-Query-Encoder + MedCPT-Article-Encoder | Biomedical retriever | 12 |
| Phi-3-mini | microsoft/Phi-3-mini-4k-instruct | General LLM | 32 |
| BioMistral-7B | BioMistral/BioMistral-7B | Domain LLM | 32 |
| E5-Mistral-7B | intfloat/e5-mistral-7b-instruct | General LLM (retrieval-instruction-tuned) | 32 |

Ablation configurations: E5-Mistral-7B-ablation (mean pooling, no instruction), Nomic-embed-text-nopfx (no task prefixes).

## Corpora

**Primary 100-document working set per corpus** (the unit of analysis throughout):
- **MTSamples**: first 100 rows of `mtsamples_sample.csv`
- **PMC-Patients**: 100 documents sampled from [zhengyun21/PMC-Patients](https://huggingface.co/datasets/zhengyun21/PMC-Patients) with `seed=42`
- **Synthetic**: 100 clinical notes generated by Mistral-7B-Instruct

Document-length tercile boundaries (33.3/66.7 percentiles of word counts) on the primary subsets: 286/453 (MTSamples), 331/518 (PMC-Patients), 290/308 (Synthetic).

**LLM-scale validation corpora:**
- 500 PMC-Patients (seed=123, no overlap with primary 100)
- 400 MTSamples (remaining documents after the primary 100)

**Queries**: 100 keyword + 100 natural-language queries per corpus, generated from document metadata using GPT-4o (temperature 0.3, max_tokens 150).

## Reproducing results

### Requirements

```bash
pip install -r requirements.txt
```

Embedding extraction tested on Google Colab Pro+ with NVIDIA H100 80GB. Analytical scripts run on CPU.

### Three levels of reproduction

**(A) Verify manuscript numbers against published outputs (minutes, no GPU).**
Every manuscript table and figure value is derivable from a file in `results/` or `supplementary/`. See [`docs/reproducibility.md`](docs/reproducibility.md) for the claim-to-file mapping.

**(B) Re-run the corrected LME from cached per-query ranks (~10 min, CPU).**

```bash
python scripts/revision/v1_final_refit.py \
    --input path/to/per_query_ranks.parquet \
    --output-dir results/
```

The diagnostic comparing positional vs corpus-unique grouping:

```bash
python scripts/revision/v1_run_now.py --input path/to/per_query_ranks.parquet
```

**(C) Full pipeline from raw text (~3 hours, GPU).**

```bash
python scripts/paper2_layer_analysis.py
python scripts/paper2_supplementary.py
python scripts/paper2_figures.py
```

Note: revision-era streams (B, C, E, G) used additional Colab pipelines whose source scripts are being prepared for separate deposit. The output parquets are included; see [`scripts/revision/README.md`](scripts/revision/README.md) for status.

## A note on the corrected LME analysis

The per-query LME (`scripts/revision/v1_final_refit.py`) uses a corpus-unique random-effects grouping (`corpus_query_idx`, 300 groups per model = 100 queries × 3 corpora). An earlier internal fit had keyed the random intercept on positional query index (0–99) and additionally produced a degenerate solution (zero variance, non-finite log-likelihood); the corrected specification converges cleanly for all 13 models with median ICC 0.41. The fixed-effect coefficients and bootstrap CIs that the manuscript's inferences rest on are independent of the random-intercept variance and are stable across both groupings. See [`docs/analysis_decisions.md`](docs/analysis_decisions.md) §1 for the full diagnostic story. Only files ending `_CORRECTED.parquet` are authoritative.

## Companion study

> Mikkelsen Y. Clinical context variables collectively rival model choice in embedding-based retrieval: multi-corpus benchmark study. *JMIR Medical Informatics* 2026;14:e94241. doi:[10.2196/94241](https://doi.org/10.2196/94241)

Code and data: https://github.com/yngvemikkelsen/clinical-rag-retrieval-benchmark

## Citation

```bibtex
@article{mikkelsen2026layer,
  title   = {Layer-Level Analysis of Embedding Degradation in Clinical Document Retrieval: Effects of Model Choice, Corpus Context, and Post-Hoc Correction},
  author  = {Mikkelsen, Yngve},
  journal = {JMIR Medical Informatics},
  year    = {2026},
  note    = {In revision, MS\#99639}
}
```

(Citation will be updated with volume/DOI on acceptance.)

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt the material for any purpose, provided you give appropriate attribution.
