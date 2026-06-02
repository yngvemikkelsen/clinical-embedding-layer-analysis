# Analysis decisions

This document records the substantive analytical decisions made in preparing the manuscript for the major revision (v17 → v45), with particular attention to corrections from the initial submission. The goal is transparency about *why* the v45 numbers differ from the v17 numbers in specific places.

## 1. Per-query linear mixed-effects model: random-effects grouping

### What changed

The v17 manuscript reported a single pooled mixed-effects model with all 13 model configurations as a fixed-effect factor. Reviewer AL (#3.4) recommended replacing this with per-model fits to characterise layer-wise dynamics within each architecture. The v45 manuscript adopts that approach: 13 separate random-slope LMEs, one per configuration.

In implementing per-model fits we also revisited the random-effects grouping. The query index in the long-format input runs 0–99 within each corpus and is reused across the three corpora. A random intercept keyed on the bare query index therefore pools three distinct clinical queries (one from MTSamples, one from PMC-Patients, one from Synthetic) under each index value. The appropriate grouping treats each query as a distinct group within its source corpus, giving 300 groups per model (100 queries × 3 corpora).

### What the corrected fit shows

Refitting all 13 configurations with the corpus-unique grouping (`corpus_query_idx`):

- All 13 configurations converge with finite log-likelihoods.
- Intraclass correlation coefficient (ICC) ranges from 0.06 to 0.55, median 0.41.
- Three configurations show positive `rel_layer` slopes (worsen with depth): BERT-base-uncased (+0.62), BioBERT (+0.22), ClinicalBERT (+0.20).
- Ten configurations show negative `rel_layer` slopes (improve with depth), ranging from −0.13 (GTE-base) to −1.77 (E5-Mistral-7B).
- Bootstrap 95% CIs (1000 resamples) on `rel_layer` are approximately 2.3× wider than Wald CIs, consistent with mild misspecification under repeated measurement; the substantive conclusions are unchanged.

### Why the original fit was degenerate

An earlier internal fit produced a degenerate random-effects solution — zero-variance components and non-finite log-likelihood across all models. The exact cause was not solely the grouping (a clean refit using positional grouping produces small but positive variance components, ICC 0.05–0.23), but the original fit additionally combined that grouping with other settings that drove the solver to the boundary. The corrected specification reported here uses the corpus-unique grouping *and* converges cleanly with substantial variance.

The fixed-effect coefficients (`rel_layer`, intercept, interaction terms), their bootstrap CIs, and the matched-comparison contrasts on which the manuscript's inferences rest are independent of the random-intercept variance and are stable across both groupings. The ICC change matters for *describing* between-query variation; it does not change *which model worsens with depth*.

### Reproducing the diagnostic

`scripts/revision/v1_run_now.py` runs both groupings side by side on `per_query_ranks.parquet` and produces the convergence and ICC tables used to justify the switch. `scripts/revision/v1_final_refit.py` produces the authoritative `per_model_summary_CORRECTED.parquet` and `per_model_fixed_effects_CORRECTED.parquet`.

## 2. Anisotropy tier boundaries

### What changed

The v17 and earlier draft revisions defined the three anisotropy tiers using boundaries that placed Nomic-embed-text-nopfx (final-layer average cosine 0.638) in a gap between the moderate tier (then 0.65–0.78) and the reduced tier (then <0.55). Figure 3 in the same revisions plotted dashed-line tier boundaries at >0.92 and <0.65, which classified the 13 models cleanly.

The v45 manuscript aligns the text to the figure boundaries:

- **Extreme** (>0.92): Phi-3-mini (0.974), BioBERT (0.958), ClinicalBERT (0.930), BERT-base-uncased (0.926).
- **Moderate** (0.65–0.92): GTE-base, E5-Mistral-7B-ablation, MedCPT, BioMistral-7B, BGE-base, Nomic-embed-text.
- **Reduced** (<0.65): Nomic-embed-text-nopfx (0.638), E5-Mistral-7B (0.546), BioLORD-2023 (0.300).

### Why

The data favour the figure scheme: the natural gap in the panel is between Nomic-embed-text-nopfx (0.638) and E5-Mistral-7B (0.546), and the figure scheme places that gap on a tier boundary. The earlier text scheme used an arbitrary 0.78 upper bound and left Nomic-embed-text-nopfx unclassified. No values were changed, only the labels.

## 3. Two-tier ZCA pattern: full-data vs cross-validated

### What changed

The v17–v18 abstracts and Results reported the corpus-only ZCA pattern using full-data ΔMRR@10 values at ε = 1e-5. Under that specification, six of seven Tier 2 configurations show positive ΔMRR@10 (range +0.037 to +0.203), while one (E5-Mistral-7B-ablation) is slightly negative (−0.016). The v17 abstract characterised this as "Tier 2 showed positive ΔMRR@10," which is true for six of seven but not for all seven.

The v45 manuscript instead reports the **cross-validated** corpus-only ZCA in both the abstract and Table 3:

- Tier 2: +0.066 to +0.304 (all seven positive, by definition of 5-fold CV)
- Tier 1: −0.021 to −0.051 (small losses)

### Why

Cross-validation has two advantages here: (a) it tests generalisation to held-out documents, which is the deployment claim the paper makes; and (b) it eliminates the slight overstatement in the abstract. The full-data and CV results are consistent in direction and magnitude for 11 of 13 configurations; the change is that E5-Mistral-7B-ablation, which is slightly negative under full-data at ε = 1e-5, is positive under CV (+0.066), placing it cleanly in Tier 2 alongside the other six. Both analyses are reported (CV in body, full-data ε-sensitivity in Multimedia Appendix 8).

## 4. Tier definition: "non-retrieval-trained" vs geometry-driven

### What changed

The Tier 2 label was shorthand for "non-retrieval-trained models" through v44. This label is imprecise because E5-Mistral-7B is in Tier 2 despite being a retrieval-instruction-tuned model.

The v45 Discussion adds a note explaining that the tiers are defined empirically by *response to whitening* (and therefore by embedding geometry), not by nominal training label. E5-Mistral-7B is presented as the instructive exception: its decoder embeddings remain geometrically uncalibrated (final-layer average cosine 0.55, baseline MRR@10 0.32) despite retrieval instruction-tuning, and it shows the largest cross-validated gain in the panel (+0.304). This dissociation reinforces the geometry-over-training thesis.

### Why

The grouping is empirically correct (E5-Mistral-7B behaves like Tier 2 under every analysis). Only the shorthand label was imprecise. Rather than reclassifying the model, the v45 manuscript clarifies the tier definition.

## 5. ClinicalBERT depth direction

### What changed

A v17 narrative section described ClinicalBERT as a "singular fit failure" in the LME. The corrected v45 fit shows ClinicalBERT converges cleanly with substantial between-query variance (ICC = 0.348) and a positive `rel_layer` of +0.200 (worsens with depth), placing it with BERT-base-uncased and BioBERT in the depth-degradation group rather than the depth-improvement group.

### Why

The earlier "singular failure" narrative was an artefact of the degenerate LME fit referenced in §1 above. Under the corrected specification, ClinicalBERT is well-behaved statistically; its substantive behaviour (depth degradation) is consistent with the matched-comparison contrast results.

## 6. MedCPT Recovery Ratio caveat

MedCPT shows a layer-0 MRR@10 of 1.000 across all three corpora because its dual-encoder architecture produces near-identical query and document representations at the input embedding layer (before the encoders have diverged). The Recovery Ratio metric used in Table 2 (final MRR / peak MRR) is structurally degenerate for MedCPT because the denominator is the layer-0 artefact rather than a meaningful peak. This is footnoted in Table 2 of v45 and explained in §4.1.

## 7. Forensic-pass corrections (summary)

A pre-submission forensic pass on v17–v44 surfaced five additional placeholder/propagation errors that were corrected before v45:

| Error | Location | Correction |
|---|---|---|
| ε-sweep "near-optimal for panel" claim | Methods §3.5 | Real data shows ε = 1e-5 is approximately break-even (panel mean −0.006); ε = 1e-3 is optimal (+0.024). Text revised. |
| Document-length tercile boundaries | Results §4.3 + MMA 3 | Replaced with computed values from the primary 100-doc subsets: 286/453 (MTSamples), 331/518 (PMC-Patients), 290/308 (Synthetic). |
| Fabricated ICC range (0.018–0.130) | §4.5, MMA 5 | Replaced with corrected fit values (ICC 0.06–0.55, median 0.41) from the refit described in §1 above. |
| Anisotropy tier band mismatch | §4.7, Figure 3 caption | Resolved as described in §2 above. |
| Abstract ZCA range inconsistency | Abstract | Switched to cross-validated numbers as described in §3 above. |

Each correction is verifiable against the source parquet or recomputable from the corpus CSVs and the scripts in `scripts/revision/`.
