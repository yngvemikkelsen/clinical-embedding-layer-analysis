"""
Paper 12 Revision — Figure Generation (Step 0)
=================================================
Generates all 10 manuscript figures from cached parquet outputs of Streams A-G.

Inputs (Drive):
  /MyDrive/Content/paper2_layer_analysis/revision/
    ├── results/{BERT-base-uncased,BioMistral-7B}.parquet         (Stream A)
    ├── interventions/interventions_new_models.parquet            (Stream A)
    ├── stream_b/results/{methodology_comparison,
    │                     epsilon_sensitivity,
    │                     cross_validation}.parquet                (Stream B)
    ├── stream_c/rank_data/per_query_ranks.parquet                (Stream C)
    ├── stream_c/lme_results/{refit_bfgs, manual_icc}.parquet     (Stream C)
    └── stream_eg/
        ├── stream_e/{lexical_overlap, synthetic_audit_descriptive,
        │             chunking_sensitivity}.parquet                (Stream E)
        └── stream_g/{medcpt_expanded, geometry_full,
                      e5_ablation_intervention_sweep}.parquet      (Stream G)

Outputs (Drive):
  /MyDrive/Content/paper2_layer_analysis/revision/manuscript_figures/
    Figure1_layer_trajectories.png
    Figure2_participation_ratio.png
    Figure3_lme_coefficients.png
    Figure4_matched_comparisons.png
    Figure5_intervention_comparison.png
    Figure6_pr_vs_mrr.png
    Figure7_anisotropy_13models.png
    Figure8_two_tier_pattern.png
    Figure9_e5_ablation_recovery.png
    Figure10_cross_validation.png
    figure_captions.md
"""

# ==CELL 1: SETUP==
import subprocess, sys

for pkg in ["matplotlib", "seaborn", "pyarrow"]:
    try:
        __import__(pkg.replace("-","_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

REVISION = Path('/content/drive/MyDrive/Content/paper2_layer_analysis/revision')
FIG_DIR = REVISION / 'manuscript_figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Verify inputs
required = [
    REVISION / 'results' / 'BERT-base-uncased.parquet',
    REVISION / 'results' / 'BioMistral-7B.parquet',
    REVISION / 'stream_b' / 'results' / 'methodology_comparison.parquet',
    REVISION / 'stream_b' / 'results' / 'cross_validation.parquet',
    REVISION / 'stream_c' / 'rank_data' / 'per_query_ranks.parquet',
    REVISION / 'stream_c' / 'lme_results' / 'refit_bfgs.parquet',
    REVISION / 'stream_c' / 'lme_results' / 'manual_icc.parquet',
    REVISION / 'stream_eg' / 'stream_g' / 'geometry_full.parquet',
    REVISION / 'stream_eg' / 'stream_g' / 'e5_ablation_intervention_sweep.parquet',
    REVISION / 'stream_eg' / 'stream_g' / 'medcpt_expanded.parquet',
    REVISION / 'stream_eg' / 'stream_e' / 'lexical_overlap.parquet',
    REVISION / 'stream_eg' / 'stream_e' / 'chunking_sensitivity.parquet',
]
for r in required:
    assert r.exists(), f"Missing: {r}"
print("All input files present.\n")


# ==CELL 2: STYLE & COLOR MAPPING==
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 6 categories × distinct colors (colorblind-safe)
CATEGORY_COLOR = {
    'General Encoder':       '#9467bd',  # purple — BERT-base (new)
    'Domain Encoder':        '#d62728',  # red    — BioBERT, ClinicalBERT
    'Biomedical Retriever':  '#ff7f0e',  # orange — BioLORD-2023, MedCPT
    'General Embedding':     '#2ca02c',  # green  — BGE, GTE, Nomic
    'General LLM':           '#1f77b4',  # blue   — E5-Mistral, Phi-3
    'Domain LLM':            '#8c564b',  # brown  — BioMistral-7B (new)
}

# 13 models × consistent ordering
MODELS_ORDERED = [
    # Tier 2 — non-retrieval-trained (expect ZCA helps)
    'BERT-base-uncased', 'BioBERT', 'ClinicalBERT',
    'Phi-3-mini', 'BioMistral-7B',
    'E5-Mistral-7B', 'E5-Mistral-7B-ablation',
    # Tier 1 — retrieval-trained (expect ZCA neutral/hurts at final layer)
    'BioLORD-2023', 'MedCPT',
    'BGE-base', 'GTE-base',
    'Nomic-embed-text', 'Nomic-embed-text-nopfx',
]

# Tier mapping
TIER = {
    'BERT-base-uncased': 'Tier 2 (non-retrieval)',
    'BioBERT': 'Tier 2 (non-retrieval)',
    'ClinicalBERT': 'Tier 2 (non-retrieval)',
    'Phi-3-mini': 'Tier 2 (non-retrieval)',
    'BioMistral-7B': 'Tier 2 (non-retrieval)',
    'E5-Mistral-7B': 'Tier 2 (non-retrieval)',
    'E5-Mistral-7B-ablation': 'Tier 2 (non-retrieval)',
    'BioLORD-2023': 'Tier 1 (retrieval-trained)',
    'MedCPT': 'Tier 1 (retrieval-trained)',
    'BGE-base': 'Tier 1 (retrieval-trained)',
    'GTE-base': 'Tier 1 (retrieval-trained)',
    'Nomic-embed-text': 'Tier 1 (retrieval-trained)',
    'Nomic-embed-text-nopfx': 'Tier 1 (retrieval-trained)',
}

# Model → category (derived from Stream C rank data which has 'category' column)
ranks = pd.read_parquet(REVISION / 'stream_c' / 'rank_data' / 'per_query_ranks.parquet')
MODEL_CATEGORY = ranks[['model','category']].drop_duplicates().set_index('model')['category'].to_dict()
# Override for models extracted from Stream A (which have updated category labels)
print("Model → category:")
for m in MODELS_ORDERED:
    print(f"  {m}: {MODEL_CATEGORY.get(m, '?')}")


# ==CELL 3: FIGURE 1 — Layer-wise MRR@10 trajectories (raw + normalized)==
print("\n[Figure 1] Layer trajectories...")

# Compute baseline MRR@10 per (model, layer) from per-query ranks
baseline_ranks = ranks[ranks['intervention']=='baseline']
mrr_by_layer = (baseline_ranks
                .assign(rr=lambda d: np.where(d['rank']<=10, 1.0/d['rank'], 0.0))
                .groupby(['model','layer','rel_layer','max_layer'])['rr']
                .mean()
                .reset_index()
                .rename(columns={'rr':'MRR@10'}))

# Min-max normalize per model
mrr_by_layer['MRR_normalized'] = mrr_by_layer.groupby('model')['MRR@10'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x*0
)

fig, axes = plt.subplots(2, 1, figsize=(7, 8))

for ax, ycol, ylabel in [(axes[0], 'MRR@10', 'MRR@10 (raw)'),
                          (axes[1], 'MRR_normalized', 'MRR@10 (normalized 0-1)')]:
    for model in MODELS_ORDERED:
        sub = mrr_by_layer[mrr_by_layer['model']==model].sort_values('rel_layer')
        if len(sub) == 0:
            continue
        cat = MODEL_CATEGORY.get(model, 'General Embedding')
        color = CATEGORY_COLOR.get(cat, '#888888')
        ax.plot(sub['rel_layer'], sub[ycol], '-o', color=color, alpha=0.85,
                linewidth=1.5, markersize=4, label=model if ax is axes[0] else None)
    ax.set_xlabel('Relative layer depth (0 = embedding, 1 = final layer)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)

# Category legend on top panel
category_handles = [mpatches.Patch(color=c, label=cat) for cat, c in CATEGORY_COLOR.items()]
axes[0].legend(handles=category_handles, loc='upper left', frameon=False, ncol=2, fontsize=8)
axes[0].set_title('(A) Raw MRR@10 trajectories across 13 model configurations', loc='left')
axes[1].set_title('(B) Normalized MRR@10 (rescaled per model to 0–1 range)', loc='left')

plt.tight_layout()
plt.savefig(FIG_DIR / 'Figure1_layer_trajectories.png')
plt.close()
print(f"  Saved Figure1_layer_trajectories.png")


# ==CELL 4: FIGURE 2 — Final-layer geometry by category (PR, anisotropy, avg_cosine)==
print("\n[Figure 2] Geometry diagnostics...")

geom = pd.read_parquet(REVISION / 'stream_eg' / 'stream_g' / 'geometry_full.parquet')
geom_doc = geom[geom['embedding_type']=='doc'].copy()
geom_summary = geom_doc.groupby('model').agg(
    participation_ratio=('participation_ratio','mean'),
    avg_cosine=('avg_cosine','mean'),
    anisotropy=('anisotropy','mean'),
).reset_index()
geom_summary = geom_summary.set_index('model').loc[MODELS_ORDERED].reset_index()

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

for ax, col, ylabel, title in zip(
    axes,
    ['participation_ratio', 'avg_cosine', 'anisotropy'],
    ['Participation ratio', 'Avg pairwise cosine', 'Anisotropy (top SV² / Σ SV²)'],
    ['(A) Effective dimensionality', '(B) Geometric collapse', '(C) Anisotropy (SVD-based)']
):
    colors = [CATEGORY_COLOR.get(MODEL_CATEGORY.get(m,''), '#888') for m in geom_summary['model']]
    bars = ax.barh(geom_summary['model'], geom_summary[col], color=colors, alpha=0.85)
    ax.set_xlabel(ylabel)
    ax.set_title(title, loc='left')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

axes[0].set_ylabel('')
plt.tight_layout()
plt.savefig(FIG_DIR / 'Figure2_geometry_diagnostics.png')
plt.close()
print(f"  Saved Figure2_geometry_diagnostics.png")


# ==CELL 5: FIGURE 3 — LME fixed effects per model (key coefficients)==
print("\n[Figure 3] LME coefficients...")

fe = pd.read_parquet(REVISION / 'stream_c' / 'lme_results' / 'refit_bfgs.parquet')
icc = pd.read_parquet(REVISION / 'stream_c' / 'lme_results' / 'manual_icc.parquet')

KEY_TERMS = [
    ('rel_layer', 'Layer depth\n(rel_layer)'),
    ('C(intervention)[T.corpus_only_ZCA]', 'Corpus-only ZCA\nmain effect'),
    ('C(intervention)[T.transductive_ZCA]', 'Transductive ZCA\nmain effect'),
    ('rel_layer:C(intervention)[T.corpus_only_ZCA]', 'rel_layer ×\ncorpus-only ZCA'),
]

fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)

for ax, (term, label) in zip(axes, KEY_TERMS):
    sub = fe[fe['term']==term].set_index('model').reindex(MODELS_ORDERED).reset_index()
    sub['sig'] = sub['p_value'].apply(lambda p: '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'n.s.')) if pd.notna(p) else '')
    colors = ['#d62728' if e > 0 else '#2ca02c' for e in sub['estimate']]
    bars = ax.barh(sub['model'], sub['estimate'], color=colors, alpha=0.8)
    for i, (e, s) in enumerate(zip(sub['estimate'], sub['sig'])):
        if pd.notna(e):
            ax.text(e + (0.05 if e > 0 else -0.05), i, s,
                   ha='left' if e>0 else 'right', va='center', fontsize=8)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Coefficient (log rank+1)')
    ax.set_title(label, loc='left', fontsize=10)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

axes[0].set_ylabel('')
fig.suptitle('LME fixed effects: log(rank+1) ~ rel_layer × intervention + corpus + qf + (1|query_idx)\n'
             '(positive = worse retrieval; negative = better; significance: *** p<0.001, ** p<0.01, * p<0.05)',
             fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'Figure3_lme_coefficients.png')
plt.close()
print(f"  Saved Figure3_lme_coefficients.png")


# ==CELL 6: FIGURE 4 — Matched comparisons (BERT-base vs BGE; BioBERT vs BioLORD; BioMistral vs Phi-3)==
print("\n[Figure 4] Matched comparisons...")

matched_pairs = [
    ('General encoder: training objective effect',
     [('BERT-base-uncased', 'no retrieval training'),
      ('BGE-base', 'retrieval-trained')]),
    ('Biomedical encoder: training objective effect',
     [('BioBERT', 'no retrieval training'),
      ('BioLORD-2023', 'retrieval-trained')]),
    ('LLM scale: training objective effect',
     [('BioMistral-7B', 'biomedical, no retrieval'),
      ('Phi-3-mini', 'general, no retrieval'),
      ('E5-Mistral-7B', 'retrieval-trained')]),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for ax, (title, models) in zip(axes, matched_pairs):
    for model, label in models:
        sub = mrr_by_layer[mrr_by_layer['model']==model].sort_values('rel_layer')
        if len(sub) == 0:
            continue
        cat = MODEL_CATEGORY.get(model, '')
        color = CATEGORY_COLOR.get(cat, '#888')
        # Distinguish retrieval-trained vs not by linestyle
        ls = '-' if 'retrieval-trained' in label else '--'
        ax.plot(sub['rel_layer'], sub['MRR@10'], ls, color=color,
                linewidth=2, marker='o', markersize=5,
                label=f"{model}\n({label})")
    ax.set_xlabel('Relative layer depth')
    ax.set_ylabel('MRR@10')
    ax.set_title(title, loc='left', fontsize=10)
    ax.legend(loc='best', fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)

plt.tight_layout()
plt.savefig(FIG_DIR / 'Figure4_matched_comparisons.png')
plt.close()
print(f"  Saved Figure4_matched_comparisons.png")


# ==CELL 7: FIGURE 5 — Intervention comparison (baseline / corpus-only / transductive)==
print("\n[Figure 5] Intervention comparison...")

method = pd.read_parquet(REVISION / 'stream_b' / 'results' / 'methodology_comparison.parquet')
m_agg = method.groupby('model').agg(
    baseline=('baseline_MRR@10','mean'),
    corpus_only=('corpus_only_ZCA_MRR@10','mean'),
    transductive=('transductive_ZCA_MRR@10','mean'),
).reset_index()
m_agg = m_agg.set_index('model').loc[MODELS_ORDERED].reset_index()

fig, ax = plt.subplots(figsize=(11, 6))

x = np.arange(len(MODELS_ORDERED))
w = 0.27

ax.bar(x - w, m_agg['baseline'], w, label='Baseline (no intervention)', color='#888', alpha=0.85)
ax.bar(x,     m_agg['corpus_only'], w, label='Corpus-only ZCA (primary)', color='#2ca02c', alpha=0.85)
ax.bar(x + w, m_agg['transductive'], w, label='Transductive ZCA (upper bound)', color='#1f77b4', alpha=0.85)

# Tier separator
tier1_start = MODELS_ORDERED.index('BioLORD-2023')
ax.axvline(tier1_start - 0.5, color='black', linewidth=1, linestyle=':', alpha=0.7)
ax.text(tier1_start/2, 1.0, 'Tier 2 (non-retrieval-trained)', ha='center',
        fontsize=10, style='italic', color='#555')
ax.text((tier1_start + len(MODELS_ORDERED))/2, 1.0, 'Tier 1 (retrieval-trained)', ha='center',
        fontsize=10, style='italic', color='#555')

ax.set_xticks(x)
ax.set_xticklabels(MODELS_ORDERED, rotation=45, ha='right')
ax.set_ylabel('MRR@10 (mean across 6 conditions)')
ax.set_title('Final-layer MRR@10: baseline vs corpus-only ZCA vs transductive ZCA (13 models)', loc='left')
ax.legend(loc='upper right', frameon=False)
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(FIG_DIR / 'Figure5_intervention_comparison.png')
plt.close()
print(f"  Saved Figure5_intervention_comparison.png")


# ==CELL 8: FIGURE 6 — PR vs final-layer MRR scatter==
print("\n[Figure 6] PR vs MRR...")

pr_mrr = geom_summary.merge(
    m_agg[['model','baseline']], on='model'
).rename(columns={'baseline':'MRR@10'})

from scipy.stats import spearmanr
rho, p = spearmanr(pr_mrr['participation_ratio'], pr_mrr['MRR@10'])

fig, ax = plt.subplots(figsize=(7, 6))
for _, row in pr_mrr.iterrows():
    cat = MODEL_CATEGORY.get(row['model'], '')
    color = CATEGORY_COLOR.get(cat, '#888')
    ax.scatter(row['participation_ratio'], row['MRR@10'], s=120, color=color,
               alpha=0.85, edgecolors='black', linewidth=0.5)
    ax.annotate(row['model'], (row['participation_ratio'], row['MRR@10']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax.set_xlabel('Final-layer participation ratio (mean across conditions)')
ax.set_ylabel('Final-layer baseline MRR@10')
ax.set_title(f'Participation ratio vs MRR@10 across 13 models\n(Spearman ρ = {rho:.3f}, p = {p:.3f})', loc='left')
ax.grid(True, alpha=0.3)

cat_handles = [mpatches.Patch(color=c, label=cat) for cat, c in CATEGORY_COLOR.items() if cat in MODEL_CATEGORY.values()]
ax.legend(handles=cat_handles, loc='lower right', fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig(FIG_DIR / 'Figure6_pr_vs_mrr.png')
plt.close()
print(f"  Saved Figure6_pr_vs_mrr.png (ρ={rho:.3f})")


# ==CELL 9: FIGURE 7 — Anisotropy 3-tier visualization==
print("\n[Figure 7] Anisotropy 3-tier...")

geom_summary_sorted = geom_summary.sort_values('avg_cosine', ascending=False).reset_index(drop=True)
colors = [CATEGORY_COLOR.get(MODEL_CATEGORY.get(m,''),'#888') for m in geom_summary_sorted['model']]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(geom_summary_sorted['model'], geom_summary_sorted['avg_cosine'], color=colors, alpha=0.85)

# Tier boundaries
ax.axhline(0.92, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(0.65, color='orange', linestyle='--', linewidth=1, alpha=0.7)
ax.text(0, 0.94, 'Extreme anisotropy (>0.92)', fontsize=9, style='italic', color='red')
ax.text(0, 0.67, 'Moderate anisotropy (0.65-0.92)', fontsize=9, style='italic', color='orange')
ax.text(0, 0.20, 'Reduced anisotropy (<0.65)', fontsize=9, style='italic', color='green')

ax.set_ylabel('Average pairwise cosine similarity (document embeddings)')
ax.set_title('Anisotropy across 13 model configurations (G.2)', loc='left')
ax.set_xticklabels(geom_summary_sorted['model'], rotation=45, ha='right')
ax.set_ylim(0, 1.05)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'Figure7_anisotropy_13models.png')
plt.close()
print(f"  Saved Figure7_anisotropy_13models.png")


# ==CELL 10: FIGURE 8 — Two-tier ZCA pattern (deltas)==
print("\n[Figure 8] Two-tier pattern...")

m_delta = method.groupby('model').agg(
    delta_corpus_only=('delta_corpus_only','mean'),
    delta_transductive=('delta_transductive','mean'),
).reset_index().set_index('model').loc[MODELS_ORDERED].reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, col, title in zip(axes,
                          ['delta_corpus_only', 'delta_transductive'],
                          ['(A) Corpus-only ZCA: ΔMRR@10 vs baseline',
                           '(B) Transductive ZCA: ΔMRR@10 vs baseline']):
    colors = ['#2ca02c' if d > 0 else '#d62728' for d in m_delta[col]]
    ax.barh(m_delta['model'], m_delta[col], color=colors, alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Δ MRR@10 (post-ZCA − baseline)')
    ax.set_title(title, loc='left')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    # Tier separator
    tier1_start = MODELS_ORDERED.index('BioLORD-2023')
    ax.axhline(tier1_start - 0.5, color='black', linewidth=1, linestyle=':', alpha=0.7)
    ax.text(ax.get_xlim()[1]*0.95, tier1_start/2, 'Tier 2', ha='right', fontsize=9, style='italic', color='#555')
    ax.text(ax.get_xlim()[1]*0.95, (tier1_start + len(MODELS_ORDERED))/2, 'Tier 1', ha='right', fontsize=9, style='italic', color='#555')

axes[0].set_ylabel('')
plt.tight_layout()
plt.savefig(FIG_DIR / 'Figure8_two_tier_pattern.png')
plt.close()
print(f"  Saved Figure8_two_tier_pattern.png")


# ==CELL 11: FIGURE 9 — E5-Mistral-ablation intervention recovery==
print("\n[Figure 9] E5-ablation recovery...")

e5 = pd.read_parquet(REVISION / 'stream_eg' / 'stream_g' / 'e5_ablation_intervention_sweep.parquet')
e5_agg = e5.groupby('intervention').agg(
    MRR=('MRR@10','mean'),
    Recall=('Recall@10','mean'),
).reset_index()

# Order
intervention_order = ['baseline', 'mean_centering', 'corpus_only_ZCA_eps1e-5',
                       'transductive_ZCA_eps1e-5', 'transductive_ZCA_eps1e-3']
intervention_labels = ['Baseline\n(no intervention)', 'Mean centering',
                        'Corpus-only ZCA\nε=1e-5', 'Transductive ZCA\nε=1e-5',
                        'Transductive ZCA\nε=1e-3']
e5_agg = e5_agg.set_index('intervention').loc[intervention_order].reset_index()

fig, ax = plt.subplots(figsize=(9, 5.5))
baseline_mrr = e5_agg[e5_agg['intervention']=='baseline']['MRR'].values[0]
colors_e5 = ['#888', '#d62728', '#2ca02c', '#1f77b4', '#9467bd']
bars = ax.bar(range(len(e5_agg)), e5_agg['MRR'], color=colors_e5, alpha=0.85)

for i, mrr in enumerate(e5_agg['MRR']):
    delta = mrr - baseline_mrr
    label = f"{mrr:.3f}\n(Δ={delta:+.3f})" if i > 0 else f"{mrr:.3f}\n(baseline)"
    ax.text(i, mrr + 0.02, label, ha='center', fontsize=9)

ax.axhline(baseline_mrr, color='black', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xticks(range(len(e5_agg)))
ax.set_xticklabels(intervention_labels, fontsize=9)
ax.set_ylabel('MRR@10 (mean across 6 conditions)')
ax.set_title('E5-Mistral-7B-ablation: post-hoc intervention recovery (Stream G.3)\n'
             'Removing instruction-tuning degrades performance; transductive ZCA at ε=1e-3 recovers it', loc='left')
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, max(e5_agg['MRR']) * 1.15)

plt.tight_layout()
plt.savefig(FIG_DIR / 'Figure9_e5_ablation_recovery.png')
plt.close()
print(f"  Saved Figure9_e5_ablation_recovery.png")


# ==CELL 12: FIGURE 10 — Cross-validation per model with fold variance==
print("\n[Figure 10] Cross-validation...")

cv = pd.read_parquet(REVISION / 'stream_b' / 'results' / 'cross_validation.parquet')
cv_agg = cv.groupby('model').agg(
    baseline=('baseline_MRR@10_full','mean'),
    cv_mean=('cv_MRR@10','mean'),
    cv_std=('cv_MRR@10','std'),
).reset_index()
cv_agg['delta'] = cv_agg['cv_mean'] - cv_agg['baseline']
cv_agg = cv_agg.set_index('model').loc[MODELS_ORDERED].reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2ca02c' if d > 0 else '#d62728' for d in cv_agg['delta']]
bars = ax.barh(cv_agg['model'], cv_agg['delta'], color=colors, alpha=0.85,
               xerr=cv_agg['cv_std']/np.sqrt(5),  # SE across 5 folds
               error_kw={'capsize': 3, 'elinewidth': 0.8, 'ecolor': '#444'})
ax.axvline(0, color='black', linewidth=0.5)
tier1_start = MODELS_ORDERED.index('BioLORD-2023')
ax.axhline(tier1_start - 0.5, color='black', linewidth=1, linestyle=':', alpha=0.7)
ax.text(ax.get_xlim()[1]*0.95, tier1_start/2, 'Tier 2', ha='right', fontsize=10, style='italic')
ax.text(ax.get_xlim()[1]*0.95, (tier1_start + len(MODELS_ORDERED))/2, 'Tier 1', ha='right', fontsize=10, style='italic')

ax.set_xlabel('Δ MRR@10 (5-fold CV mean − baseline) ± SE')
ax.set_title('Cross-validation: corpus-only ZCA fitted on 80 docs generalizes to held-out 20', loc='left')
ax.invert_yaxis()
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'Figure10_cross_validation.png')
plt.close()
print(f"  Saved Figure10_cross_validation.png")


# ==CELL 13: FIGURE CAPTIONS DRAFT==
print("\n[Captions] Drafting...")

captions = """# Figure Captions — Paper 12 Revision (draft)

## Figure 1. Layer-wise MRR@10 trajectories across 13 model configurations.
(A) Raw MRR@10 by relative layer depth (0 = input embeddings, 1 = final layer). Each line represents one of 13 model configurations, averaged across 3 corpora × 2 query formats. Models color-coded by category. (B) Same data, min-max normalized per model to a 0-1 range to facilitate cross-model trajectory comparison; absolute performance differences obscured. Most models exhibit U-shaped or monotonically improving curves; non-retrieval-trained encoder models (BERT-base, BioBERT) show classical mid-layer degradation, while LLMs and retrieval-trained models improve with depth.

## Figure 2. Final-layer geometric diagnostics across 13 model configurations.
(A) Participation ratio (effective embedding dimensionality), (B) average pairwise cosine similarity (anisotropy in cone width), and (C) anisotropy index (top squared singular value / sum of squared singular values). Document embeddings only; values averaged across 3 corpora × 2 query formats. Color coding by model category. BioLORD-2023 (contrastive isotropy-promoting training) reaches the lowest avg cosine (0.30); BERT-base, BioBERT, ClinicalBERT, and Phi-3-mini cluster at >0.92 avg cosine (extreme anisotropy).

## Figure 3. Linear mixed-effects model fixed effects across 13 model configurations.
Per-model LME: log(rank+1) ~ rel_layer × C(intervention) + C(corpus) + C(query_format) + (1|query_idx). Four key coefficients shown: (A) rel_layer main effect (degradation with depth vs improvement), (B) corpus-only ZCA main effect, (C) transductive ZCA main effect, (D) rel_layer × corpus-only ZCA interaction. Red bars: positive estimates (worse retrieval); green bars: negative estimates (better retrieval). Significance: *** p<.001, ** p<.01, * p<.05. All 13 models converged with refit (BFGS optimizer); manual ICC range 0.018-0.130, median 0.081.

## Figure 4. Matched-comparison panels demonstrating training-objective effects independent of training domain.
(A) General encoder pair: BERT-base-uncased (general, no retrieval training) vs BGE-base (general, retrieval-trained). (B) Biomedical encoder pair: BioBERT (biomedical, no retrieval training) vs BioLORD-2023 (biomedical, retrieval-trained). (C) LLM-scale comparison: BioMistral-7B (biomedical) and Phi-3-mini (general) — both without retrieval training — vs E5-Mistral-7B (retrieval-trained). Dashed lines indicate no-retrieval-training models; solid lines indicate retrieval-trained models. Within each architectural class, retrieval training is the operative variable; biomedical pretraining alone does not produce well-calibrated retrieval embeddings.

## Figure 5. Final-layer MRR@10 across 13 model configurations under three retrieval configurations.
Grouped bars: baseline (no intervention), corpus-only ZCA whitening (primary post-hoc methodology; fit on document embeddings only), and transductive ZCA whitening (upper bound; fit on combined documents + queries). Dotted vertical line separates Tier 2 (non-retrieval-trained models, left) from Tier 1 (retrieval-trained models, right). For Tier 2 models, both ZCA variants substantially improve retrieval; for Tier 1 models, the transductive variant slightly degrades performance while corpus-only ZCA is largely neutral.

## Figure 6. Participation ratio versus final-layer baseline MRR@10 across 13 model configurations.
Each point represents one of 13 models; final-layer baseline MRR@10 averaged across 3 corpora × 2 query formats; participation ratio computed on document embeddings averaged across the same conditions. Spearman ρ reported in title (95% CI not shown due to n=13 limiting statistical power). Color coding by model category.

## Figure 7. Anisotropy bands across 13 model configurations.
Document-embedding average pairwise cosine similarity sorted descending. Three empirical tiers visible: extreme anisotropy (>0.92; non-retrieval-trained models — Phi-3, BioBERT, ClinicalBERT, BERT-base), moderate (0.65-0.92; most retrieval-trained encoders and LLMs), and reduced (<0.65; BioLORD-2023 and E5-Mistral-7B with instruction). BioLORD-2023's contrastive isotropy-promoting training objective yields the lowest avg cosine in the panel.

## Figure 8. Two-tier ZCA recovery pattern across 13 model configurations.
ΔMRR@10 (post-intervention minus baseline) for corpus-only ZCA (A) and transductive ZCA (B). Tier 2 (non-retrieval-trained) models show large positive deltas under both interventions; Tier 1 (retrieval-trained) models show small or negative deltas under transductive ZCA and small to slightly negative deltas under corpus-only ZCA. This pattern is the central empirical finding of the revision: ZCA whitening achieves a similar geometric calibration that retrieval training accomplishes implicitly.

## Figure 9. E5-Mistral-7B-ablation post-hoc intervention sweep (Stream G.3).
Five interventions applied to the E5-Mistral-7B-ablation embeddings (model with retrieval-specific instruction-tuning removed): baseline, mean centering only, corpus-only ZCA at ε=1e-5, transductive ZCA at ε=1e-5, and transductive ZCA at ε=1e-3. Values are MRR@10 averaged across 6 conditions; Δ values shown above each bar relative to baseline. Removing instruction-tuning degrades the full E5-Mistral-7B (baseline 0.92) to 0.66; transductive ZCA at ε=1e-3 recovers performance to 0.89, demonstrating that post-hoc whitening can approximate the calibration effect of retrieval-specific instruction-tuning.

## Figure 10. Cross-validation of corpus-only ZCA across 13 model configurations.
ΔMRR@10 (5-fold cross-validated minus full-data baseline) per model. ZCA fitted on 80 documents per fold, evaluated on held-out 20 query-document pairs against the full 100-document candidate set. Error bars: standard error of the mean across 5 folds. Tier 2 models show large positive deltas (BERT-base +0.199, BioBERT +0.196, ClinicalBERT +0.207, BioMistral-7B +0.161, E5-Mistral +0.304); Tier 1 models show modest negative deltas (BGE -0.046, GTE -0.049, Nomic -0.051). Confirms ZCA generalizes from training to held-out documents and replicates the Tier 1/Tier 2 pattern under stricter evaluation.
"""

(FIG_DIR / 'figure_captions.md').write_text(captions)
print(f"  Saved figure_captions.md")


# ==CELL 14: SUMMARY==
print("\n" + "="*70)
print("FIGURE GENERATION COMPLETE")
print("="*70)
print(f"Output directory: {FIG_DIR}")
print("\nGenerated:")
for f in sorted(FIG_DIR.iterdir()):
    print(f"  {f.name}: {f.stat().st_size/1024:.0f} KB")