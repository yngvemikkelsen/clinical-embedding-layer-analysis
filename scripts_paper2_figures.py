"""
Paper 2 Figure Generation — Publication-Ready
Run in Colab after experiments complete.
Generates 6 figures matching manuscript V4 figure legends.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ──
base = Path("/content/drive/MyDrive/Content/paper2_layer_analysis")
FIGURES_DIR = base / "figures_v2"
FIGURES_DIR.mkdir(exist_ok=True)

df = pd.read_parquet(base / "results/all_models_combined.parquet")
overall = df[df["length_tercile"] == "all"].copy()

# Load Type II ANOVA (run the proper ANOVA cell first if not present)
try:
    bert_anova = pd.read_parquet(base / "anova/anova_bert_typeII.parquet")
    llm_anova = pd.read_parquet(base / "anova/anova_llm_typeII.parquet")
except FileNotFoundError:
    print("⚠ Type II ANOVA parquets not found. Run ANOVA cell first.")
    bert_anova, llm_anova = None, None

# ── Style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linewidth": 0.5,
})

CAT_COLORS = {
    "Domain Encoder": "#c0392b",
    "Biomedical Retriever": "#e67e22",
    "General Embedding": "#27ae60",
    "General LLM": "#2980b9",
}

# Consistent model ordering and styling
MODEL_STYLE = {
    "BioBERT":              {"color": "#c0392b", "marker": "o", "ls": "-"},
    "ClinicalBERT":         {"color": "#e74c3c", "marker": "s", "ls": "--"},
    "BioLORD-2023":         {"color": "#e67e22", "marker": "^", "ls": "-"},
    "MedCPT":               {"color": "#f39c12", "marker": "D", "ls": "--"},
    "BGE-base":             {"color": "#27ae60", "marker": "P", "ls": "-"},
    "GTE-base":             {"color": "#2ecc71", "marker": "*", "ls": "-"},
    "Nomic-embed-text":     {"color": "#1abc9c", "marker": "X", "ls": "-"},
    "Nomic-embed-text-nopfx":{"color": "#16a085","marker": "x", "ls": ":"},
    "E5-Mistral-7B":        {"color": "#2980b9", "marker": "v", "ls": "-"},
    "E5-Mistral-7B-ablation":{"color":"#3498db", "marker": "v", "ls": ":"},
    "Phi-3-mini":           {"color": "#8e44ad", "marker": "p", "ls": "-"},
}

# Primary models (exclude ablations for main figures)
PRIMARY = [m for m in MODEL_STYLE if "ablation" not in m and "nopfx" not in m]
BERT_PRIMARY = ["BioBERT","ClinicalBERT","BioLORD-2023","MedCPT",
                "BGE-base","GTE-base","Nomic-embed-text"]
LLM_MODELS = ["E5-Mistral-7B","Phi-3-mini"]


def get_style(model):
    return MODEL_STYLE.get(model, {"color": "#999", "marker": "o", "ls": "-"})


def get_cat_color(model):
    cat = overall[overall["model"] == model]["category"].iloc[0]
    return CAT_COLORS.get(cat, "#999")


# ════════════════════════════════════════════════════════
# FIGURE 1: Layer-wise MRR@10 trajectories
# ════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: BERT-scale
for model in BERT_PRIMARY:
    sub = overall[overall["model"] == model]
    avg = sub.groupby("layer")["MRR@10"].mean().sort_index()
    st = get_style(model)
    ax1.plot(avg.index, avg.values, marker=st["marker"], markersize=4,
             color=st["color"], linestyle=st["ls"], label=model, linewidth=1.5)
ax1.set_xlabel("Layer")
ax1.set_ylabel("MRR@10")
ax1.set_title("(A) BERT-scale Models", fontweight="bold")
ax1.legend(fontsize=7, ncol=2, loc="upper left")
ax1.set_ylim(-0.02, 1.05)

# Panel B: LLM-scale
for model in LLM_MODELS:
    sub = overall[overall["model"] == model]
    avg = sub.groupby("layer")["MRR@10"].mean().sort_index()
    st = get_style(model)
    ax2.plot(avg.index, avg.values, marker=st["marker"], markersize=4,
             color=st["color"], linestyle=st["ls"], label=model, linewidth=1.5)
ax2.set_xlabel("Layer")
ax2.set_ylabel("MRR@10")
ax2.set_title("(B) LLM-scale Models", fontweight="bold")
ax2.legend(fontsize=7)
ax2.set_ylim(-0.02, 1.05)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig1_mrr_by_layer.pdf")
fig.savefig(FIGURES_DIR / "fig1_mrr_by_layer.png")
plt.close()
print("✅ Figure 1: MRR@10 by layer")


# ════════════════════════════════════════════════════════
# FIGURE 2: Participation ratio by layer
# ════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for model in BERT_PRIMARY:
    sub = overall[overall["model"] == model]
    avg = sub.groupby("layer")["participation_ratio"].mean().sort_index()
    st = get_style(model)
    ax1.plot(avg.index, avg.values, marker=st["marker"], markersize=4,
             color=st["color"], linestyle=st["ls"], label=model, linewidth=1.5)
ax1.set_xlabel("Layer")
ax1.set_ylabel("Participation Ratio")
ax1.set_title("(A) BERT-scale Models", fontweight="bold")
ax1.legend(fontsize=7, ncol=2, loc="upper left")

for model in LLM_MODELS:
    sub = overall[overall["model"] == model]
    avg = sub.groupby("layer")["participation_ratio"].mean().sort_index()
    st = get_style(model)
    ax2.plot(avg.index, avg.values, marker=st["marker"], markersize=4,
             color=st["color"], linestyle=st["ls"], label=model, linewidth=1.5)
ax2.set_xlabel("Layer")
ax2.set_ylabel("Participation Ratio")
ax2.set_title("(B) LLM-scale Models", fontweight="bold")
ax2.legend(fontsize=7)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig2_participation_ratio_by_layer.pdf")
fig.savefig(FIGURES_DIR / "fig2_participation_ratio_by_layer.png")
plt.close()
print("✅ Figure 2: Participation ratio by layer")


# ════════════════════════════════════════════════════════
# FIGURE 3: Type II ANOVA variance decomposition by layer
# ════════════════════════════════════════════════════════
if bert_anova is not None and llm_anova is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: BERT-scale
    ba = bert_anova.sort_values("layer")
    layers = ba["layer"].values
    factors = ["eta_sq_model", "eta_sq_corpus", "eta_sq_query_format",
               "eta_sq_model_x_corpus", "eta_sq_residual"]
    labels = ["Model", "Corpus", "Query format", "Model × Corpus", "Residual"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#bdc3c7"]

    vals = [ba[f].values for f in factors]
    ax1.stackplot(layers, *vals, labels=labels, colors=colors, alpha=0.85)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("η² (variance explained)")
    ax1.set_title("(A) BERT-scale: Type II ANOVA", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=7)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, 12)
    ax1.text(6, 0.02, "R² = 0.85–0.98",
             fontsize=7, ha="center", color="#666", style="italic")

    # Panel B: LLM-scale
    la = llm_anova.sort_values("layer")
    layers_l = la["layer"].values
    factors_l = ["eta_sq_model", "eta_sq_corpus", "eta_sq_query_format", "eta_sq_residual"]
    labels_l = ["Model", "Corpus", "Query format", "Residual"]
    colors_l = ["#e74c3c", "#3498db", "#2ecc71", "#bdc3c7"]

    vals_l = [la[f].values for f in factors_l]
    ax2.stackplot(layers_l, *vals_l, labels=labels_l, colors=colors_l, alpha=0.85)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("η² (variance explained)")
    ax2.set_title("(B) LLM-scale: Type II ANOVA (main effects)", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=7)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(0, 32)
    ax2.text(16, 0.02, "n = 3 configurations; interpret cautiously",
             fontsize=7, ha="center", color="#666", style="italic")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_anova_decomposition.pdf")
    fig.savefig(FIGURES_DIR / "fig3_anova_decomposition.png")
    plt.close()
    print("✅ Figure 3: ANOVA decomposition by layer")
else:
    print("⚠ Skipped Figure 3 — ANOVA data not available")


# ════════════════════════════════════════════════════════
# FIGURE 4: BGE-base vs BioBERT layer-by-layer
# ════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

bge = overall[(overall["model"] == "BGE-base") & (overall["corpus"] == "MTSamples")
              & (overall["query_format"] == "keyword")].sort_values("layer")
bio = overall[(overall["model"] == "BioBERT") & (overall["corpus"] == "MTSamples")
              & (overall["query_format"] == "keyword")].sort_values("layer")

ax.plot(bge["layer"], bge["MRR@10"], "o-", color="#27ae60", linewidth=2, markersize=6, label="BGE-base")
ax.plot(bio["layer"], bio["MRR@10"], "s-", color="#c0392b", linewidth=2, markersize=6, label="BioBERT")

# Fill the crossover region
bge_vals = bge.set_index("layer")["MRR@10"]
bio_vals = bio.set_index("layer")["MRR@10"]
common = sorted(set(bge_vals.index) & set(bio_vals.index))
bge_arr = np.array([bge_vals[l] for l in common])
bio_arr = np.array([bio_vals[l] for l in common])
ax.fill_between(common, bge_arr, bio_arr, where=(bio_arr > bge_arr),
                alpha=0.15, color="#c0392b", label="BioBERT leads")
ax.fill_between(common, bge_arr, bio_arr, where=(bge_arr > bio_arr),
                alpha=0.15, color="#27ae60", label="BGE-base leads")

# Annotate key layers with PR
for layer_ann in [1, 8, 11, 12]:
    if layer_ann in bge.set_index("layer").index:
        bge_row = bge[bge["layer"] == layer_ann].iloc[0]
        bio_row = bio[bio["layer"] == layer_ann].iloc[0]
        y_top = max(bge_row["MRR@10"], bio_row["MRR@10"])
        ax.annotate(f"L{layer_ann}\nPR: {bge_row['participation_ratio']:.0f} / {bio_row['participation_ratio']:.0f}",
                    xy=(layer_ann, y_top + 0.02), fontsize=7, ha="center", color="#555")

ax.set_xlabel("Layer")
ax.set_ylabel("MRR@10")
ax.set_title("BGE-base vs BioBERT — MTSamples (Keyword Queries)", fontweight="bold")
ax.legend(fontsize=8, loc="center right")
ax.set_ylim(-0.02, 1.05)
ax.set_xlim(-0.5, 12.5)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig4_bge_vs_biobert.pdf")
fig.savefig(FIGURES_DIR / "fig4_bge_vs_biobert.png")
plt.close()
print("✅ Figure 4: BGE-base vs BioBERT")


# ════════════════════════════════════════════════════════
# FIGURE 5: Intervention comparison (ZCA vs layer selection)
# ════════════════════════════════════════════════════════
wc = pd.read_parquet(base / "interventions/whitening_centering.parquet")
ls = pd.read_parquet(base / "interventions/layer_selection.parquet")

wc_zca = wc[wc["intervention"] == "zca_whitening"]

target_models = ["BioBERT", "ClinicalBERT", "E5-Mistral-7B", "Phi-3-mini"]
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(target_models))
width = 0.25

baselines = []
zca_vals = []
ls_vals = []
for model in target_models:
    wc_sub = wc_zca[wc_zca["model"] == model]
    ls_sub = ls[ls["model"] == model]
    baselines.append(wc_sub["baseline_MRR@10"].mean() if not wc_sub.empty else 0)
    zca_vals.append(wc_sub["intervention_MRR@10"].mean() if not wc_sub.empty else 0)
    ls_vals.append(ls_sub["intervention_MRR@10"].mean() if not ls_sub.empty else 0)

bars1 = ax.bar(x - width, baselines, width, label="Final-layer baseline", color="#bdc3c7", edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x, zca_vals, width, label="ZCA whitening", color="#27ae60", edgecolor="black", linewidth=0.5)
bars3 = ax.bar(x + width, ls_vals, width, label="Layer selection", color="#3498db", edgecolor="black", linewidth=0.5)

# Add delta annotations
for i in range(len(target_models)):
    delta_zca = zca_vals[i] - baselines[i]
    delta_ls = ls_vals[i] - baselines[i]
    ax.text(x[i], zca_vals[i] + 0.01, f"+{delta_zca:.2f}", ha="center", fontsize=7, color="#27ae60", fontweight="bold")
    if delta_ls > 0.005:
        ax.text(x[i] + width, ls_vals[i] + 0.01, f"+{delta_ls:.2f}", ha="center", fontsize=7, color="#2980b9")

ax.set_xticks(x)
ax.set_xticklabels(target_models, fontsize=9)
ax.set_ylabel("MRR@10 (avg across conditions)")
ax.set_title("Post-Hoc Interventions on Degraded Models", fontweight="bold")
ax.legend(fontsize=8)
ax.set_ylim(0, 1.0)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig5_intervention_comparison.pdf")
fig.savefig(FIGURES_DIR / "fig5_intervention_comparison.png")
plt.close()
print("✅ Figure 5: Intervention comparison")


# ════════════════════════════════════════════════════════
# FIGURE 6: Final-layer PR vs MRR@10 (all models)
# ════════════════════════════════════════════════════════
final = overall.groupby("model").apply(
    lambda g: g[g["layer"] == g["layer"].max()]
).reset_index(drop=True)
summary = final.groupby("model").agg(
    mrr=("MRR@10", "mean"),
    pr=("participation_ratio", "mean"),
    category=("category", "first")
).reset_index()

from scipy.stats import spearmanr
r, p = spearmanr(summary["pr"], summary["mrr"])

fig, ax = plt.subplots(figsize=(9, 6.5))

# Model-specific label offsets to avoid crowding
LABEL_OFFSETS = {
    "GTE-base":             (1.0, -0.035),
    "BGE-base":             (1.0, +0.015),
    "Nomic-embed-text":     (-8.0, +0.020),
    "Nomic-embed-text-nopfx":(-8.0, -0.030),
    "MedCPT":               (1.0, +0.015),
    "BioLORD-2023":         (1.0, -0.030),
    "E5-Mistral-7B-ablation":(1.0, +0.015),
    "Phi-3-mini":           (1.0, -0.030),
    "BioBERT":              (1.0, +0.015),
    "E5-Mistral-7B":        (1.0, -0.030),
    "ClinicalBERT":         (1.0, +0.015),
}

for _, row in summary.iterrows():
    st = get_style(row["model"])
    cat_color = CAT_COLORS.get(row["category"], "#999")
    ax.scatter(row["pr"], row["mrr"], color=cat_color, marker=st["marker"],
               s=120, edgecolors="black", linewidth=0.5, zorder=5)
    ox, oy = LABEL_OFFSETS.get(row["model"], (1.0, 0.015))
    ha = "left" if ox > 0 else "right"
    ax.annotate(row["model"], (row["pr"] + ox, row["mrr"] + oy),
                fontsize=7, color="#333", ha=ha)

ax.set_xlabel("Participation Ratio (effective dimensionality)")
ax.set_ylabel("MRR@10")
ax.set_title(f"Final-Layer Geometry vs Retrieval Performance (ρ = {r:.3f}, p = {p:.3f})",
             fontweight="bold")

# Add category legend
for cat, color in CAT_COLORS.items():
    ax.scatter([], [], color=color, label=cat, s=60, edgecolors="black", linewidth=0.5)
ax.legend(fontsize=7, loc="lower right")

plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig6_pr_vs_mrr.pdf")
fig.savefig(FIGURES_DIR / "fig6_pr_vs_mrr.png")
plt.close()
print(f"✅ Figure 6: PR vs MRR@10 (ρ = {r:.3f})")


print(f"\n{'='*60}")
print(f"All figures saved to: {FIGURES_DIR}")
print(f"{'='*60}")
for f in sorted(FIGURES_DIR.glob("*.png")):
    print(f"  {f.name}  ({f.stat().st_size:,} bytes)")
