# Paper 12 v19 — Validation extension: 5 missing models at 500 PMC + 400 MTSamples
#
# Closes the §4.15 / §5.8 validation gap. Re-extracts per-layer embeddings
# at 4-5x scale for the 5 models that v17 did not validate at scale:
#   - BERT-base-uncased  (12-layer encoder, fast)
#   - Phi-3-mini         (32-layer decoder, medium)
#   - BioMistral-7B      (32-layer decoder, slow)
#   - E5-Mistral-7B      (32-layer decoder + EOS pooling + instruction prefix)
#   - E5-Mistral-7B-ablation  (32-layer decoder + mean pooling, no prefix)
#
# Target environment: Colab with Blackwell-class GPU (~96 GB VRAM).
# Expected runtime: ~1.5-2 hours.
#
# Output (per model, saved to OUTPUT_DIR):
#   {Model}_validation500_layers.parquet
# matching the v17 schema:
#   columns = [layer, length_tercile, anisotropy_svd, participation_ratio,
#              avg_cosine, MRR@10, Recall@10, model, category, corpus,
#              query_format, is_ablation, n_docs, n_queries, intervention]
# Plus:
#   validation_summary_5models.parquet
# columns = [model, corpus, query_format, mrr_at_100docs, mrr_at_500docs,
#            ranking_spearman, ranking_kendall]
#
# Match Table 5 / §4.12 by including intervention column with values:
# baseline | corpus_only_ZCA | transductive_ZCA

# ============================================================
# CELL 1 — Setup
# ============================================================
# !pip install -q transformers accelerate sentencepiece tqdm pyarrow

import os
import gc
import time
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

# Mount Drive (uncomment in Colab):
# from google.colab import drive
# drive.mount('/content/drive')

# ============================================================
# CELL 2 — Configuration
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# === EDIT THESE PATHS to match your Drive layout ===
DATA_DIR = Path("/content/drive/MyDrive/Content/paper2_layer_analysis")
OUTPUT_DIR = Path("/content/drive/MyDrive/Content/paper2_layer_analysis/revision/validation_extension")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Where do the 500/400 validation corpora live? ===
# Adjust based on your file layout. The script will print directory listings
# to help you locate them.
CORPUS_PATHS = {
    "PMC-Patients": DATA_DIR / "data" / "pmc500_docs.parquet",  # ADJUST
    "MTSamples": DATA_DIR / "data" / "mtsamples400_docs.parquet",  # ADJUST
}
QUERY_PATHS = {
    "PMC-Patients": DATA_DIR / "data" / "pmc500_queries.parquet",  # ADJUST
    "MTSamples": DATA_DIR / "data" / "mtsamples400_queries.parquet",  # ADJUST
}

# 5 models to extend
MODELS_TO_EXTEND = [
    {
        "name": "BERT-base-uncased",
        "hf_id": "bert-base-uncased",
        "category": "General Encoder",
        "pooling": "mean",
        "instruction_prefix": None,
        "is_ablation": False,
        "dtype": torch.float32,
        "max_length": 512,
        "batch_size": 32,
    },
    {
        "name": "Phi-3-mini",
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "category": "General LLM",
        "pooling": "mean",
        "instruction_prefix": None,
        "is_ablation": False,
        "dtype": torch.float16,
        "max_length": 2048,
        "batch_size": 8,
    },
    {
        "name": "BioMistral-7B",
        "hf_id": "BioMistral/BioMistral-7B",
        "category": "Biomedical LLM",
        "pooling": "mean",
        "instruction_prefix": None,
        "is_ablation": False,
        "dtype": torch.float16,
        "max_length": 2048,
        "batch_size": 4,
    },
    {
        "name": "E5-Mistral-7B",
        "hf_id": "intfloat/e5-mistral-7b-instruct",
        "category": "General LLM",
        "pooling": "eos",
        "instruction_prefix": "Instruct: Given a clinical document, retrieve relevant clinical case descriptions\nQuery: ",
        "is_ablation": False,
        "dtype": torch.float16,
        "max_length": 2048,
        "batch_size": 4,
    },
    {
        "name": "E5-Mistral-7B-ablation",
        "hf_id": "intfloat/e5-mistral-7b-instruct",  # same weights
        "category": "General LLM",
        "pooling": "mean",
        "instruction_prefix": None,
        "is_ablation": True,
        "dtype": torch.float16,
        "max_length": 2048,
        "batch_size": 4,
    },
]

QUERY_FORMATS = ["keyword", "natural_language"]
INTERVENTIONS = ["baseline", "corpus_only_ZCA", "transductive_ZCA"]
ZCA_EPSILON = 1e-5

# ============================================================
# CELL 3 — Schema discovery (RUN THIS FIRST, verify paths)
# ============================================================
def discover_paths():
    print("\n=== DATA_DIR contents ===")
    if DATA_DIR.exists():
        for p in sorted(DATA_DIR.iterdir())[:30]:
            tag = "/" if p.is_dir() else ""
            print(f"  {p.name}{tag}")
    else:
        print(f"  ✗ DATA_DIR does not exist: {DATA_DIR}")

    print("\n=== Looking for 500/400 validation corpus files ===")
    for corpus, path in CORPUS_PATHS.items():
        if path.exists():
            df = pd.read_parquet(path)
            print(f"  ✓ {corpus} corpus at {path}")
            print(f"    shape: {df.shape}, columns: {list(df.columns)}")
            print(f"    first row: {df.iloc[0].to_dict()}")
        else:
            print(f"  ✗ {corpus} corpus NOT FOUND at {path}")
            # Try to find candidates
            parent = path.parent if path.parent.exists() else DATA_DIR
            candidates = [p for p in parent.rglob("*.parquet")
                          if ("500" in p.name or "validation" in p.name.lower()
                              or "pmc" in p.name.lower())]
            if candidates:
                print(f"    Possible candidates in {parent}:")
                for c in candidates[:10]:
                    print(f"      - {c.relative_to(DATA_DIR)}")

    print("\n=== Looking for 500/400 validation queries ===")
    for corpus, path in QUERY_PATHS.items():
        if path.exists():
            df = pd.read_parquet(path)
            print(f"  ✓ {corpus} queries at {path}")
            print(f"    shape: {df.shape}, columns: {list(df.columns)}")
        else:
            print(f"  ✗ {corpus} queries NOT FOUND at {path}")


discover_paths()

# ============================================================
# CELL 4 — Pooling, query formatting
# ============================================================
def pool_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor,
                method: str) -> torch.Tensor:
    """Pool token-level hidden states to single document embedding.

    hidden_states: (B, T, D)
    attention_mask: (B, T)
    returns: (B, D)
    """
    if method == "cls":
        return hidden_states[:, 0, :]
    elif method == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
    elif method == "eos":
        # Pool the last non-padding token
        seq_lens = attention_mask.sum(dim=1) - 1  # (B,)
        idx = seq_lens.long().unsqueeze(-1).unsqueeze(-1).expand(
            -1, 1, hidden_states.size(-1))
        return hidden_states.gather(1, idx).squeeze(1)
    else:
        raise ValueError(f"Unknown pooling: {method}")


def format_text(text: str, instruction_prefix: Optional[str]) -> str:
    if instruction_prefix:
        return instruction_prefix + text
    return text


# ============================================================
# CELL 5 — Batched per-layer embedding extraction
# ============================================================
@torch.no_grad()
def extract_layer_embeddings(
    texts: List[str],
    tokenizer,
    model,
    config: dict,
    desc: str = "extracting",
) -> np.ndarray:
    """Returns (n_texts, n_layers, embedding_dim) numpy array (fp32).

    Extracts pooled embeddings for every layer in one forward pass via
    output_hidden_states=True. Memory-conscious: moves layer embeddings
    to CPU after each batch.
    """
    model.eval()
    pooling = config["pooling"]
    max_length = config["max_length"]
    batch_size = config["batch_size"]
    prefix = config["instruction_prefix"]
    dtype = config["dtype"]

    # Format
    formatted = [format_text(t, prefix) for t in texts]

    all_layer_embs = None  # will become list-of-arrays per layer

    for i in tqdm(range(0, len(formatted), batch_size), desc=desc):
        batch = formatted[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.amp.autocast(device_type="cuda",
                                dtype=dtype, enabled=(DEVICE == "cuda")):
            outputs = model(**enc, output_hidden_states=True,
                            return_dict=True)

        # outputs.hidden_states is tuple of (n_layers+1,) tensors,
        # each (B, T, D). Index 0 is the input embeddings.
        hidden_states = outputs.hidden_states
        n_layers = len(hidden_states)

        if all_layer_embs is None:
            all_layer_embs = [[] for _ in range(n_layers)]

        for layer_idx, h in enumerate(hidden_states):
            pooled = pool_hidden(h.float(), enc["attention_mask"], pooling)
            all_layer_embs[layer_idx].append(pooled.cpu().numpy())

        # Free VRAM
        del outputs, hidden_states, enc
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Concatenate batches per layer, stack layers
    per_layer = [np.concatenate(layer_batches, axis=0)
                 for layer_batches in all_layer_embs]
    # (n_texts, n_layers, D)
    return np.stack(per_layer, axis=1)


# ============================================================
# CELL 6 — Geometric diagnostics
# ============================================================
def compute_geometry(embs: np.ndarray) -> dict:
    """embs: (n_docs, D). Returns participation_ratio, avg_cosine,
    anisotropy_svd."""
    if embs.shape[0] < 2:
        return {"participation_ratio": np.nan, "avg_cosine": np.nan,
                "anisotropy_svd": np.nan}
    # Centered SVD for anisotropy
    centered = embs - embs.mean(axis=0, keepdims=True)
    # Singular values
    try:
        s = np.linalg.svd(centered, compute_uv=False)
        s2 = s ** 2
        anisotropy_svd = float(s2[0] / s2.sum()) if s2.sum() > 0 else np.nan
        # Participation ratio: (sum s^2)^2 / sum s^4
        pr = float((s2.sum() ** 2) / (s2 ** 2).sum()) if (s2 ** 2).sum() > 0 \
            else np.nan
    except Exception:
        pr = np.nan
        anisotropy_svd = np.nan
    # Avg pairwise cosine
    normed = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    sim = normed @ normed.T
    n = sim.shape[0]
    iu = np.triu_indices(n, k=1)
    avg_cos = float(sim[iu].mean())
    return {"participation_ratio": pr, "avg_cosine": avg_cos,
            "anisotropy_svd": anisotropy_svd}


# ============================================================
# CELL 7 — ZCA whitening
# ============================================================
def zca_whiten(fit_data: np.ndarray, apply_data: np.ndarray,
               epsilon: float = ZCA_EPSILON) -> np.ndarray:
    """Fit ZCA on fit_data, apply to apply_data."""
    mu = fit_data.mean(axis=0, keepdims=True)
    fit_c = fit_data - mu
    apply_c = apply_data - mu
    cov = (fit_c.T @ fit_c) / max(fit_c.shape[0] - 1, 1)
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Regularize
    eigvals = np.maximum(eigvals, 0) + epsilon
    # W = V * diag(1/sqrt(eigvals)) * V.T
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return apply_c @ W


# ============================================================
# CELL 8 — Retrieval metric
# ============================================================
def compute_mrr_recall(query_embs: np.ndarray, doc_embs: np.ndarray,
                       relevant_doc_idx: np.ndarray,
                       k: int = 10) -> Tuple[float, float, np.ndarray]:
    """Return MRR@k, Recall@k, and per-query rank of relevant doc."""
    qn = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-9)
    dn = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-9)
    sim = qn @ dn.T  # (Q, D)
    # Rank of each query's relevant doc
    n_queries = len(relevant_doc_idx)
    ranks = np.zeros(n_queries, dtype=np.int64)
    for q in range(n_queries):
        scores = sim[q]
        rel_score = scores[relevant_doc_idx[q]]
        rank = int((scores > rel_score).sum()) + 1
        ranks[q] = rank
    rr = np.where(ranks <= k, 1.0 / ranks, 0.0)
    hit = (ranks <= k).astype(float)
    return float(rr.mean()), float(hit.mean()), ranks


# ============================================================
# CELL 9 — Run one model × corpus × query_format
# ============================================================
def evaluate_model_corpus(
    config: dict,
    tokenizer,
    model,
    docs_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    corpus: str,
    query_format: str,
) -> List[dict]:
    """Returns one row per (layer × intervention)."""
    # === EDIT column names to match your corpus / query parquets ===
    doc_text_col = "document_text"  # ADJUST
    doc_id_col = "doc_id"  # ADJUST
    query_text_col = (f"query_{query_format}" if f"query_{query_format}"
                      in queries_df.columns else "query_text")  # ADJUST
    query_relevant_col = "relevant_doc_id"  # ADJUST

    doc_texts = docs_df[doc_text_col].tolist()
    doc_ids = docs_df[doc_id_col].tolist()
    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}

    # Filter queries to those whose relevant doc is in this corpus
    q_filtered = queries_df[queries_df[query_relevant_col].isin(
        doc_id_to_idx.keys())].reset_index(drop=True)
    if len(q_filtered) == 0:
        print(f"    ⚠ no queries match docs for {corpus} {query_format}")
        return []
    query_texts = q_filtered[query_text_col].tolist()
    relevant_idx = np.array([doc_id_to_idx[d]
                             for d in q_filtered[query_relevant_col]])

    print(f"    Docs: {len(doc_texts)}, Queries: {len(query_texts)}")

    # Extract embeddings (one forward pass per text, all layers)
    doc_embs = extract_layer_embeddings(
        doc_texts, tokenizer, model, config, desc=f"  docs/{query_format}")
    q_embs = extract_layer_embeddings(
        query_texts, tokenizer, model, config, desc=f"  queries/{query_format}")
    n_layers = doc_embs.shape[1]

    # Length terciles for stratification
    doc_lens = np.array([len(t.split()) for t in doc_texts])
    tercile_edges = np.percentile(doc_lens, [33.33, 66.67])
    doc_terciles = np.digitize(doc_lens, tercile_edges)

    results = []
    for layer in range(n_layers):
        d = doc_embs[:, layer, :]
        q = q_embs[:, layer, :]
        for intervention in INTERVENTIONS:
            if intervention == "baseline":
                d_use, q_use = d, q
            elif intervention == "corpus_only_ZCA":
                # Fit on docs only, apply to docs and queries
                W_fit = d
                d_use = zca_whiten(W_fit, d)
                q_use = zca_whiten(W_fit, q)
            elif intervention == "transductive_ZCA":
                # Fit on union of docs + queries
                W_fit = np.concatenate([d, q], axis=0)
                d_use = zca_whiten(W_fit, d)
                q_use = zca_whiten(W_fit, q)

            mrr, recall, _ = compute_mrr_recall(q_use, d_use, relevant_idx)
            for tercile_label in ["all"]:
                geom = compute_geometry(d_use)
                results.append({
                    "layer": layer,
                    "length_tercile": tercile_label,
                    "anisotropy_svd": geom["anisotropy_svd"],
                    "participation_ratio": geom["participation_ratio"],
                    "avg_cosine": geom["avg_cosine"],
                    "MRR@10": mrr,
                    "Recall@10": recall,
                    "model": config["name"],
                    "category": config["category"],
                    "corpus": corpus,
                    "query_format": query_format,
                    "is_ablation": config["is_ablation"],
                    "n_docs": len(doc_texts),
                    "n_queries": len(query_texts),
                    "intervention": intervention,
                })
    return results


# ============================================================
# CELL 10 — Main loop
# ============================================================
def main():
    # Sanity: check all corpus + query files exist
    missing = []
    for corpus, path in CORPUS_PATHS.items():
        if not path.exists():
            missing.append(("corpus", corpus, path))
    for corpus, path in QUERY_PATHS.items():
        if not path.exists():
            missing.append(("query", corpus, path))
    if missing:
        print("\n✗ Cannot run main loop — missing files:")
        for kind, corpus, path in missing:
            print(f"  {kind} | {corpus} | {path}")
        print("\nEdit CORPUS_PATHS / QUERY_PATHS in Cell 2 and re-run.")
        return

    # Load corpora / queries
    corpora = {c: pd.read_parquet(p) for c, p in CORPUS_PATHS.items()}
    queries = {c: pd.read_parquet(p) for c, p in QUERY_PATHS.items()}
    for c, df in corpora.items():
        print(f"  {c} corpus: {df.shape}")
    for c, df in queries.items():
        print(f"  {c} queries: {df.shape}")

    for config in MODELS_TO_EXTEND:
        out_path = OUTPUT_DIR / f"{config['name']}_validation500_layers.parquet"
        if out_path.exists():
            print(f"\n✓ {config['name']} — already done at {out_path}, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"Model: {config['name']} ({config['hf_id']})")
        print(f"{'=' * 60}")
        t0 = time.time()

        # Load model + tokenizer
        print(f"  Loading model (dtype={config['dtype']})...")
        tokenizer = AutoTokenizer.from_pretrained(
            config["hf_id"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(
            config["hf_id"], torch_dtype=config["dtype"],
            trust_remote_code=True,
        ).to(DEVICE)

        all_rows = []
        for corpus_name in CORPUS_PATHS.keys():
            for qf in QUERY_FORMATS:
                print(f"\n  {corpus_name} / {qf}")
                rows = evaluate_model_corpus(
                    config, tokenizer, model,
                    corpora[corpus_name], queries[corpus_name],
                    corpus_name, qf,
                )
                all_rows.extend(rows)

        # Save
        df_out = pd.DataFrame(all_rows)
        df_out.to_parquet(out_path)
        elapsed = time.time() - t0
        print(f"\n  ✓ Saved {out_path}  ({len(df_out)} rows, {elapsed:.0f}s)")

        # Cleanup
        del model, tokenizer
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


# Uncomment to run after verifying paths via discover_paths():
# main()

# ============================================================
# CELL 11 — Validation comparison: 100-doc vs 500-doc Spearman ρ
# ============================================================
def validation_comparison():
    """Compare final-layer baseline MRR@10 rankings between 100-doc
    (existing) and 500-doc (new) data per corpus × query_format.
    """
    from scipy.stats import spearmanr, kendalltau

    # Load the new 500-doc results
    new_dfs = []
    for config in MODELS_TO_EXTEND:
        p = OUTPUT_DIR / f"{config['name']}_validation500_layers.parquet"
        if p.exists():
            new_dfs.append(pd.read_parquet(p))
    if not new_dfs:
        print("✗ No 500-doc parquets found")
        return
    new500 = pd.concat(new_dfs, ignore_index=True)

    # Load the existing 100-doc results (v17 + v19 fresh runs)
    # === EDIT: path to existing combined parquet ===
    old100_path = DATA_DIR / "results" / "all_models_combined.parquet"
    if not old100_path.exists():
        print(f"✗ Existing 100-doc combined parquet not found at {old100_path}")
        return
    old100 = pd.read_parquet(old100_path)

    # Filter both to: baseline intervention, final layer (within each model),
    # length_tercile == 'all'
    for df in [new500, old100]:
        if "intervention" not in df.columns:
            df["intervention"] = "baseline"

    rows = []
    for config in MODELS_TO_EXTEND:
        m = config["name"]
        for corpus in CORPUS_PATHS.keys():
            for qf in QUERY_FORMATS:
                # Final layer per model
                old_sub = old100[(old100["model"] == m) &
                                 (old100["corpus"] == corpus) &
                                 (old100["query_format"] == qf) &
                                 (old100["intervention"] == "baseline") &
                                 (old100["length_tercile"] == "all")]
                new_sub = new500[(new500["model"] == m) &
                                 (new500["corpus"] == corpus) &
                                 (new500["query_format"] == qf) &
                                 (new500["intervention"] == "baseline") &
                                 (new500["length_tercile"] == "all")]
                if old_sub.empty or new_sub.empty:
                    rows.append({"model": m, "corpus": corpus,
                                 "query_format": qf,
                                 "old100_mrr": np.nan,
                                 "new500_mrr": np.nan,
                                 "delta": np.nan})
                    continue
                # Use max layer per side
                old_final = old_sub.loc[old_sub["layer"].idxmax()]
                new_final = new_sub.loc[new_sub["layer"].idxmax()]
                rows.append({"model": m, "corpus": corpus,
                             "query_format": qf,
                             "old100_mrr": float(old_final["MRR@10"]),
                             "new500_mrr": float(new_final["MRR@10"]),
                             "delta": float(new_final["MRR@10"] -
                                            old_final["MRR@10"])})

    cmp = pd.DataFrame(rows)
    print("\n=== Final-layer baseline MRR@10: 100-doc vs 500-doc ===")
    print(cmp.to_string(index=False))

    # Spearman ρ across the 20 (5 models × 4 conditions) data points
    valid = cmp.dropna()
    if len(valid) >= 3:
        rho, p = spearmanr(valid["old100_mrr"], valid["new500_mrr"])
        tau, _ = kendalltau(valid["old100_mrr"], valid["new500_mrr"])
        print(f"\nSpearman ρ (old vs new, all conditions pooled): "
              f"{rho:.4f} (p={p:.4g})")
        print(f"Kendall τ: {tau:.4f}")

    cmp.to_parquet(OUTPUT_DIR / "validation_summary_5models.parquet")
    print(f"\n✓ Saved to {OUTPUT_DIR}/validation_summary_5models.parquet")


# Uncomment after main() completes:
# validation_comparison()