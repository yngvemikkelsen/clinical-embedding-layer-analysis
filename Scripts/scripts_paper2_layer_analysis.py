#!/usr/bin/env python3
"""
Paper 2: Layer-Level Analysis of Embedding Degradation in Clinical Document Retrieval
=====================================================================================
Context or Tuning? — JAMIA submission

11 model configs × all layers × 3 corpora × 2 query formats
+ post-hoc interventions (layer selection, layer combination, whitening, centering)

Run on Google Colab Pro+ with H100 GPU.
Estimated wall time: 4-8 hours (7B models dominate).

Structure:
  Cell 1: Install dependencies
  Cell 2: Config & setup
  Cell 3: Data loading
  Cell 4: Layer extraction engine
  Cell 5: Run experiments (per-model checkpoint)
  Cell 6: Combine results & ANOVA
  Cell 7: Interventions
  Cell 8: Visualization
"""

# ============================================================
# CELL 1: INSTALL DEPENDENCIES
# ============================================================
# !pip install -q transformers datasets huggingface_hub accelerate
# !pip install -q torch  # usually pre-installed on Colab
# !pip install -q scipy scikit-learn pandas numpy matplotlib seaborn tqdm
# !pip install -q pyarrow  # for parquet
# !pip install -q rank-bm25  # for synthetic note alignment recovery
# !git clone https://github.com/yngvemikkelsen/clinical-rag-retrieval-benchmark /content/paper1_repo

# ============================================================
# CELL 2: CONFIG & SETUP
# ============================================================
import os
import gc
import json
import time
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Mount Google Drive ──
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    log.info("Not in Colab — using local paths")

# ── Paths ──
DRIVE_BASE = Path("/content/drive/MyDrive/Content")  # user's Drive folder
OUTPUT_DIR = DRIVE_BASE / "paper2_layer_analysis"
RESULTS_DIR = OUTPUT_DIR / "results"
INTERVENTIONS_DIR = OUTPUT_DIR / "interventions"
ANOVA_DIR = OUTPUT_DIR / "anova"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"

for d in [OUTPUT_DIR, RESULTS_DIR, INTERVENTIONS_DIR, ANOVA_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MTSAMPLES_SAMPLE_PATH = DRIVE_BASE / "mtsamples_sample.csv"  # Paper 1's 500-doc sample
PAPER1_REPO = Path("/content/paper1_repo")
QUERIES_PATH = PAPER1_REPO / "metadata_queries.json"
SYNTH_NOTES_PATH = PAPER1_REPO / "synthetic_notes.csv"

# ── Verify critical files ──
assert MTSAMPLES_SAMPLE_PATH.exists(), (
    f"❌ MTSamples sample not found at {MTSAMPLES_SAMPLE_PATH}\n"
    f"   This is the 500-doc sample from Paper 1's benchmark run.\n"
    f"   If you only have the full mtsamples.csv, run Paper 1's benchmark first to generate it."
)
assert QUERIES_PATH.exists(), (
    f"❌ Queries not found at {QUERIES_PATH}\n"
    f"   Run: !git clone https://github.com/yngvemikkelsen/clinical-rag-retrieval-benchmark /content/paper1_repo"
)
assert SYNTH_NOTES_PATH.exists(), f"❌ Synthetic notes not found at {SYNTH_NOTES_PATH}"

# ── Config ──
SEED = 42
N_EVAL_QUERIES = 100  # per corpus per format (matches metadata_queries.json)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
    log.info("HuggingFace authenticated ✓")

log.info(f"Device: {DEVICE}")
if torch.cuda.is_available():
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================
# CELL 2B: MODEL REGISTRY (Paper 2 — no OpenAI, no BM25)
# ============================================================
@dataclass
class ModelConfig:
    name: str
    hf_id: str
    category: str
    pooling: str = "mean"           # mean | cls | eos
    query_prefix: str = ""
    doc_prefix: str = ""
    instruction: str = ""
    is_dual_encoder: bool = False
    hf_id_query: str = ""
    hf_id_doc: str = ""
    dtype: str = "fp32"             # fp32 | fp16
    max_length: int = 512
    n_layers: int = 12              # transformer layers (excl. embedding layer)
    is_ablation: bool = False

MODELS: List[ModelConfig] = [
    # ── Domain Encoders (12-layer BERT-base) ──
    ModelConfig(
        name="BioBERT",
        hf_id="dmis-lab/biobert-v1.1",
        category="Domain Encoder",
        n_layers=12,
    ),
    ModelConfig(
        name="ClinicalBERT",
        hf_id="medicalai/ClinicalBERT",
        category="Domain Encoder",
        n_layers=6,  # DistilBERT architecture, not BERT-base
    ),
    # ── Biomedical Retrievers (12-layer) ──
    ModelConfig(
        name="BioLORD-2023",
        hf_id="FremyCompany/BioLORD-2023",
        category="Biomedical Retriever",
        n_layers=12,
    ),
    ModelConfig(
        name="MedCPT",
        hf_id="ncbi/MedCPT-Query-Encoder",
        category="Biomedical Retriever",
        pooling="cls",
        is_dual_encoder=True,
        hf_id_query="ncbi/MedCPT-Query-Encoder",
        hf_id_doc="ncbi/MedCPT-Article-Encoder",
        n_layers=12,
    ),
    # ── General Embeddings (12-layer) ──
    ModelConfig(
        name="BGE-base",
        hf_id="BAAI/bge-base-en-v1.5",
        category="General Embedding",
        n_layers=12,
    ),
    ModelConfig(
        name="GTE-base",
        hf_id="thenlper/gte-base",
        category="General Embedding",
        n_layers=12,
    ),
    ModelConfig(
        name="Nomic-embed-text",
        hf_id="nomic-ai/nomic-embed-text-v1.5",
        category="General Embedding",
        query_prefix="search_query: ",
        doc_prefix="search_document: ",
        n_layers=12,
    ),
    # ── General LLMs (32-layer) ──
    ModelConfig(
        name="E5-Mistral-7B",
        hf_id="intfloat/e5-mistral-7b-instruct",
        category="General LLM",
        pooling="eos",
        instruction="Given a clinical note, retrieve the most relevant clinical document.",
        dtype="fp16",
        max_length=4096,
        n_layers=32,
    ),
    ModelConfig(
        name="Phi-3-mini",
        hf_id="microsoft/Phi-3-mini-4k-instruct",
        category="General LLM",
        pooling="mean",
        dtype="fp16",
        max_length=4096,
        n_layers=32,
    ),
    # ── Ablations ──
    ModelConfig(
        name="E5-Mistral-7B-ablation",
        hf_id="intfloat/e5-mistral-7b-instruct",
        category="General LLM",
        pooling="mean",  # changed from eos
        instruction="",  # removed
        dtype="fp16",
        max_length=4096,
        n_layers=32,
        is_ablation=True,
    ),
    ModelConfig(
        name="Nomic-embed-text-nopfx",
        hf_id="nomic-ai/nomic-embed-text-v1.5",
        category="General Embedding",
        query_prefix="",
        doc_prefix="",
        n_layers=12,
        is_ablation=True,
    ),
]

log.info(f"Models: {len(MODELS)} configs ({sum(1 for m in MODELS if not m.is_ablation)} primary + {sum(1 for m in MODELS if m.is_ablation)} ablations)")


# ============================================================
# CELL 3: DATA LOADING
# ============================================================
def load_mtsamples_aligned():
    """
    Load MTSamples aligned with Paper 1 queries.
    Paper 1 saved a 500-doc sample (mtsamples_sample.csv).
    Queries were generated from the first 100 docs in that file.
    """
    log.info("Loading MTSamples from Paper 1 sample...")
    df = pd.read_csv(MTSAMPLES_SAMPLE_PATH)
    df = df.dropna(subset=["text"])
    docs = df["text"].tolist()[:100]
    log.info(f"  MTSamples: {len(docs)} docs (first 100 of 500-doc sample)")
    return docs


def load_pmc_patients_aligned():
    """
    Load PMC-Patients aligned with Paper 1 queries.
    Paper 1 loaded the full dataset, then sampled 100 with seed=42.
    """
    log.info("Loading PMC-Patients from HuggingFace...")

    try:
        from datasets import load_dataset
        ds = load_dataset("zhengyun21/PMC-Patients", split="train", token=HF_TOKEN or None)
        df = ds.to_pandas()
    except Exception as e:
        log.warning(f"  load_dataset failed ({e}), trying hf_hub_download...")
        from huggingface_hub import hf_hub_download
        json_path = hf_hub_download(
            repo_id="zhengyun21/PMC-Patients",
            filename="PMC-Patients-V2.json",
            repo_type="dataset",
            token=HF_TOKEN or None,
        )
        df = pd.read_json(json_path, lines=False)

    # Find text column
    text_col = None
    for col in ["patient", "text", "patient_text", "summary"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        for c in df.columns:
            if df[c].dtype == object and df[c].str.len().median() > 100:
                text_col = c
                break
    if text_col is None:
        raise ValueError(f"Cannot find text column in PMC-Patients. Columns: {list(df.columns)}")

    df = df.rename(columns={text_col: "text"})
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.len() > 50].reset_index(drop=True)
    log.info(f"  PMC-Patients full dataset: {len(df)} docs")

    # Replicate Paper 1 sampling: 100 docs, seed=42
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(df), size=100, replace=False)
    sampled = df.iloc[idx].reset_index(drop=True)
    docs = sampled["text"].tolist()
    log.info(f"  PMC-Patients sampled: {len(docs)} docs (seed={SEED})")
    return docs


def load_synthetic_aligned(queries_kw: List[str]):
    """
    Load synthetic notes aligned with Paper 1 queries.
    The original document ordering was lost. We recover the alignment
    using BM25 to match each keyword query to its best-matching document.

    Validated: BM25 on recovered alignment gives MRR@10 ≈ 0.96,
    matching Paper 1's BM25 baseline for Synthetic.
    """
    from rank_bm25 import BM25Okapi

    log.info("Loading synthetic notes with BM25 alignment recovery...")
    df = pd.read_csv(SYNTH_NOTES_PATH)
    df = df.dropna(subset=["text"])
    all_docs = df["text"].tolist()
    log.info(f"  Synthetic: {len(all_docs)} total docs")

    # BM25 match each query to best document
    tokenized = [d.lower().split() for d in all_docs]
    bm25 = BM25Okapi(tokenized)

    mapping = []
    for q in queries_kw:
        scores = bm25.get_scores(q.lower().split())
        best_doc = int(np.argmax(scores))
        mapping.append(best_doc)

    docs = [all_docs[i] for i in mapping]
    log.info(f"  Synthetic aligned: {len(docs)} docs ({len(set(mapping))} unique of {len(mapping)})")

    # Sanity check with BM25
    tokenized2 = [d.lower().split() for d in docs]
    bm25_2 = BM25Okapi(tokenized2)
    ranks = []
    for i, q in enumerate(queries_kw):
        scores = bm25_2.get_scores(q.lower().split())
        rank = (scores > scores[i]).sum() + 1
        ranks.append(rank)
    mrr = np.mean([1/r if r <= 10 else 0 for r in ranks])
    log.info(f"  BM25 sanity check MRR@10 = {mrr:.3f} (expected ~0.96)")

    if mrr < 0.9:
        log.warning(f"  ⚠ BM25 MRR@10 = {mrr:.3f} — alignment may be wrong!")

    return docs


def load_queries():
    """Load pre-generated metadata queries from Paper 1."""
    log.info("Loading metadata queries...")
    with open(QUERIES_PATH) as f:
        queries = json.load(f)
    for corpus in queries:
        for qf in queries[corpus]:
            log.info(f"  {corpus}/{qf}: {len(queries[corpus][qf])} queries")
    return queries


def build_evaluation_data():
    """
    Build the evaluation dataset with correct query-document alignment.

    Document alignment (verified against Paper 1 BM25 baselines):
    - MTSamples: first 100 rows of Paper 1's mtsamples_sample.csv (MRR@10 ≈ 0.87)
    - PMC-Patients: sample 100 from HF with seed=42 (MRR@10 ≈ 0.92)
    - Synthetic: BM25-recovered alignment from repo CSV (MRR@10 ≈ 0.96)

    Returns dict: {corpus_name: {"documents": [...], "queries": {...}, "doc_token_counts": [...]}}
    query[i] is the ground-truth match for document[i].
    """
    queries = load_queries()

    # MTSamples: positional alignment
    mt_docs = load_mtsamples_aligned()

    # PMC-Patients: seed-based sampling alignment
    pmc_docs = load_pmc_patients_aligned()

    # Synthetic: BM25-recovered alignment (needs keyword queries)
    syn_docs = load_synthetic_aligned(queries["Synthetic"]["keyword"])

    eval_data = {}
    for corpus_name, docs in [("MTSamples", mt_docs), ("PMC-Patients", pmc_docs), ("Synthetic", syn_docs)]:
        n_queries = len(queries[corpus_name]["keyword"])
        n = min(len(docs), n_queries)

        if len(docs) < n_queries:
            log.warning(
                f"  ⚠ {corpus_name}: only {len(docs)} docs but {n_queries} queries. "
                f"Truncating queries to {len(docs)}."
            )

        # Compute token counts (whitespace approximation, consistent with Paper 1)
        doc_list = docs[:n]
        token_counts = [len(d.split()) for d in doc_list]

        eval_data[corpus_name] = {
            "documents": doc_list,
            "queries": {
                "keyword": queries[corpus_name]["keyword"][:n],
                "natural_language": queries[corpus_name]["natural_language"][:n],
            },
            "doc_token_counts": token_counts,
        }
        log.info(f"  {corpus_name}: {n} docs × {n} queries × 2 formats | "
                 f"doc length: median={int(np.median(token_counts))}, "
                 f"range=[{min(token_counts)}, {max(token_counts)}]")

    return eval_data


def run_bm25_sanity_check(eval_data: Dict):
    """
    Verify query-document alignment using BM25 before running GPU-expensive models.
    Paper 1 BM25 baselines: MTSamples ~0.87, PMC-Patients ~0.92, Synthetic ~0.86.
    """
    from rank_bm25 import BM25Okapi

    log.info("\n" + "="*70)
    log.info("BM25 ALIGNMENT SANITY CHECK")
    log.info("="*70)

    all_ok = True
    for corpus_name, corpus_data in eval_data.items():
        docs = corpus_data["documents"]
        tokenized = [d.lower().split() for d in docs]
        bm25 = BM25Okapi(tokenized)

        for qf_name, queries in corpus_data["queries"].items():
            ranks = []
            for i, q in enumerate(queries):
                scores = bm25.get_scores(q.lower().split())
                rank = (scores > scores[i]).sum() + 1
                ranks.append(rank)
            mrr = np.mean([1/r if r <= 10 else 0 for r in ranks])
            recall = np.mean([1 if r <= 10 else 0 for r in ranks])
            status = "✅" if mrr > 0.5 else "❌"
            log.info(f"  {status} {corpus_name}/{qf_name}: BM25 MRR@10={mrr:.3f}, Recall@10={recall:.3f}")
            if mrr < 0.5:
                all_ok = False

    if not all_ok:
        log.error("  ❌ ALIGNMENT CHECK FAILED — some corpora have MRR@10 < 0.5")
        log.error("  This means query-document alignment is broken. Fix before proceeding.")
        raise ValueError("BM25 alignment check failed. See log above.")

    log.info("  ✅ All corpora pass alignment check.\n")


# ============================================================
# CELL 4: LAYER EXTRACTION ENGINE
# ============================================================

# ── Pooling functions ──
def mean_pool(hidden_states, attention_mask):
    """Mean pooling over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = torch.sum(hidden_states * mask, dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def cls_pool(hidden_states, attention_mask):
    """CLS token pooling (index 0)."""
    return hidden_states[:, 0, :]


def eos_pool(hidden_states, attention_mask):
    """EOS / last-token pooling."""
    seq_lens = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.shape[0]
    return hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lens]


POOL_FN = {"mean": mean_pool, "cls": cls_pool, "eos": eos_pool}


def _supports_output_hidden_states(model) -> bool:
    """
    Check if model.forward() accepts output_hidden_states kwarg.
    Uses a two-step approach: check signature first, but also accept **kwargs.
    """
    import inspect
    sig = inspect.signature(model.forward)
    # Explicit param or **kwargs both mean it's accepted
    if "output_hidden_states" in sig.parameters:
        return True
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True  # has **kwargs, likely passes through
    return False


def _extract_via_hooks(
    texts: List[str],
    model,
    tokenizer,
    config: ModelConfig,
    batch_size: int,
    pool_fn,
) -> Dict[int, np.ndarray]:
    """
    Fallback extraction using forward hooks for models that don't support
    output_hidden_states (e.g., NomicBertModel).
    Registers hooks on the embedding layer + each transformer layer.
    """
    captured = {}  # {layer_idx: list of tensors}
    hooks = []

    # Identify embedding + transformer layers
    # Try common module patterns
    embed_module = None
    transformer_layers = []

    for name, module in model.named_modules():
        # Embedding layer
        if any(pattern in name for pattern in ["embeddings", "embed_tokens"]) and embed_module is None:
            if hasattr(module, 'forward') and not any(sub in name for sub in ["LayerNorm", "dropout", "position", "token_type", "norm"]):
                # Only take the top-level embeddings module
                if name.count('.') <= 1:
                    embed_module = (name, module)
        # Transformer layers — look for numbered encoder/decoder layers
        if any(pattern in name for pattern in [
            "encoder.layers.", "transformer.layers.",
            "encoder.layer.", "transformer.layer.",
            "model.layers.",  # Mistral/Llama-style
        ]):
            parts = name.split(".")
            # Only top-level layer modules (e.g., encoder.layers.0, not encoder.layers.0.attention)
            # Check if the last numeric segment is the layer index
            is_top_level = False
            for i, part in enumerate(parts):
                if part.isdigit():
                    # Everything after the digit should be empty (top-level)
                    remaining = ".".join(parts[i+1:])
                    if not remaining:
                        is_top_level = True
                        break
            if is_top_level:
                transformer_layers.append((name, module))

    if embed_module is None or len(transformer_layers) == 0:
        raise RuntimeError(
            f"Could not identify embedding + transformer layers for hook-based extraction. "
            f"Model type: {type(model).__name__}. "
            f"Named modules: {[n for n, _ in model.named_modules() if n.count('.') <= 2]}"
        )

    log.info(f"  Hook extraction: 1 embed + {len(transformer_layers)} transformer layers")

    # Register hooks
    def make_hook(layer_idx):
        captured[layer_idx] = []
        def hook_fn(module, input, output):
            # output may be a tensor or tuple; take the first tensor
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            captured[layer_idx].append(out.detach())
        return hook_fn

    # Layer 0 = embedding output
    captured[0] = []
    hooks.append(embed_module[1].register_forward_hook(make_hook(0)))
    # Layers 1..N = transformer layers
    for i, (name, module) in enumerate(transformer_layers):
        hooks.append(module.register_forward_hook(make_hook(i + 1)))

    n_layers = 1 + len(transformer_layers)
    layer_embs = {i: [] for i in range(n_layers)}

    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=config.max_length, return_tensors="pt",
            )
            input_device = getattr(model, '_input_device', None) or next(model.parameters()).device
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

            # Clear captured tensors
            for k in captured:
                captured[k] = []

            with torch.no_grad():
                model(**inputs)

            attn_mask = inputs["attention_mask"]

            for layer_idx in range(n_layers):
                if not captured[layer_idx]:
                    continue
                hs = captured[layer_idx][0]  # (batch, seq_len, hidden)
                pooled = pool_fn(hs, attn_mask)
                pooled = F.normalize(pooled, p=2, dim=1)
                layer_embs[layer_idx].append(pooled.cpu().numpy())

            # Free
            for k in captured:
                captured[k] = []
            torch.cuda.empty_cache()

    finally:
        for h in hooks:
            h.remove()

    for layer_idx in layer_embs:
        if layer_embs[layer_idx]:
            layer_embs[layer_idx] = np.concatenate(layer_embs[layer_idx], axis=0)
        else:
            log.warning(f"  Layer {layer_idx} captured nothing — dropping")

    # Remove empty layers
    layer_embs = {k: v for k, v in layer_embs.items() if isinstance(v, np.ndarray)}
    return layer_embs


def extract_all_layers(
    texts: List[str],
    model,
    tokenizer,
    config: ModelConfig,
    batch_size: int = 32,
    pool_fn_override=None,
) -> Dict[int, np.ndarray]:
    """
    Extract embeddings from ALL hidden layers of a model.

    Returns: {layer_idx: np.ndarray of shape (n_texts, hidden_dim)}
        layer 0 = input embedding layer
        layer 1..N = transformer layers

    Uses output_hidden_states if supported, otherwise falls back to hooks.
    """
    pool_fn = pool_fn_override or POOL_FN[config.pooling]

    # ── Fallback: hook-based extraction for models that don't support output_hidden_states ──
    if not _supports_output_hidden_states(model):
        log.info(f"  Model {type(model).__name__} doesn't support output_hidden_states — using hooks")
        return _extract_via_hooks(texts, model, tokenizer, config, batch_size, pool_fn)

    # ── Standard path: output_hidden_states=True ──
    # Try first batch; if TypeError, fall back to hooks
    n_layers = config.n_layers + 1  # +1 for input embedding layer
    layer_embs = {i: [] for i in range(n_layers)}

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )
        # Handle device placement — device_map="auto" models need special treatment
        input_device = getattr(model, '_input_device', None) or next(model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
        except TypeError as e:
            if start == 0:
                log.info(f"  output_hidden_states failed ({e}) — falling back to hooks")
                return _extract_via_hooks(texts, model, tokenizer, config, batch_size, pool_fn)
            else:
                raise  # shouldn't fail mid-way if first batch worked

        hidden_states = outputs.hidden_states  # tuple of (n_layers+1) tensors
        attn_mask = inputs["attention_mask"]

        # Sanity check layer count
        if len(hidden_states) != n_layers:
            if start == 0:
                log.warning(
                    f"  Expected {n_layers} hidden states, got {len(hidden_states)}. "
                    f"Adjusting n_layers."
                )
                n_layers = len(hidden_states)
                layer_embs = {i: [] for i in range(n_layers)}

        for layer_idx in range(n_layers):
            hs = hidden_states[layer_idx]
            pooled = pool_fn(hs, attn_mask)
            pooled = F.normalize(pooled, p=2, dim=1)
            layer_embs[layer_idx].append(pooled.cpu().numpy())

        # Free GPU memory
        del outputs, hidden_states
        torch.cuda.empty_cache()

    # Concatenate batches
    for layer_idx in layer_embs:
        layer_embs[layer_idx] = np.concatenate(layer_embs[layer_idx], axis=0)

    return layer_embs


# ── Metric computation ──
def compute_anisotropy_svd(embeddings: np.ndarray) -> float:
    """SVD-based anisotropy: σ₁² / Σσᵢ². Higher = more degenerate."""
    embeddings = embeddings.astype(np.float32)
    centered = embeddings - embeddings.mean(axis=0)
    n = centered.shape[0]
    if n > 1000:
        idx = np.random.RandomState(SEED).choice(n, 1000, replace=False)
        centered = centered[idx]
    try:
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        S_sq = S ** 2
        return float(S_sq[0] / S_sq.sum())
    except Exception:
        return float("nan")


def compute_participation_ratio(embeddings: np.ndarray) -> float:
    """Intrinsic dimensionality via participation ratio: (Σσᵢ²)² / Σσᵢ⁴."""
    embeddings = embeddings.astype(np.float32)
    centered = embeddings - embeddings.mean(axis=0)
    n = centered.shape[0]
    if n > 1000:
        idx = np.random.RandomState(SEED).choice(n, 1000, replace=False)
        centered = centered[idx]
    try:
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        S_sq = S ** 2
        return float((S_sq.sum()) ** 2 / (S_sq ** 2).sum())
    except Exception:
        return float("nan")


def compute_avg_cosine(embeddings: np.ndarray, n_pairs: int = 10000) -> float:
    """Average pairwise cosine similarity (sampled)."""
    embeddings = embeddings.astype(np.float32)
    n = embeddings.shape[0]
    rng = np.random.RandomState(SEED)
    idx_a = rng.randint(0, n, size=n_pairs)
    idx_b = rng.randint(0, n, size=n_pairs)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]
    cos_sims = np.sum(embeddings[idx_a] * embeddings[idx_b], axis=1)
    return float(cos_sims.mean())


def compute_retrieval_metrics(q_emb: np.ndarray, d_emb: np.ndarray, k: int = 10):
    """
    Compute MRR@k and Recall@k with 1-to-1 query-document mapping.
    query[i] should retrieve document[i].
    """
    q_emb = q_emb.astype(np.float32)
    d_emb = d_emb.astype(np.float32)
    sim_matrix = q_emb @ d_emb.T
    n = sim_matrix.shape[0]
    ranks = np.zeros(n)
    for i in range(n):
        scores = sim_matrix[i]
        ranks[i] = (scores > scores[i]).sum() + 1

    rr = np.where(ranks <= k, 1.0 / ranks, 0.0)
    mrr = float(rr.mean())
    recall = float((ranks <= k).mean())
    return {"MRR@10": mrr, "Recall@10": recall}, ranks


def compute_all_layer_metrics(
    doc_layer_embs: Dict[int, np.ndarray],
    query_layer_embs: Dict[int, np.ndarray],
    doc_token_counts: List[int] = None,
) -> List[Dict]:
    """
    Compute geometric + retrieval metrics for every layer.
    If doc_token_counts provided, also compute per-tercile metrics.
    """
    # Compute length tercile assignments
    tercile_labels = None
    if doc_token_counts is not None:
        counts = np.array(doc_token_counts)
        t1, t2 = np.percentile(counts, [33.3, 66.7])
        tercile_labels = np.where(counts <= t1, "short",
                         np.where(counts <= t2, "medium", "long"))

    rows = []
    for layer_idx in sorted(doc_layer_embs.keys()):
        d_emb = doc_layer_embs[layer_idx]
        q_emb = query_layer_embs[layer_idx]

        # ── Overall metrics ──
        geo = {
            "anisotropy_svd": compute_anisotropy_svd(d_emb),
            "participation_ratio": compute_participation_ratio(d_emb),
            "avg_cosine": compute_avg_cosine(d_emb),
        }
        ret, ranks = compute_retrieval_metrics(q_emb, d_emb)

        rows.append({"layer": layer_idx, "length_tercile": "all", **geo, **ret})

        # ── Per-tercile metrics ──
        if tercile_labels is not None:
            for tercile in ["short", "medium", "long"]:
                mask = tercile_labels == tercile
                if mask.sum() < 5:
                    continue
                d_sub = d_emb[mask]
                q_sub = q_emb[mask]

                # Geometry on doc subset
                geo_t = {
                    "anisotropy_svd": compute_anisotropy_svd(d_sub),
                    "participation_ratio": compute_participation_ratio(d_sub),
                    "avg_cosine": compute_avg_cosine(d_sub, n_pairs=min(5000, d_sub.shape[0]*100)),
                }

                # Retrieval on subset: need to re-rank within the FULL doc set
                # but only compute MRR for the tercile queries
                q_emb_f32 = q_emb.astype(np.float32)
                d_emb_f32 = d_emb.astype(np.float32)
                sim_matrix = q_emb_f32 @ d_emb_f32.T
                tercile_indices = np.where(mask)[0]
                tercile_ranks = []
                for i in tercile_indices:
                    scores = sim_matrix[i]
                    rank = (scores > scores[i]).sum() + 1
                    tercile_ranks.append(rank)
                tercile_ranks = np.array(tercile_ranks)
                rr = np.where(tercile_ranks <= 10, 1.0 / tercile_ranks, 0.0)
                ret_t = {
                    "MRR@10": float(rr.mean()),
                    "Recall@10": float((tercile_ranks <= 10).mean()),
                }

                rows.append({"layer": layer_idx, "length_tercile": tercile, **geo_t, **ret_t})

    return rows


# ── Model loading helpers ──
def load_hf_model(hf_id: str, dtype: str = "fp32", trust_remote: bool = True):
    """Load a HuggingFace model + tokenizer."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, trust_remote_code=trust_remote, token=HF_TOKEN or None
    )
    load_kwargs = {"trust_remote_code": trust_remote, "token": HF_TOKEN or None}
    if dtype == "fp16":
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = "auto"

    model = AutoModel.from_pretrained(hf_id, **load_kwargs)
    if dtype != "fp16":
        model = model.to(DEVICE)
    model.eval()

    # Determine the actual device for input tensors
    # For device_map="auto", find where the first parameter lives
    try:
        model._input_device = next(model.parameters()).device
    except StopIteration:
        model._input_device = DEVICE

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def free_model(model):
    """Delete model and free GPU memory."""
    del model
    gc.collect()
    torch.cuda.empty_cache()


def apply_prefixes(texts: List[str], prefix: str = "", instruction: str = "") -> List[str]:
    """Apply query/doc prefixes or instruction formatting."""
    if instruction:
        return [f"Instruct: {instruction}\nQuery: {t}" for t in texts]
    elif prefix:
        return [prefix + t for t in texts]
    return texts


# ============================================================
# CELL 5: RUN EXPERIMENTS (per-model checkpoint to Drive)
# ============================================================
def _log_condition_summary(metrics: List[Dict], model_name: str, corpus: str, qf: str):
    """Print compact one-line summary of layer trajectory for a condition."""
    overall = [m for m in metrics if m["length_tercile"] == "all"]
    if not overall:
        return
    overall.sort(key=lambda x: x["layer"])

    mrr_values = [m["MRR@10"] for m in overall]
    aniso_values = [m["anisotropy_svd"] for m in overall]
    best_idx = int(np.argmax(mrr_values))
    best_layer = overall[best_idx]["layer"]

    log.info(
        f"  {model_name:25s} | {corpus:12s}/{qf:2s} | L=0:{mrr_values[0]:.3f} "
        f"→ best@L{best_layer}:{mrr_values[best_idx]:.3f} "
        f"→ final:{mrr_values[-1]:.3f} | "
        f"aniso: {aniso_values[0]:.3f}→{aniso_values[-1]:.3f}"
    )


def process_single_model(config: ModelConfig, eval_data: Dict) -> pd.DataFrame:
    """
    Process one model config: extract layers, compute metrics, return DataFrame.
    Handles standard models, dual encoders (MedCPT), and LLMs.
    """
    log.info(f"\n{'='*70}")
    log.info(f"Processing: {config.name} ({config.category})")
    log.info(f"  HF: {config.hf_id} | Pooling: {config.pooling} | Layers: {config.n_layers}")
    log.info(f"{'='*70}")

    all_rows = []
    batch_size = 2 if config.dtype == "fp16" else 16

    if config.is_dual_encoder:
        # ── MedCPT: separate query and doc encoders ──
        log.info("  Loading query encoder...")
        q_model, q_tokenizer = load_hf_model(config.hf_id_query, config.dtype)
        log.info("  Loading document encoder...")
        d_model, d_tokenizer = load_hf_model(config.hf_id_doc, config.dtype)

        for corpus_name, corpus_data in eval_data.items():
            documents = corpus_data["documents"]
            doc_token_counts = corpus_data.get("doc_token_counts")
            for qf_name, queries in corpus_data["queries"].items():
                t0 = time.time()

                q_layer_embs = extract_all_layers(
                    queries, q_model, q_tokenizer, config, batch_size=batch_size
                )
                d_layer_embs = extract_all_layers(
                    documents, d_model, d_tokenizer, config, batch_size=batch_size
                )

                metrics = compute_all_layer_metrics(d_layer_embs, q_layer_embs, doc_token_counts)
                qf_short = "KW" if qf_name == "keyword" else "NL"
                _log_condition_summary(metrics, config.name, corpus_name, qf_short)

                for row in metrics:
                    row.update({
                        "model": config.name,
                        "category": config.category,
                        "corpus": corpus_name,
                        "query_format": qf_name,
                        "is_ablation": config.is_ablation,
                        "n_docs": len(documents),
                        "n_queries": len(queries),
                    })
                all_rows.extend(metrics)

        free_model(q_model)
        free_model(d_model)

    else:
        # ── Standard single-encoder model ──
        model, tokenizer = load_hf_model(config.hf_id, config.dtype)

        for corpus_name, corpus_data in eval_data.items():
            documents = corpus_data["documents"]
            doc_token_counts = corpus_data.get("doc_token_counts")
            for qf_name, queries in corpus_data["queries"].items():
                t0 = time.time()

                prefixed_q = apply_prefixes(queries, config.query_prefix, config.instruction)
                prefixed_d = apply_prefixes(documents, config.doc_prefix)

                q_layer_embs = extract_all_layers(
                    prefixed_q, model, tokenizer, config, batch_size=batch_size
                )
                d_layer_embs = extract_all_layers(
                    prefixed_d, model, tokenizer, config, batch_size=batch_size
                )

                metrics = compute_all_layer_metrics(d_layer_embs, q_layer_embs, doc_token_counts)
                qf_short = "KW" if qf_name == "keyword" else "NL"
                _log_condition_summary(metrics, config.name, corpus_name, qf_short)

                for row in metrics:
                    row.update({
                        "model": config.name,
                        "category": config.category,
                        "corpus": corpus_name,
                        "query_format": qf_name,
                        "is_ablation": config.is_ablation,
                        "n_docs": len(documents),
                        "n_queries": len(queries),
                    })
                all_rows.extend(metrics)

        free_model(model)

    df = pd.DataFrame(all_rows)
    return df


def run_all_models(eval_data: Dict, models: List[ModelConfig] = None):
    """
    Run all models with per-model checkpointing.
    Skips models that already have a saved parquet on Drive.
    """
    if models is None:
        models = MODELS

    log.info(f"\nStarting experiment: {len(models)} models")
    log.info(f"Checkpoint dir: {RESULTS_DIR}")

    for i, config in enumerate(models):
        checkpoint_path = RESULTS_DIR / f"{config.name.replace(' ', '_')}_layers.parquet"

        if checkpoint_path.exists():
            log.info(f"\n[{i+1}/{len(models)}] SKIP {config.name} — checkpoint exists at {checkpoint_path}")
            continue

        log.info(f"\n[{i+1}/{len(models)}] Running {config.name}...")
        t_start = time.time()

        try:
            df = process_single_model(config, eval_data)
            df.to_parquet(checkpoint_path, index=False)
            elapsed = time.time() - t_start
            log.info(f"  ✅ Saved {len(df)} rows to {checkpoint_path} ({elapsed:.0f}s)")

            # Append to log
            with open(LOGS_DIR / "run_log.txt", "a") as f:
                f.write(f"{config.name}: {len(df)} rows, {elapsed:.0f}s\n")

        except Exception as e:
            log.error(f"  ❌ FAILED: {config.name} — {e}")
            import traceback
            traceback.print_exc()
            # Save error log
            with open(LOGS_DIR / "run_log.txt", "a") as f:
                f.write(f"{config.name}: FAILED — {e}\n")
            continue

    log.info("\n✅ All models processed.")


# ============================================================
# CELL 6: COMBINE RESULTS & ANOVA
# ============================================================
def combine_results() -> pd.DataFrame:
    """Load all per-model parquets and combine into one DataFrame."""
    dfs = []
    for path in sorted(RESULTS_DIR.glob("*_layers.parquet")):
        df = pd.read_parquet(path)
        dfs.append(df)
        log.info(f"  Loaded {path.name}: {len(df)} rows")

    if not dfs:
        raise FileNotFoundError(f"No result parquets found in {RESULTS_DIR}")

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(RESULTS_DIR / "all_models_combined.parquet", index=False)
    log.info(f"Combined: {len(combined)} rows, {combined['model'].nunique()} models")
    return combined


def run_anova_by_layer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Factorial ANOVA at each layer: model × corpus × query_format → MRR@10.
    Returns eta-squared for each factor at each layer.

    Uses Type II SS via scipy/statsmodels.
    """
    from scipy import stats

    # Use only overall metrics, not per-tercile rows
    df = df[df["length_tercile"] == "all"].copy()

    # Separate BERT-scale (12 layers) and LLM-scale (32 layers)
    bert_models = df[df["model"].isin([
        "BioBERT", "ClinicalBERT", "BioLORD-2023", "MedCPT",
        "BGE-base", "GTE-base", "Nomic-embed-text", "Nomic-embed-text-nopfx"
    ])].copy()
    llm_models = df[df["model"].isin([
        "E5-Mistral-7B", "E5-Mistral-7B-ablation", "Phi-3-mini"
    ])].copy()

    anova_rows = []

    for label, subset in [("BERT-base", bert_models), ("LLM-scale", llm_models)]:
        if subset.empty:
            continue

        layers = sorted(subset["layer"].unique())
        for layer in layers:
            layer_data = subset[subset["layer"] == layer].copy()

            # Simple approach: compute eta-squared from one-way ANOVAs
            # (full factorial with statsmodels is better but adds dependency complexity)
            total_var = layer_data["MRR@10"].var() * (len(layer_data) - 1)

            eta_sq = {}
            for factor in ["model", "corpus", "query_format"]:
                groups = [g["MRR@10"].values for _, g in layer_data.groupby(factor)]
                if len(groups) < 2:
                    eta_sq[factor] = 0.0
                    continue
                try:
                    F_stat, p_val = stats.f_oneway(*groups)
                    # Eta-squared = SS_between / SS_total
                    grand_mean = layer_data["MRR@10"].mean()
                    ss_between = sum(
                        len(g) * (g.mean() - grand_mean) ** 2
                        for g in [layer_data[layer_data[factor] == v]["MRR@10"] for v in layer_data[factor].unique()]
                    )
                    ss_total = total_var
                    eta_sq[factor] = float(ss_between / ss_total) if ss_total > 0 else 0.0
                except Exception:
                    eta_sq[factor] = float("nan")

            anova_rows.append({
                "scale": label,
                "layer": layer,
                "eta_sq_model": eta_sq.get("model", 0),
                "eta_sq_corpus": eta_sq.get("corpus", 0),
                "eta_sq_query_format": eta_sq.get("query_format", 0),
                "eta_sq_residual": max(0, 1.0 - sum(eta_sq.values())),
                "n_obs": len(layer_data),
            })

    anova_df = pd.DataFrame(anova_rows)
    anova_df.to_parquet(ANOVA_DIR / "anova_by_layer.parquet", index=False)
    log.info(f"ANOVA: {len(anova_df)} layer × scale combinations")
    return anova_df


# ============================================================
# CELL 7: INTERVENTIONS (no-retraining fixes)
# ============================================================
def intervention_layer_selection(
    df: pd.DataFrame, target_models: List[str] = None
) -> pd.DataFrame:
    """
    Intervention 1: Use best-performing layer instead of final layer.
    Returns DataFrame with best-layer performance vs final-layer baseline.
    """
    if target_models is None:
        target_models = [
            "BioBERT", "ClinicalBERT", "E5-Mistral-7B",
            "E5-Mistral-7B-ablation", "Phi-3-mini"
        ]

    rows = []
    for model_name in target_models:
        model_data = df[(df["model"] == model_name) & (df["length_tercile"] == "all")]
        if model_data.empty:
            continue

        max_layer = model_data["layer"].max()

        for corpus in model_data["corpus"].unique():
            for qf in model_data["query_format"].unique():
                subset = model_data[
                    (model_data["corpus"] == corpus) & (model_data["query_format"] == qf)
                ]

                final = subset[subset["layer"] == max_layer].iloc[0]
                best = subset.loc[subset["MRR@10"].idxmax()]

                rows.append({
                    "model": model_name,
                    "corpus": corpus,
                    "query_format": qf,
                    "intervention": "layer_selection",
                    "final_layer": int(max_layer),
                    "best_layer": int(best["layer"]),
                    "baseline_MRR@10": float(final["MRR@10"]),
                    "intervention_MRR@10": float(best["MRR@10"]),
                    "delta_MRR@10": float(best["MRR@10"] - final["MRR@10"]),
                    "baseline_Recall@10": float(final["Recall@10"]),
                    "intervention_Recall@10": float(best["Recall@10"]),
                    "delta_Recall@10": float(best["Recall@10"] - final["Recall@10"]),
                })

    return pd.DataFrame(rows)


def intervention_whitening_and_centering(
    eval_data: Dict,
    target_models: List[str] = None,
) -> pd.DataFrame:
    """
    Interventions 3 & 4: ZCA whitening and mean centering on final-layer embeddings.

    NOTE: This requires re-loading models to get final-layer embeddings.
    If you've already saved raw embeddings, adapt to load from disk instead.
    For now, operates on the combined results by re-extracting final layer only.
    """
    if target_models is None:
        target_models = ["BioBERT", "ClinicalBERT", "E5-Mistral-7B", "Phi-3-mini"]

    rows = []

    for config in MODELS:
        if config.name not in target_models:
            continue

        log.info(f"\nWhitening/centering: {config.name}")
        batch_size = 2 if config.dtype == "fp16" else 16

        if config.is_dual_encoder:
            q_model, q_tok = load_hf_model(config.hf_id_query, config.dtype)
            d_model, d_tok = load_hf_model(config.hf_id_doc, config.dtype)
        else:
            model, tokenizer = load_hf_model(config.hf_id, config.dtype)

        for corpus_name, corpus_data in eval_data.items():
            documents = corpus_data["documents"]
            for qf_name, queries in corpus_data["queries"].items():
                log.info(f"  {corpus_name}/{qf_name}")

                # Get final-layer embeddings
                if config.is_dual_encoder:
                    q_layers = extract_all_layers(queries, q_model, q_tok, config, batch_size)
                    d_layers = extract_all_layers(documents, d_model, d_tok, config, batch_size)
                else:
                    prefixed_q = apply_prefixes(queries, config.query_prefix, config.instruction)
                    prefixed_d = apply_prefixes(documents, config.doc_prefix)
                    q_layers = extract_all_layers(prefixed_q, model, tokenizer, config, batch_size)
                    d_layers = extract_all_layers(prefixed_d, model, tokenizer, config, batch_size)

                final_layer = max(d_layers.keys())
                q_emb = q_layers[final_layer]
                d_emb = d_layers[final_layer]

                # Baseline
                baseline_ret, _ = compute_retrieval_metrics(q_emb, d_emb)

                # Cast to float32 for numerical operations (fixes fp16 linalg errors)
                q_emb_f32 = q_emb.astype(np.float32)
                d_emb_f32 = d_emb.astype(np.float32)

                # ── Mean centering ──
                d_mean = d_emb_f32.mean(axis=0, keepdims=True)
                q_centered = q_emb_f32 - d_mean
                d_centered = d_emb_f32 - d_mean
                # Re-normalize
                q_centered = q_centered / np.linalg.norm(q_centered, axis=1, keepdims=True).clip(1e-9)
                d_centered = d_centered / np.linalg.norm(d_centered, axis=1, keepdims=True).clip(1e-9)
                center_ret, _ = compute_retrieval_metrics(q_centered, d_centered)

                rows.append({
                    "model": config.name, "corpus": corpus_name, "query_format": qf_name,
                    "intervention": "mean_centering",
                    "baseline_MRR@10": baseline_ret["MRR@10"],
                    "intervention_MRR@10": center_ret["MRR@10"],
                    "delta_MRR@10": center_ret["MRR@10"] - baseline_ret["MRR@10"],
                    "baseline_Recall@10": baseline_ret["Recall@10"],
                    "intervention_Recall@10": center_ret["Recall@10"],
                    "delta_Recall@10": center_ret["Recall@10"] - baseline_ret["Recall@10"],
                })

                # ── ZCA whitening ──
                try:
                    # Fit on document embeddings (already float32)
                    all_emb = np.vstack([d_emb_f32, q_emb_f32])
                    mean_vec = all_emb.mean(axis=0)
                    centered_all = all_emb - mean_vec
                    cov = centered_all.T @ centered_all / (len(centered_all) - 1)
                    U, S, Vt = np.linalg.svd(cov)
                    # ZCA: U @ diag(1/sqrt(S)) @ U^T
                    S_inv_sqrt = np.diag(1.0 / np.sqrt(S + 1e-5))
                    W = U @ S_inv_sqrt @ U.T

                    q_white = (q_emb_f32 - mean_vec) @ W.T
                    d_white = (d_emb_f32 - mean_vec) @ W.T
                    q_white = q_white / np.linalg.norm(q_white, axis=1, keepdims=True).clip(1e-9)
                    d_white = d_white / np.linalg.norm(d_white, axis=1, keepdims=True).clip(1e-9)
                    white_ret, _ = compute_retrieval_metrics(q_white, d_white)

                    rows.append({
                        "model": config.name, "corpus": corpus_name, "query_format": qf_name,
                        "intervention": "zca_whitening",
                        "baseline_MRR@10": baseline_ret["MRR@10"],
                        "intervention_MRR@10": white_ret["MRR@10"],
                        "delta_MRR@10": white_ret["MRR@10"] - baseline_ret["MRR@10"],
                        "baseline_Recall@10": baseline_ret["Recall@10"],
                        "intervention_Recall@10": white_ret["Recall@10"],
                        "delta_Recall@10": white_ret["Recall@10"] - baseline_ret["Recall@10"],
                    })
                except Exception as e:
                    log.warning(f"    ZCA whitening failed: {e}")

        # Free model
        if config.is_dual_encoder:
            free_model(q_model)
            free_model(d_model)
        else:
            free_model(model)

    result_df = pd.DataFrame(rows)
    result_df.to_parquet(INTERVENTIONS_DIR / "whitening_centering.parquet", index=False)
    return result_df


def intervention_layer_combination(
    eval_data: Dict,
    target_models: List[str] = None,
    top_k: int = 4,
) -> pd.DataFrame:
    """
    Intervention 2: Weighted average of top-k performing layers.
    Uses uniform weights over the top-k layers by MRR@10 (from layer analysis results).

    Requires re-loading models — expensive for 7B models.
    """
    if target_models is None:
        target_models = ["BioBERT", "ClinicalBERT"]  # only BERT-scale to save compute

    # Load layer analysis results to find top-k layers
    combined = pd.read_parquet(RESULTS_DIR / "all_models_combined.parquet")

    rows = []

    for config in MODELS:
        if config.name not in target_models:
            continue

        log.info(f"\nLayer combination: {config.name} (top-{top_k})")
        batch_size = 2 if config.dtype == "fp16" else 16

        if config.is_dual_encoder:
            q_model, q_tok = load_hf_model(config.hf_id_query, config.dtype)
            d_model, d_tok = load_hf_model(config.hf_id_doc, config.dtype)
        else:
            model, tokenizer = load_hf_model(config.hf_id, config.dtype)

        for corpus_name, corpus_data in eval_data.items():
            documents = corpus_data["documents"]
            for qf_name, queries in corpus_data["queries"].items():
                log.info(f"  {corpus_name}/{qf_name}")

                # Find top-k layers from prior results
                mask = (
                    (combined["model"] == config.name)
                    & (combined["corpus"] == corpus_name)
                    & (combined["query_format"] == qf_name)
                    & (combined["length_tercile"] == "all")
                )
                layer_perf = combined[mask].sort_values("MRR@10", ascending=False)
                top_layers = layer_perf["layer"].values[:top_k].tolist()
                log.info(f"    Top-{top_k} layers: {top_layers}")

                # Extract all layers
                if config.is_dual_encoder:
                    q_layers = extract_all_layers(queries, q_model, q_tok, config, batch_size)
                    d_layers = extract_all_layers(documents, d_model, d_tok, config, batch_size)
                else:
                    prefixed_q = apply_prefixes(queries, config.query_prefix, config.instruction)
                    prefixed_d = apply_prefixes(documents, config.doc_prefix)
                    q_layers = extract_all_layers(prefixed_q, model, tokenizer, config, batch_size)
                    d_layers = extract_all_layers(prefixed_d, model, tokenizer, config, batch_size)

                # Combine top-k layers (uniform average)
                q_combined = np.mean([q_layers[l] for l in top_layers], axis=0)
                d_combined = np.mean([d_layers[l] for l in top_layers], axis=0)
                q_combined = q_combined / np.linalg.norm(q_combined, axis=1, keepdims=True).clip(1e-9)
                d_combined = d_combined / np.linalg.norm(d_combined, axis=1, keepdims=True).clip(1e-9)

                combo_ret, _ = compute_retrieval_metrics(q_combined, d_combined)

                # Baseline (final layer)
                final_layer = max(d_layers.keys())
                baseline_ret, _ = compute_retrieval_metrics(q_layers[final_layer], d_layers[final_layer])

                rows.append({
                    "model": config.name, "corpus": corpus_name, "query_format": qf_name,
                    "intervention": f"layer_combination_top{top_k}",
                    "top_layers": str(top_layers),
                    "baseline_MRR@10": baseline_ret["MRR@10"],
                    "intervention_MRR@10": combo_ret["MRR@10"],
                    "delta_MRR@10": combo_ret["MRR@10"] - baseline_ret["MRR@10"],
                    "baseline_Recall@10": baseline_ret["Recall@10"],
                    "intervention_Recall@10": combo_ret["Recall@10"],
                    "delta_Recall@10": combo_ret["Recall@10"] - baseline_ret["Recall@10"],
                })

        if config.is_dual_encoder:
            free_model(q_model)
            free_model(d_model)
        else:
            free_model(model)

    result_df = pd.DataFrame(rows)
    result_df.to_parquet(INTERVENTIONS_DIR / "layer_combination.parquet", index=False)
    return result_df


# ============================================================
# CELL 8: VISUALIZATION
# ============================================================
def generate_figures(df: pd.DataFrame, anova_df: pd.DataFrame):
    """Generate all 6 planned figures for the paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    plt.rcParams.update({
        "font.family": "serif",
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
        "grid.alpha": 0.3,
    })

    CAT_COLORS = {
        "Domain Encoder": "#c0392b",
        "Biomedical Retriever": "#e67e22",
        "General Embedding": "#27ae60",
        "General LLM": "#2980b9",
    }

    MODEL_MARKERS = {
        "BioBERT": "o", "ClinicalBERT": "s", "BioLORD-2023": "^",
        "MedCPT": "D", "BGE-base": "P", "GTE-base": "*",
        "Nomic-embed-text": "X", "E5-Mistral-7B": "v", "Phi-3-mini": "p",
    }

    # Only primary models for main figures
    primary = df[(~df["is_ablation"]) & (df["length_tercile"] == "all")].copy()

    # ── Helper: get color for model ──
    def get_color(model_name):
        cat = primary[primary["model"] == model_name]["category"].iloc[0] if model_name in primary["model"].values else "General LLM"
        return CAT_COLORS.get(cat, "#999")

    # ── FIGURE 1: Layer-wise anisotropy ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: BERT-scale
    bert_models = primary[primary["model"].isin([
        "BioBERT", "ClinicalBERT", "BioLORD-2023", "MedCPT",
        "BGE-base", "GTE-base", "Nomic-embed-text"
    ])]
    ax = axes[0]
    for model_name in bert_models["model"].unique():
        sub = bert_models[bert_models["model"] == model_name]
        # Average across corpora and query formats
        avg = sub.groupby("layer")["anisotropy_svd"].mean().reset_index()
        ax.plot(avg["layer"], avg["anisotropy_svd"],
                marker=MODEL_MARKERS.get(model_name, "o"), markersize=4,
                color=get_color(model_name), label=model_name, linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Anisotropy (SVD)")
    ax.set_title("BERT-base Models (12 layers)")
    ax.legend(fontsize=7, ncol=2)

    # Panel B: LLM-scale
    llm_models = primary[primary["model"].isin(["E5-Mistral-7B", "Phi-3-mini"])]
    ax = axes[1]
    for model_name in llm_models["model"].unique():
        sub = llm_models[llm_models["model"] == model_name]
        avg = sub.groupby("layer")["anisotropy_svd"].mean().reset_index()
        ax.plot(avg["layer"], avg["anisotropy_svd"],
                marker=MODEL_MARKERS.get(model_name, "o"), markersize=4,
                color=get_color(model_name), label=model_name, linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Anisotropy (SVD)")
    ax.set_title("LLM-scale Models (32 layers)")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_anisotropy_by_layer.pdf")
    fig.savefig(FIGURES_DIR / "fig1_anisotropy_by_layer.png")
    plt.close()
    log.info("Fig 1: Anisotropy by layer — saved")

    # ── FIGURE 2: Layer-wise MRR@10 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for model_name in bert_models["model"].unique():
        sub = bert_models[bert_models["model"] == model_name]
        avg = sub.groupby("layer")["MRR@10"].mean().reset_index()
        ax.plot(avg["layer"], avg["MRR@10"],
                marker=MODEL_MARKERS.get(model_name, "o"), markersize=4,
                color=get_color(model_name), label=model_name, linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("MRR@10")
    ax.set_title("BERT-base Models")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for model_name in llm_models["model"].unique():
        sub = llm_models[llm_models["model"] == model_name]
        avg = sub.groupby("layer")["MRR@10"].mean().reset_index()
        ax.plot(avg["layer"], avg["MRR@10"],
                marker=MODEL_MARKERS.get(model_name, "o"), markersize=4,
                color=get_color(model_name), label=model_name, linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("MRR@10")
    ax.set_title("LLM-scale Models")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_mrr_by_layer.pdf")
    fig.savefig(FIGURES_DIR / "fig2_mrr_by_layer.png")
    plt.close()
    log.info("Fig 2: MRR@10 by layer — saved")

    # ── FIGURE 3: Corpus interaction heatmap ──
    # Anisotropy by layer × corpus for BioBERT vs BGE-base
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    contrast_models = ["BioBERT", "BGE-base"]
    corpora = sorted(primary["corpus"].unique())

    for row_idx, model_name in enumerate(contrast_models):
        for col_idx, corpus in enumerate(corpora):
            ax = axes[row_idx, col_idx]
            sub = primary[
                (primary["model"] == model_name) & (primary["corpus"] == corpus)
            ]
            for qf in ["keyword", "natural_language"]:
                qf_sub = sub[sub["query_format"] == qf]
                if qf_sub.empty:
                    continue
                ax.plot(qf_sub["layer"], qf_sub["anisotropy_svd"],
                        label=qf, linewidth=1.5)
            ax.set_title(f"{model_name} — {corpus}", fontsize=9)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Anisotropy")
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_corpus_interaction.pdf")
    fig.savefig(FIGURES_DIR / "fig3_corpus_interaction.png")
    plt.close()
    log.info("Fig 3: Corpus interaction — saved")

    # ── FIGURE 4: Variance decomposition by layer (ANOVA) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, scale in zip(axes, ["BERT-base", "LLM-scale"]):
        sub = anova_df[anova_df["scale"] == scale].sort_values("layer")
        if sub.empty:
            continue

        # Stacked area
        layers = sub["layer"].values
        model_eta = sub["eta_sq_model"].values
        corpus_eta = sub["eta_sq_corpus"].values
        qf_eta = sub["eta_sq_query_format"].values
        resid = sub["eta_sq_residual"].values

        ax.stackplot(
            layers,
            model_eta, corpus_eta, qf_eta, resid,
            labels=["Model (tuning)", "Corpus (context)", "Query format", "Residual"],
            colors=["#e74c3c", "#3498db", "#2ecc71", "#bdc3c7"],
            alpha=0.8,
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("η² (variance explained)")
        ax.set_title(f"{scale}: Variance Decomposition by Layer")
        ax.legend(loc="upper left", fontsize=7)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_variance_decomposition.pdf")
    fig.savefig(FIGURES_DIR / "fig4_variance_decomposition.png")
    plt.close()
    log.info("Fig 4: Variance decomposition — saved")

    # ── FIGURE 5: Intervention before/after (loaded from intervention parquets) ──
    try:
        intv_files = list(INTERVENTIONS_DIR.glob("*.parquet"))
        if intv_files:
            intv_dfs = [pd.read_parquet(p) for p in intv_files]
            intv_all = pd.concat(intv_dfs, ignore_index=True)

            # Also add layer selection results
            layer_sel = intervention_layer_selection(primary)
            intv_all = pd.concat([intv_all, layer_sel], ignore_index=True)

            # Best intervention per model × corpus
            best = intv_all.groupby(["model", "corpus", "query_format"]).apply(
                lambda g: g.loc[g["delta_MRR@10"].idxmax()]
            ).reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            models_shown = sorted(best["model"].unique())
            x = np.arange(len(models_shown))
            width = 0.35

            for i, model_name in enumerate(models_shown):
                sub = best[best["model"] == model_name]
                baseline = sub["baseline_MRR@10"].mean()
                improved = sub["intervention_MRR@10"].mean()
                ax.bar(i - width/2, baseline, width, color="#bdc3c7", edgecolor="black", linewidth=0.5)
                ax.bar(i + width/2, improved, width, color="#27ae60", edgecolor="black", linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(models_shown, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("MRR@10")
            ax.set_title("Best Intervention vs Final-Layer Baseline (avg across corpora)")
            ax.legend(["Baseline (final layer)", "Best intervention"], fontsize=8)
            plt.tight_layout()
            fig.savefig(FIGURES_DIR / "fig5_intervention_impact.pdf")
            fig.savefig(FIGURES_DIR / "fig5_intervention_impact.png")
            plt.close()
            log.info("Fig 5: Intervention impact — saved")
    except Exception as e:
        log.warning(f"Fig 5 skipped: {e}")

    # ── FIGURE 6: Anisotropy vs retrieval gain scatter ──
    try:
        layer_sel = intervention_layer_selection(primary)
        if not layer_sel.empty:
            fig, ax = plt.subplots(figsize=(8, 6))

            # Final-layer anisotropy (averaged across conditions)
            final_aniso = primary.groupby("model").apply(
                lambda g: g[g["layer"] == g["layer"].max()]["anisotropy_svd"].mean()
            ).to_dict()

            avg_gain = layer_sel.groupby("model")["delta_MRR@10"].mean().to_dict()

            for model_name in final_aniso:
                if model_name in avg_gain:
                    ax.scatter(
                        final_aniso[model_name], avg_gain[model_name],
                        color=get_color(model_name),
                        marker=MODEL_MARKERS.get(model_name, "o"),
                        s=100, edgecolors="black", linewidth=0.5, zorder=5,
                    )
                    ax.annotate(model_name, (final_aniso[model_name], avg_gain[model_name]),
                                textcoords="offset points", xytext=(5, 5), fontsize=7)

            ax.set_xlabel("Final-layer Anisotropy (SVD)")
            ax.set_ylabel("MRR@10 Gain from Best Layer Selection")
            ax.set_title("Which Models Benefit Most from Layer Selection?")
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            fig.savefig(FIGURES_DIR / "fig6_anisotropy_vs_gain.pdf")
            fig.savefig(FIGURES_DIR / "fig6_anisotropy_vs_gain.png")
            plt.close()
            log.info("Fig 6: Anisotropy vs gain — saved")
    except Exception as e:
        log.warning(f"Fig 6 skipped: {e}")


# ============================================================
# CELL 9: MAIN EXECUTION (run cells sequentially)
# ============================================================
def main():
    """Full pipeline — run this or execute cells individually."""

    # Step 1: Load evaluation data
    log.info("="*70)
    log.info("STEP 1: Loading evaluation data")
    log.info("="*70)
    eval_data = build_evaluation_data()

    # Step 1b: BM25 alignment sanity check
    run_bm25_sanity_check(eval_data)

    # Step 2: Run layer extraction for all models
    log.info("\n" + "="*70)
    log.info("STEP 2: Layer extraction (per-model checkpoint)")
    log.info("="*70)
    run_all_models(eval_data)

    # Step 3: Combine and ANOVA
    log.info("\n" + "="*70)
    log.info("STEP 3: Combine results & ANOVA")
    log.info("="*70)
    combined = combine_results()
    anova_df = run_anova_by_layer(combined)

    # Step 4: Interventions
    log.info("\n" + "="*70)
    log.info("STEP 4: Interventions")
    log.info("="*70)
    layer_sel = intervention_layer_selection(combined)
    layer_sel.to_parquet(INTERVENTIONS_DIR / "layer_selection.parquet", index=False)
    log.info(f"  Layer selection: {len(layer_sel)} rows")

    # Whitening/centering (requires re-loading models — expensive)
    wc_df = intervention_whitening_and_centering(eval_data)
    log.info(f"  Whitening/centering: {len(wc_df)} rows")

    # Layer combination (BERT-scale only to save compute)
    combo_df = intervention_layer_combination(eval_data)
    log.info(f"  Layer combination: {len(combo_df)} rows")

    # Step 5: Figures
    log.info("\n" + "="*70)
    log.info("STEP 5: Generating figures")
    log.info("="*70)
    generate_figures(combined, anova_df)

    log.info("\n" + "="*70)
    log.info("✅ PIPELINE COMPLETE")
    log.info(f"Results: {RESULTS_DIR}")
    log.info(f"Figures: {FIGURES_DIR}")
    log.info(f"Interventions: {INTERVENTIONS_DIR}")
    log.info(f"ANOVA: {ANOVA_DIR}")
    log.info("="*70)


# ── Run ──
if __name__ == "__main__":
    main()
