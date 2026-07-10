"""
Paper 12 Revision — Stream B: Decision 2 Implementation
========================================================
SELF-CONTAINED. Paste entire file into Colab.

Implements Decision 2: Reframe corpus-only ZCA as primary methodology
+ add cross-validation + add epsilon sensitivity sweep.

Scope: ALL 13 models (11 original Paper 12 panel + 2 new from Stream A).

Pipeline:
  1. Extract final-layer embeddings for all 13 models, save as .npz
  2. For each (model, corpus, query_format), compute:
     a. Baseline MRR@10 (no intervention)
     b. Transductive ZCA at eps=1e-5 (reproduces original Paper 12)
     c. Corpus-only ZCA at eps=1e-5 (PRIMARY per Decision 2)
  3. Epsilon sweep on corpus-only ZCA: eps ∈ {1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2}
  4. 5-fold cross-validation on corpus-only ZCA

Inputs (at /content/sample_data/):
  - mtsamples_sample.csv
  - metadata_queries.json
  - synthetic_notes.csv

Outputs (at /content/paper12_revision/stream_b/):
  - embeddings/{ModelName}.npz                       (intermediate, re-usable)
  - results/methodology_comparison.parquet           (transductive vs corpus-only)
  - results/epsilon_sensitivity.parquet              (eps sweep)
  - results/cross_validation.parquet                 (5-fold CV)

Sanity check runs FIRST. If any model tokenizer fails to load, aborts in
~2-3 minutes BEFORE committing to compute.

Runtime on RTX PRO 6000 Blackwell:
  - Embedding extraction: ~75-100 min total
  - Analyses: ~5 min
  - Total: ~80-105 min
"""

# ==CELL 1: INSTALL DEPENDENCIES (runs unconditionally)==
import subprocess
import sys

def _ensure_packages(packages):
    """Install packages programmatically. Works as .py file or Colab cell."""
    for pkg in packages:
        try:
            mod_name = pkg.replace("-", "_")
            __import__(mod_name)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure_packages([
    "rank-bm25",
    "sentencepiece",
    "pyarrow",
])
print("Dependencies ready.\n")


# ==CELL 2: IMPORTS, PATHS, CONFIG==
import os
import gc
import json
import time
import logging
import warnings
import inspect
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

INPUT_DIR = Path("/content/sample_data")
MTSAMPLES_SAMPLE_PATH = INPUT_DIR / "mtsamples_sample.csv"
QUERIES_PATH = INPUT_DIR / "metadata_queries.json"
SYNTH_NOTES_PATH = INPUT_DIR / "synthetic_notes.csv"

STREAM_B_DIR = Path("/content/paper12_revision/stream_b")
EMBEDDINGS_DIR = STREAM_B_DIR / "embeddings"
RESULTS_DIR = STREAM_B_DIR / "results"

for d in [STREAM_B_DIR, EMBEDDINGS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Files in {INPUT_DIR}:")
for f in sorted(INPUT_DIR.iterdir()):
    print(f"  {f.name}")

assert MTSAMPLES_SAMPLE_PATH.exists(), f"Missing: {MTSAMPLES_SAMPLE_PATH}"
assert QUERIES_PATH.exists(), f"Missing: {QUERIES_PATH}"
assert SYNTH_NOTES_PATH.exists(), f"Missing: {SYNTH_NOTES_PATH}"
print("\nAll input files present.\n")

SEED = 42
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ZCA epsilon values to test
EPSILON_VALUES = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
DEFAULT_EPSILON = 1e-5
N_FOLDS = 5

np.random.seed(SEED)
torch.manual_seed(SEED)

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
    log.info("HuggingFace authenticated")

log.info(f"Device: {DEVICE}")
if torch.cuda.is_available():
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ==CELL 3: ALL 13 MODEL CONFIGS==
@dataclass
class ModelConfig:
    name: str
    hf_id: str
    category: str
    pooling: str = "mean"
    query_prefix: str = ""
    doc_prefix: str = ""
    instruction: str = ""
    is_dual_encoder: bool = False
    hf_id_query: str = ""
    hf_id_doc: str = ""
    dtype: str = "fp32"
    max_length: int = 512
    n_layers: int = 12
    is_ablation: bool = False


MODELS: List[ModelConfig] = [
    # ── Original 11 models (verbatim from scripts_paper2_layer_analysis.py) ──
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
    ModelConfig(
        name="E5-Mistral-7B-ablation",
        hf_id="intfloat/e5-mistral-7b-instruct",
        category="General LLM",
        pooling="mean",   # ablation: changed from eos
        instruction="",   # ablation: removed
        dtype="fp16",
        max_length=4096,
        n_layers=32,
        is_ablation=True,
    ),
    ModelConfig(
        name="Nomic-embed-text-nopfx",
        hf_id="nomic-ai/nomic-embed-text-v1.5",
        category="General Embedding",
        query_prefix="",  # ablation: removed
        doc_prefix="",    # ablation: removed
        n_layers=12,
        is_ablation=True,
    ),
    # ── New models from Stream A ──
    ModelConfig(
        name="BERT-base-uncased",
        hf_id="bert-base-uncased",
        category="General Encoder",
        pooling="mean",
        n_layers=12,
        dtype="fp32",
        max_length=512,
    ),
    ModelConfig(
        name="BioMistral-7B",
        hf_id="BioMistral/BioMistral-7B",
        category="Domain LLM",
        pooling="mean",
        dtype="fp16",
        max_length=2048,
        n_layers=32,
    ),
]

log.info(f"Total models: {len(MODELS)}")
for m in MODELS:
    marker = " [ablation]" if m.is_ablation else ""
    dual = " [dual encoder]" if m.is_dual_encoder else ""
    log.info(f"  - {m.name} ({m.category}) | {m.dtype}{marker}{dual}")


# ==CELL 4: DATA LOADING (verbatim from Stream A)==
def load_mtsamples_aligned():
    log.info("Loading MTSamples...")
    df = pd.read_csv(MTSAMPLES_SAMPLE_PATH)
    df = df.dropna(subset=["text"])
    docs = df["text"].tolist()[:100]
    log.info(f"  MTSamples: {len(docs)} docs")
    return docs


def load_pmc_patients_aligned():
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
        raise ValueError(f"Cannot find text column. Columns: {list(df.columns)}")

    df = df.rename(columns={text_col: "text"})
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.len() > 50].reset_index(drop=True)
    log.info(f"  PMC-Patients full dataset: {len(df)} docs")

    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(df), size=100, replace=False)
    sampled = df.iloc[idx].reset_index(drop=True)
    docs = sampled["text"].tolist()
    log.info(f"  PMC-Patients sampled: {len(docs)} docs (seed={SEED})")
    return docs


def load_synthetic_aligned(queries_kw: List[str]):
    from rank_bm25 import BM25Okapi

    log.info("Loading synthetic notes with BM25 alignment recovery...")
    df = pd.read_csv(SYNTH_NOTES_PATH)
    df = df.dropna(subset=["text"])
    all_docs = df["text"].tolist()
    log.info(f"  Synthetic: {len(all_docs)} total docs")

    tokenized = [d.lower().split() for d in all_docs]
    bm25 = BM25Okapi(tokenized)

    mapping = []
    for q in queries_kw:
        scores = bm25.get_scores(q.lower().split())
        best_doc = int(np.argmax(scores))
        mapping.append(best_doc)

    docs = [all_docs[i] for i in mapping]
    log.info(f"  Synthetic aligned: {len(docs)} docs ({len(set(mapping))} unique of {len(mapping)})")
    return docs


def load_queries():
    log.info("Loading metadata queries...")
    with open(QUERIES_PATH) as f:
        queries = json.load(f)
    for corpus in queries:
        for qf in queries[corpus]:
            log.info(f"  {corpus}/{qf}: {len(queries[corpus][qf])} queries")
    return queries


def build_evaluation_data():
    queries = load_queries()
    mt_docs = load_mtsamples_aligned()
    pmc_docs = load_pmc_patients_aligned()
    syn_docs = load_synthetic_aligned(queries["Synthetic"]["keyword"])

    eval_data = {}
    for corpus_name, docs in [("MTSamples", mt_docs), ("PMC-Patients", pmc_docs), ("Synthetic", syn_docs)]:
        n_queries = len(queries[corpus_name]["keyword"])
        n = min(len(docs), n_queries)

        doc_list = docs[:n]
        eval_data[corpus_name] = {
            "documents": doc_list,
            "queries": {
                "keyword": queries[corpus_name]["keyword"][:n],
                "natural_language": queries[corpus_name]["natural_language"][:n],
            },
        }
        log.info(f"  {corpus_name}: {n} docs x {n} queries x 2 formats")

    return eval_data


# ==CELL 5: FINAL-LAYER EMBEDDING EXTRACTION==
def mean_pool(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = torch.sum(hidden_states * mask, dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def cls_pool(hidden_states, attention_mask):
    return hidden_states[:, 0, :]


def eos_pool(hidden_states, attention_mask):
    seq_lens = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.shape[0]
    return hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lens]


POOL_FN = {"mean": mean_pool, "cls": cls_pool, "eos": eos_pool}


def load_hf_model(hf_id: str, dtype: str = "fp32", trust_remote: bool = True):
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

    try:
        model._input_device = next(model.parameters()).device
    except StopIteration:
        model._input_device = DEVICE

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def apply_prefixes(texts: List[str], prefix: str = "", instruction: str = "") -> List[str]:
    if instruction:
        return [f"Instruct: {instruction}\nQuery: {t}" for t in texts]
    elif prefix:
        return [prefix + t for t in texts]
    return texts


def extract_final_layer(texts, model, tokenizer, config, batch_size=16):
    """Extract ONLY final-layer embeddings (faster than all-layer)."""
    pool_fn = POOL_FN[config.pooling]
    embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=config.max_length, return_tensors="pt"
        )
        input_device = getattr(model, '_input_device', None) or next(model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state
        attn_mask = inputs["attention_mask"]
        pooled = pool_fn(last_hidden, attn_mask)
        pooled = F.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.cpu().numpy().astype(np.float32))

        del outputs, last_hidden, pooled
        torch.cuda.empty_cache()

    return np.concatenate(embeddings, axis=0)


def extract_model_embeddings(config: ModelConfig, eval_data: Dict) -> Dict:
    """
    Extract final-layer embeddings for one model across all (corpus, query_format) conditions.
    Handles single-encoder and dual-encoder (MedCPT) cases.
    Returns dict: {corpus_qf_role: np.ndarray}
    """
    embeddings = {}
    batch_size = 2 if config.dtype == "fp16" else 16

    if config.is_dual_encoder:
        log.info(f"  [dual encoder] loading query encoder: {config.hf_id_query}")
        q_model, q_tokenizer = load_hf_model(config.hf_id_query, config.dtype)
        for corpus_name, corpus_data in eval_data.items():
            for qf_name, queries in corpus_data["queries"].items():
                prefixed_q = apply_prefixes(queries, config.query_prefix, config.instruction)
                q_emb = extract_final_layer(prefixed_q, q_model, q_tokenizer, config, batch_size)
                embeddings[f"{corpus_name}_{qf_name}_q"] = q_emb
                log.info(f"    {corpus_name}/{qf_name} queries: {q_emb.shape}")
        free_model(q_model)

        log.info(f"  [dual encoder] loading doc encoder: {config.hf_id_doc}")
        d_model, d_tokenizer = load_hf_model(config.hf_id_doc, config.dtype)
        for corpus_name, corpus_data in eval_data.items():
            documents = corpus_data["documents"]
            prefixed_d = apply_prefixes(documents, config.doc_prefix)
            d_emb = extract_final_layer(prefixed_d, d_model, d_tokenizer, config, batch_size)
            # Share docs across query formats
            for qf_name in corpus_data["queries"].keys():
                embeddings[f"{corpus_name}_{qf_name}_d"] = d_emb
            log.info(f"    {corpus_name} docs: {d_emb.shape}")
        free_model(d_model)
    else:
        log.info(f"  [single encoder] loading: {config.hf_id}")
        model, tokenizer = load_hf_model(config.hf_id, config.dtype)
        for corpus_name, corpus_data in eval_data.items():
            documents = corpus_data["documents"]
            prefixed_d = apply_prefixes(documents, config.doc_prefix)
            d_emb = extract_final_layer(prefixed_d, model, tokenizer, config, batch_size)
            for qf_name, queries in corpus_data["queries"].items():
                prefixed_q = apply_prefixes(queries, config.query_prefix, config.instruction)
                q_emb = extract_final_layer(prefixed_q, model, tokenizer, config, batch_size)
                embeddings[f"{corpus_name}_{qf_name}_q"] = q_emb
                embeddings[f"{corpus_name}_{qf_name}_d"] = d_emb
                log.info(f"    {corpus_name}/{qf_name}: q={q_emb.shape}, d={d_emb.shape}")
        free_model(model)

    return embeddings


def save_embeddings(model_name: str, embeddings: Dict):
    """Save extracted embeddings to .npz."""
    path = EMBEDDINGS_DIR / f"{model_name}.npz"
    np.savez_compressed(path, **embeddings)
    log.info(f"  Saved: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


def load_embeddings(model_name: str) -> Dict:
    """Load extracted embeddings from .npz."""
    path = EMBEDDINGS_DIR / f"{model_name}.npz"
    with np.load(path) as data:
        return {k: data[k].copy() for k in data.files}


# ==CELL 6: ZCA + RETRIEVAL METRICS==
def compute_mrr_recall(q_emb: np.ndarray, d_emb: np.ndarray, k: int = 10):
    """1-to-1 query-document mapping: query[i] should retrieve document[i]."""
    q_emb = q_emb.astype(np.float32)
    d_emb = d_emb.astype(np.float32)
    sim_matrix = q_emb @ d_emb.T
    n = sim_matrix.shape[0]
    ranks = np.zeros(n)
    for i in range(n):
        scores = sim_matrix[i]
        ranks[i] = (scores > scores[i]).sum() + 1
    rr = np.where(ranks <= k, 1.0 / ranks, 0.0)
    return float(rr.mean()), float((ranks <= k).mean()), ranks


def fit_zca_transductive(d_emb: np.ndarray, q_emb: np.ndarray, eps: float):
    """Fit ZCA on combined doc+query embeddings (original Paper 12 methodology)."""
    d_emb = d_emb.astype(np.float32)
    q_emb = q_emb.astype(np.float32)
    all_emb = np.vstack([d_emb, q_emb])
    mean_vec = all_emb.mean(axis=0)
    centered = all_emb - mean_vec
    cov = centered.T @ centered / (len(centered) - 1)
    U, S, Vt = np.linalg.svd(cov)
    S_inv_sqrt = np.diag(1.0 / np.sqrt(S + eps))
    W = U @ S_inv_sqrt @ U.T
    return W, mean_vec


def fit_zca_corpus_only(d_emb: np.ndarray, eps: float):
    """
    Fit ZCA on documents only (PRIMARY methodology per Decision 2).
    Addresses test-set contamination concern by not using query embeddings.
    """
    d_emb = d_emb.astype(np.float32)
    mean_vec = d_emb.mean(axis=0)
    centered = d_emb - mean_vec
    cov = centered.T @ centered / (len(centered) - 1)
    U, S, Vt = np.linalg.svd(cov)
    S_inv_sqrt = np.diag(1.0 / np.sqrt(S + eps))
    W = U @ S_inv_sqrt @ U.T
    return W, mean_vec


def apply_zca(emb: np.ndarray, W: np.ndarray, mean_vec: np.ndarray):
    """Apply ZCA transformation + L2-normalize."""
    emb_f32 = emb.astype(np.float32)
    transformed = (emb_f32 - mean_vec) @ W.T
    transformed = transformed / np.linalg.norm(transformed, axis=1, keepdims=True).clip(1e-9)
    return transformed


def methodology_comparison(model_name: str, embeddings: Dict, eval_data: Dict) -> List[Dict]:
    """
    For each (corpus, query_format):
      - Baseline (no intervention)
      - Transductive ZCA @ eps=1e-5 (original Paper 12)
      - Corpus-only ZCA @ eps=1e-5 (PRIMARY per Decision 2)
    """
    rows = []
    for corpus_name, corpus_data in eval_data.items():
        for qf_name in corpus_data["queries"].keys():
            q_emb = embeddings[f"{corpus_name}_{qf_name}_q"]
            d_emb = embeddings[f"{corpus_name}_{qf_name}_d"]

            # Baseline
            base_mrr, base_recall, _ = compute_mrr_recall(q_emb, d_emb)

            # Transductive ZCA
            try:
                W_t, mu_t = fit_zca_transductive(d_emb, q_emb, DEFAULT_EPSILON)
                q_t = apply_zca(q_emb, W_t, mu_t)
                d_t = apply_zca(d_emb, W_t, mu_t)
                trans_mrr, trans_recall, _ = compute_mrr_recall(q_t, d_t)
            except Exception as e:
                log.warning(f"    Transductive ZCA failed: {e}")
                trans_mrr, trans_recall = float("nan"), float("nan")

            # Corpus-only ZCA
            try:
                W_c, mu_c = fit_zca_corpus_only(d_emb, DEFAULT_EPSILON)
                q_c = apply_zca(q_emb, W_c, mu_c)
                d_c = apply_zca(d_emb, W_c, mu_c)
                co_mrr, co_recall, _ = compute_mrr_recall(q_c, d_c)
            except Exception as e:
                log.warning(f"    Corpus-only ZCA failed: {e}")
                co_mrr, co_recall = float("nan"), float("nan")

            rows.append({
                "model": model_name,
                "corpus": corpus_name,
                "query_format": qf_name,
                "baseline_MRR@10": base_mrr,
                "transductive_ZCA_MRR@10": trans_mrr,
                "corpus_only_ZCA_MRR@10": co_mrr,
                "delta_transductive": trans_mrr - base_mrr,
                "delta_corpus_only": co_mrr - base_mrr,
                "baseline_Recall@10": base_recall,
                "transductive_ZCA_Recall@10": trans_recall,
                "corpus_only_ZCA_Recall@10": co_recall,
            })
    return rows


def epsilon_sensitivity(model_name: str, embeddings: Dict, eval_data: Dict) -> List[Dict]:
    """For each (corpus, qf, eps) compute corpus-only ZCA MRR@10."""
    rows = []
    for corpus_name, corpus_data in eval_data.items():
        for qf_name in corpus_data["queries"].keys():
            q_emb = embeddings[f"{corpus_name}_{qf_name}_q"]
            d_emb = embeddings[f"{corpus_name}_{qf_name}_d"]
            base_mrr, _, _ = compute_mrr_recall(q_emb, d_emb)
            for eps in EPSILON_VALUES:
                try:
                    W, mu = fit_zca_corpus_only(d_emb, eps)
                    q_t = apply_zca(q_emb, W, mu)
                    d_t = apply_zca(d_emb, W, mu)
                    mrr, recall, _ = compute_mrr_recall(q_t, d_t)
                except Exception as e:
                    mrr, recall = float("nan"), float("nan")
                rows.append({
                    "model": model_name,
                    "corpus": corpus_name,
                    "query_format": qf_name,
                    "epsilon": eps,
                    "baseline_MRR@10": base_mrr,
                    "corpus_only_ZCA_MRR@10": mrr,
                    "delta_MRR@10": mrr - base_mrr,
                    "Recall@10": recall,
                })
    return rows


def cross_validation(model_name: str, embeddings: Dict, eval_data: Dict,
                     n_folds: int = N_FOLDS) -> List[Dict]:
    """
    5-fold CV on corpus-only ZCA.
    For each fold:
      - Hold out 20 docs (fold_size = N // n_folds)
      - Fit corpus-only ZCA on remaining 80 docs (no queries)
      - Apply transformation to: all 100 docs + all 100 queries
      - For the 20 held-out query-doc pairs, compute MRR@10 against full 100-doc candidate set

    This shows the ZCA transformation generalizes to documents not seen during fitting.
    """
    rows = []
    rng = np.random.RandomState(SEED)
    for corpus_name, corpus_data in eval_data.items():
        for qf_name in corpus_data["queries"].keys():
            q_emb = embeddings[f"{corpus_name}_{qf_name}_q"]
            d_emb = embeddings[f"{corpus_name}_{qf_name}_d"]
            n = d_emb.shape[0]
            if n < n_folds * 2:
                continue

            base_mrr, _, _ = compute_mrr_recall(q_emb, d_emb)

            # Fixed fold assignment for reproducibility
            indices = np.arange(n)
            rng_fold = np.random.RandomState(SEED)
            rng_fold.shuffle(indices)
            fold_size = n // n_folds
            folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
            # Distribute remainder
            for i, idx in enumerate(indices[n_folds*fold_size:]):
                folds[i] = np.append(folds[i], idx)

            for fold_id in range(n_folds):
                held_out_idx = folds[fold_id]
                train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_id])

                # Fit ZCA on training docs only
                try:
                    W, mu = fit_zca_corpus_only(d_emb[train_idx], DEFAULT_EPSILON)
                    # Transform ALL docs and queries using the fold-specific ZCA
                    d_transformed = apply_zca(d_emb, W, mu)
                    q_transformed = apply_zca(q_emb, W, mu)

                    # Evaluate on held-out query-doc pairs against full candidate set
                    sim_matrix = q_transformed[held_out_idx] @ d_transformed.T
                    fold_ranks = []
                    for j, true_doc_idx in enumerate(held_out_idx):
                        scores = sim_matrix[j]
                        rank = (scores > scores[true_doc_idx]).sum() + 1
                        fold_ranks.append(rank)
                    fold_ranks = np.array(fold_ranks)
                    rr = np.where(fold_ranks <= 10, 1.0 / fold_ranks, 0.0)
                    fold_mrr = float(rr.mean())
                    fold_recall = float((fold_ranks <= 10).mean())
                except Exception as e:
                    log.warning(f"    CV fold {fold_id} failed: {e}")
                    fold_mrr, fold_recall = float("nan"), float("nan")

                rows.append({
                    "model": model_name,
                    "corpus": corpus_name,
                    "query_format": qf_name,
                    "fold": fold_id,
                    "n_train_docs": len(train_idx),
                    "n_held_out": len(held_out_idx),
                    "baseline_MRR@10_full": base_mrr,
                    "cv_MRR@10": fold_mrr,
                    "cv_Recall@10": fold_recall,
                })
    return rows


# ==CELL 7: PREFLIGHT — tokenizer-only test for all 13 models==
def preflight_tokenizers():
    """Load only tokenizers (no model weights). Fails fast if anything broken."""
    log.info("\n" + "="*70)
    log.info("PREFLIGHT: tokenizer-only test for all 13 models")
    log.info("="*70)

    from transformers import AutoTokenizer

    failed = []
    for config in MODELS:
        log.info(f"\nTesting {config.name}...")
        try:
            ids_to_test = [config.hf_id]
            if config.is_dual_encoder:
                ids_to_test = [config.hf_id_query, config.hf_id_doc]
            for hf_id in ids_to_test:
                tok = AutoTokenizer.from_pretrained(
                    hf_id, trust_remote_code=True, token=HF_TOKEN or None
                )
                test_text = "patient with chest pain and elevated troponin"
                ids = tok(test_text, return_tensors="pt")
                log.info(f"  {hf_id}: OK | {type(tok).__name__} | tokens: {ids['input_ids'].shape}")
                del tok
                gc.collect()
        except Exception as e:
            log.error(f"  {config.name} FAILED: {type(e).__name__}: {e}")
            failed.append((config.name, str(e)))

    if failed:
        log.error(f"\nPreflight failures: {failed}")
        raise RuntimeError(f"Preflight failed for {len(failed)} models: {[f[0] for f in failed]}")

    log.info("\nAll 13 model tokenizers loaded OK.\n")


# ==CELL 8: MAIN==
def main():
    log.info("="*70)
    log.info("Paper 12 Revision — Stream B: Decision 2 (corpus-only ZCA + CV)")
    log.info("="*70)
    log.info(f"Output: {STREAM_B_DIR}")

    # Step 1: Preflight
    preflight_tokenizers()

    # Step 2: Load eval data
    log.info("\nLoading evaluation data...")
    eval_data = build_evaluation_data()

    # Step 3: Extract embeddings for all 13 models (skip if cached)
    log.info("\n" + "="*70)
    log.info("Phase 1: Extract final-layer embeddings for all 13 models")
    log.info("="*70)

    for config in MODELS:
        emb_path = EMBEDDINGS_DIR / f"{config.name}.npz"
        if emb_path.exists():
            log.info(f"\n{config.name}: cached at {emb_path.name} (skipping)")
            continue

        log.info(f"\n{'='*70}")
        log.info(f"Extracting: {config.name} ({config.category})")
        log.info(f"{'='*70}")
        t0 = time.time()
        try:
            embeddings = extract_model_embeddings(config, eval_data)
            save_embeddings(config.name, embeddings)
            elapsed = (time.time() - t0) / 60
            log.info(f"  {config.name} done in {elapsed:.1f} min")
        except Exception as e:
            log.error(f"  {config.name} FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Step 4: Run analyses on cached embeddings
    log.info("\n" + "="*70)
    log.info("Phase 2: Methodology comparison + epsilon sweep + cross-validation")
    log.info("="*70)

    all_method = []
    all_eps = []
    all_cv = []

    for config in MODELS:
        emb_path = EMBEDDINGS_DIR / f"{config.name}.npz"
        if not emb_path.exists():
            log.warning(f"  {config.name}: no embeddings file, skipping analyses")
            continue
        log.info(f"\nAnalyzing: {config.name}")
        try:
            embeddings = load_embeddings(config.name)
            method_rows = methodology_comparison(config.name, embeddings, eval_data)
            eps_rows = epsilon_sensitivity(config.name, embeddings, eval_data)
            cv_rows = cross_validation(config.name, embeddings, eval_data)
            all_method.extend(method_rows)
            all_eps.extend(eps_rows)
            all_cv.extend(cv_rows)
            log.info(f"  {len(method_rows)} method rows | {len(eps_rows)} eps rows | {len(cv_rows)} cv rows")
        except Exception as e:
            log.error(f"  {config.name} analyses FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    method_df = pd.DataFrame(all_method)
    eps_df = pd.DataFrame(all_eps)
    cv_df = pd.DataFrame(all_cv)

    method_df.to_parquet(RESULTS_DIR / "methodology_comparison.parquet", index=False)
    eps_df.to_parquet(RESULTS_DIR / "epsilon_sensitivity.parquet", index=False)
    cv_df.to_parquet(RESULTS_DIR / "cross_validation.parquet", index=False)

    log.info(f"\n  Saved methodology_comparison: {len(method_df)} rows")
    log.info(f"  Saved epsilon_sensitivity: {len(eps_df)} rows")
    log.info(f"  Saved cross_validation: {len(cv_df)} rows")

    log.info("\n" + "="*70)
    log.info("STREAM B COMPLETE")
    log.info(f"  Embeddings:    {EMBEDDINGS_DIR}")
    log.info(f"  Results:       {RESULTS_DIR}")
    log.info("="*70)


if __name__ == "__main__":
    main()


# ==CELL 9: OPTIONAL — SYNC TO DRIVE==
# Run after Stream B completes:
#
# from google.colab import drive
# drive.mount('/content/drive')
# import shutil
# from pathlib import Path
# drive_target = Path("/content/drive/MyDrive/Content/paper2_layer_analysis/revision/stream_b")
# drive_target.mkdir(parents=True, exist_ok=True)
# shutil.copytree('/content/paper12_revision/stream_b', str(drive_target), dirs_exist_ok=True)
# print(f"Synced to: {drive_target}")