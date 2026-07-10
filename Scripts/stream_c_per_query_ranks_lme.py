"""
Paper 12 Revision — Stream C: Decision 3 Implementation
========================================================
SELF-CONTAINED. Paste entire file into Colab.

Implements Decision 3: Full layer-wise per-query linear mixed-effects models
(replaces condition-level ANOVA from original Paper 12).

Scope: ALL 13 models (11 original Paper 12 panel + 2 new from Stream A).

Pipeline:
  Phase 1: Layer-wise embedding extraction for all 13 models
           (re-extracts because Stream B only saved final-layer .npz)
  Phase 2: Per-query rank computation at each layer with three interventions:
           - Baseline (no intervention)
           - Corpus-only ZCA @ eps=1e-5 (Decision 2 primary methodology)
           - Transductive ZCA @ eps=1e-5 (original Paper 12 methodology, kept for comparison)
  Phase 3: LME fitting per model:
           log(rank+1) ~ rel_layer + intervention + rel_layer:intervention
                        + corpus + query_format + (1 | query_idx)
           where rel_layer = layer / max_layer (normalized to [0,1])

Inputs (at /content/sample_data/):
  - mtsamples_sample.csv
  - metadata_queries.json
  - synthetic_notes.csv

Outputs (at /content/paper12_revision/stream_c/):
  - rank_data/per_query_ranks.parquet                (long-format per-query ranks)
  - lme_results/per_model_fixed_effects.parquet      (LME coefficients with p-values)
  - lme_results/per_model_summary.parquet            (ICC, log-likelihood, n_obs per model)
  - lme_results/lme_text_summaries.txt               (human-readable LME summaries)

Sanity check runs FIRST. If any tokenizer fails, aborts in ~3 min.

Runtime on RTX PRO 6000 Blackwell:
  - Phase 1 (layer-wise extraction): ~90-150 min
  - Phase 2 (rank computation): ~5-10 min
  - Phase 3 (LME fitting): ~10-20 min
  - Total: ~105-180 min
"""

# ==CELL 1: INSTALL DEPENDENCIES==
import subprocess
import sys

def _ensure_packages(packages):
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
    "statsmodels",   # LME via mixedlm
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
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

INPUT_DIR = Path("/content/sample_data")
MTSAMPLES_SAMPLE_PATH = INPUT_DIR / "mtsamples_sample.csv"
QUERIES_PATH = INPUT_DIR / "metadata_queries.json"
SYNTH_NOTES_PATH = INPUT_DIR / "synthetic_notes.csv"

STREAM_C_DIR = Path("/content/paper12_revision/stream_c")
RANK_DATA_DIR = STREAM_C_DIR / "rank_data"
LME_RESULTS_DIR = STREAM_C_DIR / "lme_results"
EMBEDDINGS_CACHE_DIR = STREAM_C_DIR / "layer_embeddings"

for d in [STREAM_C_DIR, RANK_DATA_DIR, LME_RESULTS_DIR, EMBEDDINGS_CACHE_DIR]:
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
DEFAULT_EPSILON = 1e-5

np.random.seed(SEED)
torch.manual_seed(SEED)

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)

log.info(f"Device: {DEVICE}")
if torch.cuda.is_available():
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ==CELL 3: ALL 13 MODEL CONFIGS (verbatim from Stream B)==
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
    ModelConfig(name="BioBERT", hf_id="dmis-lab/biobert-v1.1",
                category="Domain Encoder", n_layers=12),
    ModelConfig(name="ClinicalBERT", hf_id="medicalai/ClinicalBERT",
                category="Domain Encoder", n_layers=6),
    ModelConfig(name="BioLORD-2023", hf_id="FremyCompany/BioLORD-2023",
                category="Biomedical Retriever", n_layers=12),
    ModelConfig(name="MedCPT", hf_id="ncbi/MedCPT-Query-Encoder",
                category="Biomedical Retriever", pooling="cls",
                is_dual_encoder=True,
                hf_id_query="ncbi/MedCPT-Query-Encoder",
                hf_id_doc="ncbi/MedCPT-Article-Encoder", n_layers=12),
    ModelConfig(name="BGE-base", hf_id="BAAI/bge-base-en-v1.5",
                category="General Embedding", n_layers=12),
    ModelConfig(name="GTE-base", hf_id="thenlper/gte-base",
                category="General Embedding", n_layers=12),
    ModelConfig(name="Nomic-embed-text", hf_id="nomic-ai/nomic-embed-text-v1.5",
                category="General Embedding",
                query_prefix="search_query: ", doc_prefix="search_document: ",
                n_layers=12),
    ModelConfig(name="E5-Mistral-7B", hf_id="intfloat/e5-mistral-7b-instruct",
                category="General LLM", pooling="eos",
                instruction="Given a clinical note, retrieve the most relevant clinical document.",
                dtype="fp16", max_length=4096, n_layers=32),
    ModelConfig(name="Phi-3-mini", hf_id="microsoft/Phi-3-mini-4k-instruct",
                category="General LLM", pooling="mean",
                dtype="fp16", max_length=4096, n_layers=32),
    ModelConfig(name="E5-Mistral-7B-ablation", hf_id="intfloat/e5-mistral-7b-instruct",
                category="General LLM", pooling="mean", instruction="",
                dtype="fp16", max_length=4096, n_layers=32, is_ablation=True),
    ModelConfig(name="Nomic-embed-text-nopfx", hf_id="nomic-ai/nomic-embed-text-v1.5",
                category="General Embedding", query_prefix="", doc_prefix="",
                n_layers=12, is_ablation=True),
    ModelConfig(name="BERT-base-uncased", hf_id="bert-base-uncased",
                category="General Encoder", pooling="mean", n_layers=12,
                dtype="fp32", max_length=512),
    ModelConfig(name="BioMistral-7B", hf_id="BioMistral/BioMistral-7B",
                category="Domain LLM", pooling="mean",
                dtype="fp16", max_length=2048, n_layers=32),
]

log.info(f"Total models: {len(MODELS)}")


# ==CELL 4: DATA LOADING (verbatim from Stream B)==
def load_mtsamples_aligned():
    log.info("Loading MTSamples...")
    df = pd.read_csv(MTSAMPLES_SAMPLE_PATH)
    df = df.dropna(subset=["text"])
    return df["text"].tolist()[:100]


def load_pmc_patients_aligned():
    log.info("Loading PMC-Patients from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("zhengyun21/PMC-Patients", split="train", token=HF_TOKEN or None)
        df = ds.to_pandas()
    except Exception:
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

    df = df.rename(columns={text_col: "text"})
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.len() > 50].reset_index(drop=True)

    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(df), size=100, replace=False)
    return df.iloc[idx].reset_index(drop=True)["text"].tolist()


def load_synthetic_aligned(queries_kw):
    from rank_bm25 import BM25Okapi
    df = pd.read_csv(SYNTH_NOTES_PATH)
    df = df.dropna(subset=["text"])
    all_docs = df["text"].tolist()
    tokenized = [d.lower().split() for d in all_docs]
    bm25 = BM25Okapi(tokenized)
    mapping = [int(np.argmax(bm25.get_scores(q.lower().split()))) for q in queries_kw]
    return [all_docs[i] for i in mapping]


def load_queries():
    with open(QUERIES_PATH) as f:
        return json.load(f)


def build_evaluation_data():
    queries = load_queries()
    mt_docs = load_mtsamples_aligned()
    pmc_docs = load_pmc_patients_aligned()
    syn_docs = load_synthetic_aligned(queries["Synthetic"]["keyword"])

    eval_data = {}
    for corpus_name, docs in [("MTSamples", mt_docs),
                              ("PMC-Patients", pmc_docs),
                              ("Synthetic", syn_docs)]:
        n_queries = len(queries[corpus_name]["keyword"])
        n = min(len(docs), n_queries)
        eval_data[corpus_name] = {
            "documents": docs[:n],
            "queries": {
                "keyword": queries[corpus_name]["keyword"][:n],
                "natural_language": queries[corpus_name]["natural_language"][:n],
            },
        }
        log.info(f"  {corpus_name}: {n} docs x {n} queries x 2 formats")
    return eval_data


# ==CELL 5: LAYER-WISE EMBEDDING EXTRACTION (from Stream A)==
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


def _supports_output_hidden_states(model):
    sig = inspect.signature(model.forward)
    if "output_hidden_states" in sig.parameters:
        return True
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _extract_via_hooks(texts, model, tokenizer, config, batch_size, pool_fn):
    captured = {}
    hooks = []
    embed_module = None
    transformer_layers = []

    for name, module in model.named_modules():
        if any(p in name for p in ["embeddings", "embed_tokens"]) and embed_module is None:
            if hasattr(module, 'forward') and not any(s in name for s in ["LayerNorm", "dropout", "position", "token_type", "norm"]):
                if name.count('.') <= 1:
                    embed_module = (name, module)
        if any(p in name for p in ["encoder.layers.", "transformer.layers.",
                                    "encoder.layer.", "transformer.layer.",
                                    "model.layers."]):
            parts = name.split(".")
            is_top_level = False
            for i, part in enumerate(parts):
                if part.isdigit():
                    if not ".".join(parts[i+1:]):
                        is_top_level = True
                        break
            if is_top_level:
                transformer_layers.append((name, module))

    if embed_module is None or len(transformer_layers) == 0:
        raise RuntimeError(f"Could not identify layers for {type(model).__name__}")

    def make_hook(layer_idx):
        captured[layer_idx] = []
        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            captured[layer_idx].append(out.detach())
        return hook_fn

    captured[0] = []
    hooks.append(embed_module[1].register_forward_hook(make_hook(0)))
    for i, (name, module) in enumerate(transformer_layers):
        hooks.append(module.register_forward_hook(make_hook(i + 1)))

    n_layers = 1 + len(transformer_layers)
    layer_embs = {i: [] for i in range(n_layers)}

    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=config.max_length, return_tensors="pt")
            input_device = getattr(model, '_input_device', None) or next(model.parameters()).device
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

            for k in captured:
                captured[k] = []

            with torch.no_grad():
                model(**inputs)

            attn_mask = inputs["attention_mask"]
            for layer_idx in range(n_layers):
                if not captured[layer_idx]:
                    continue
                hs = captured[layer_idx][0]
                pooled = pool_fn(hs, attn_mask)
                pooled = F.normalize(pooled, p=2, dim=1)
                layer_embs[layer_idx].append(pooled.cpu().numpy())
            for k in captured:
                captured[k] = []
            torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    for layer_idx in layer_embs:
        if layer_embs[layer_idx]:
            layer_embs[layer_idx] = np.concatenate(layer_embs[layer_idx], axis=0)
    return {k: v for k, v in layer_embs.items() if isinstance(v, np.ndarray)}


def extract_all_layers(texts, model, tokenizer, config, batch_size=16):
    pool_fn = POOL_FN[config.pooling]

    if not _supports_output_hidden_states(model):
        return _extract_via_hooks(texts, model, tokenizer, config, batch_size, pool_fn)

    n_layers = config.n_layers + 1
    layer_embs = {i: [] for i in range(n_layers)}

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=config.max_length, return_tensors="pt")
        input_device = getattr(model, '_input_device', None) or next(model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
        except TypeError:
            return _extract_via_hooks(texts, model, tokenizer, config, batch_size, pool_fn)

        hidden_states = outputs.hidden_states
        attn_mask = inputs["attention_mask"]

        if len(hidden_states) != n_layers:
            if start == 0:
                n_layers = len(hidden_states)
                layer_embs = {i: [] for i in range(n_layers)}

        for layer_idx in range(n_layers):
            hs = hidden_states[layer_idx]
            pooled = pool_fn(hs, attn_mask)
            pooled = F.normalize(pooled, p=2, dim=1)
            layer_embs[layer_idx].append(pooled.cpu().numpy())

        del outputs, hidden_states
        torch.cuda.empty_cache()

    for layer_idx in layer_embs:
        layer_embs[layer_idx] = np.concatenate(layer_embs[layer_idx], axis=0)
    return layer_embs


def load_hf_model(hf_id, dtype="fp32", trust_remote=True):
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


def apply_prefixes(texts, prefix="", instruction=""):
    if instruction:
        return [f"Instruct: {instruction}\nQuery: {t}" for t in texts]
    elif prefix:
        return [prefix + t for t in texts]
    return texts


# ==CELL 6: RANK COMPUTATION + INTERVENTIONS==
def fit_zca_transductive(d_emb, q_emb, eps=DEFAULT_EPSILON):
    d_emb = d_emb.astype(np.float32)
    q_emb = q_emb.astype(np.float32)
    all_emb = np.vstack([d_emb, q_emb])
    mean_vec = all_emb.mean(axis=0)
    centered = all_emb - mean_vec
    cov = centered.T @ centered / (len(centered) - 1)
    U, S, _ = np.linalg.svd(cov)
    W = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T
    return W, mean_vec


def fit_zca_corpus_only(d_emb, eps=DEFAULT_EPSILON):
    d_emb = d_emb.astype(np.float32)
    mean_vec = d_emb.mean(axis=0)
    centered = d_emb - mean_vec
    cov = centered.T @ centered / (len(centered) - 1)
    U, S, _ = np.linalg.svd(cov)
    W = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T
    return W, mean_vec


def apply_zca(emb, W, mean_vec):
    emb_f32 = emb.astype(np.float32)
    transformed = (emb_f32 - mean_vec) @ W.T
    return transformed / np.linalg.norm(transformed, axis=1, keepdims=True).clip(1e-9)


def per_query_ranks(q_emb, d_emb):
    """For each query i, return rank of correct doc i."""
    q_emb = q_emb.astype(np.float32)
    d_emb = d_emb.astype(np.float32)
    sim_matrix = q_emb @ d_emb.T
    n = sim_matrix.shape[0]
    ranks = np.zeros(n, dtype=np.int32)
    for i in range(n):
        scores = sim_matrix[i]
        ranks[i] = int((scores > scores[i]).sum() + 1)
    return ranks


def compute_per_query_ranks_for_model(config: ModelConfig, eval_data: Dict) -> pd.DataFrame:
    """
    For one model, extract all layers and compute per-query ranks for each
    (corpus, qf, layer, intervention) combination.
    Returns long-format DataFrame.
    """
    rows = []
    batch_size = 2 if config.dtype == "fp16" else 16

    if config.is_dual_encoder:
        # Load query encoder, extract query embeddings
        q_model, q_tok = load_hf_model(config.hf_id_query, config.dtype)
        query_layers_all = {}
        for corpus_name, corpus_data in eval_data.items():
            for qf_name, queries in corpus_data["queries"].items():
                prefixed_q = apply_prefixes(queries, config.query_prefix, config.instruction)
                q_layers = extract_all_layers(prefixed_q, q_model, q_tok, config, batch_size)
                query_layers_all[(corpus_name, qf_name)] = q_layers
        free_model(q_model)

        # Load doc encoder, extract doc embeddings
        d_model, d_tok = load_hf_model(config.hf_id_doc, config.dtype)
        doc_layers_all = {}
        for corpus_name, corpus_data in eval_data.items():
            documents = corpus_data["documents"]
            prefixed_d = apply_prefixes(documents, config.doc_prefix)
            d_layers = extract_all_layers(prefixed_d, d_model, d_tok, config, batch_size)
            doc_layers_all[corpus_name] = d_layers
        free_model(d_model)
    else:
        model, tokenizer = load_hf_model(config.hf_id, config.dtype)
        query_layers_all = {}
        doc_layers_all = {}
        for corpus_name, corpus_data in eval_data.items():
            documents = corpus_data["documents"]
            prefixed_d = apply_prefixes(documents, config.doc_prefix)
            doc_layers_all[corpus_name] = extract_all_layers(prefixed_d, model, tokenizer, config, batch_size)
            for qf_name, queries in corpus_data["queries"].items():
                prefixed_q = apply_prefixes(queries, config.query_prefix, config.instruction)
                query_layers_all[(corpus_name, qf_name)] = extract_all_layers(prefixed_q, model, tokenizer, config, batch_size)
        free_model(model)

    # Now compute per-query ranks for each layer × intervention
    for corpus_name in eval_data.keys():
        d_layers = doc_layers_all[corpus_name]
        for qf_name in eval_data[corpus_name]["queries"].keys():
            q_layers = query_layers_all[(corpus_name, qf_name)]
            common_layers = sorted(set(d_layers.keys()) & set(q_layers.keys()))
            max_layer = max(common_layers)

            for layer in common_layers:
                d_emb = d_layers[layer]
                q_emb = q_layers[layer]
                rel_layer = layer / max_layer if max_layer > 0 else 0.0

                # Baseline (no intervention)
                ranks_base = per_query_ranks(q_emb, d_emb)
                for q_idx, r in enumerate(ranks_base):
                    rows.append({
                        "model": config.name,
                        "category": config.category,
                        "corpus": corpus_name,
                        "query_format": qf_name,
                        "layer": int(layer),
                        "max_layer": int(max_layer),
                        "rel_layer": float(rel_layer),
                        "query_idx": int(q_idx),
                        "intervention": "baseline",
                        "rank": int(r),
                    })

                # Corpus-only ZCA
                try:
                    W_c, mu_c = fit_zca_corpus_only(d_emb)
                    q_c = apply_zca(q_emb, W_c, mu_c)
                    d_c = apply_zca(d_emb, W_c, mu_c)
                    ranks_co = per_query_ranks(q_c, d_c)
                    for q_idx, r in enumerate(ranks_co):
                        rows.append({
                            "model": config.name,
                            "category": config.category,
                            "corpus": corpus_name,
                            "query_format": qf_name,
                            "layer": int(layer),
                            "max_layer": int(max_layer),
                            "rel_layer": float(rel_layer),
                            "query_idx": int(q_idx),
                            "intervention": "corpus_only_ZCA",
                            "rank": int(r),
                        })
                except Exception as e:
                    log.warning(f"    corpus-only ZCA failed at layer {layer}: {e}")

                # Transductive ZCA
                try:
                    W_t, mu_t = fit_zca_transductive(d_emb, q_emb)
                    q_t = apply_zca(q_emb, W_t, mu_t)
                    d_t = apply_zca(d_emb, W_t, mu_t)
                    ranks_tr = per_query_ranks(q_t, d_t)
                    for q_idx, r in enumerate(ranks_tr):
                        rows.append({
                            "model": config.name,
                            "category": config.category,
                            "corpus": corpus_name,
                            "query_format": qf_name,
                            "layer": int(layer),
                            "max_layer": int(max_layer),
                            "rel_layer": float(rel_layer),
                            "query_idx": int(q_idx),
                            "intervention": "transductive_ZCA",
                            "rank": int(r),
                        })
                except Exception as e:
                    log.warning(f"    transductive ZCA failed at layer {layer}: {e}")

    return pd.DataFrame(rows)


# ==CELL 7: LME FITTING==
def fit_lme_for_model(rank_df: pd.DataFrame, model_name: str) -> Tuple[Dict, str]:
    """
    Fit per-model LME:
      log(rank + 1) ~ rel_layer + intervention + rel_layer:intervention
                     + C(corpus) + C(query_format)
                     + (1 | query_idx)

    Returns (fixed_effects_dict, text_summary).
    """
    df = rank_df[rank_df["model"] == model_name].copy()
    if len(df) == 0:
        return {}, f"No data for {model_name}"

    df["log_rank"] = np.log(df["rank"].astype(float) + 1.0)
    # Treat intervention as categorical with baseline as reference
    df["intervention"] = pd.Categorical(
        df["intervention"],
        categories=["baseline", "corpus_only_ZCA", "transductive_ZCA"]
    )

    formula = "log_rank ~ rel_layer * C(intervention) + C(corpus) + C(query_format)"

    try:
        md = smf.mixedlm(formula, df, groups=df["query_idx"])
        result = md.fit(method="lbfgs", maxiter=200, reml=True)
    except Exception as e:
        return {}, f"LME failed for {model_name}: {e}"

    # Extract fixed effects
    fe_rows = []
    for name in result.params.index:
        if name == "Group Var":
            continue
        fe_rows.append({
            "model": model_name,
            "term": name,
            "estimate": float(result.params[name]),
            "std_error": float(result.bse[name]) if name in result.bse.index else float("nan"),
            "z_value": float(result.tvalues[name]) if name in result.tvalues.index else float("nan"),
            "p_value": float(result.pvalues[name]) if name in result.pvalues.index else float("nan"),
        })

    # Random effect variance + ICC
    re_var = float(result.cov_re.iloc[0, 0]) if hasattr(result.cov_re, "iloc") else float(result.cov_re[0, 0])
    resid_var = float(result.scale)
    icc = re_var / (re_var + resid_var) if (re_var + resid_var) > 0 else float("nan")

    summary = {
        "model": model_name,
        "n_obs": int(len(df)),
        "n_queries_unique": int(df["query_idx"].nunique()),
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "random_effect_var": re_var,
        "residual_var": resid_var,
        "ICC": icc,
        "converged": bool(result.converged),
    }

    text = str(result.summary())

    return {"fixed_effects": fe_rows, "summary": summary}, text


# ==CELL 8: PREFLIGHT==
def preflight_tokenizers():
    log.info("\n" + "="*70)
    log.info("PREFLIGHT: tokenizer-only test for all 13 models")
    log.info("="*70)

    from transformers import AutoTokenizer

    failed = []
    for config in MODELS:
        try:
            ids_to_test = [config.hf_id]
            if config.is_dual_encoder:
                ids_to_test = [config.hf_id_query, config.hf_id_doc]
            for hf_id in ids_to_test:
                tok = AutoTokenizer.from_pretrained(
                    hf_id, trust_remote_code=True, token=HF_TOKEN or None
                )
                _ = tok("test", return_tensors="pt")
                del tok
                gc.collect()
            log.info(f"  {config.name}: OK")
        except Exception as e:
            log.error(f"  {config.name} FAILED: {type(e).__name__}: {e}")
            failed.append((config.name, str(e)))

    if failed:
        raise RuntimeError(f"Preflight failed for {len(failed)} models: {[f[0] for f in failed]}")

    log.info("\nAll 13 model tokenizers loaded OK.\n")


# ==CELL 9: MAIN==
def main():
    log.info("="*70)
    log.info("Paper 12 Revision — Stream C: Decision 3 (layer-wise per-query LME)")
    log.info("="*70)
    log.info(f"Output: {STREAM_C_DIR}")

    # Step 1: Preflight
    preflight_tokenizers()

    # Step 2: Load eval data
    log.info("\nLoading evaluation data...")
    eval_data = build_evaluation_data()

    # Step 3: Extract layer-wise ranks for each model (resumable via per-model parquets)
    log.info("\n" + "="*70)
    log.info("Phase 1+2: Layer-wise extraction + per-query rank computation")
    log.info("="*70)

    all_dfs = []
    for config in MODELS:
        out_path = RANK_DATA_DIR / f"{config.name}_ranks.parquet"
        if out_path.exists():
            log.info(f"\n{config.name}: cached at {out_path.name}")
            all_dfs.append(pd.read_parquet(out_path))
            continue

        log.info(f"\n{'='*70}")
        log.info(f"Processing: {config.name} ({config.category})")
        log.info(f"{'='*70}")
        t0 = time.time()
        try:
            df = compute_per_query_ranks_for_model(config, eval_data)
            df.to_parquet(out_path, index=False)
            elapsed = (time.time() - t0) / 60
            log.info(f"  {config.name} done in {elapsed:.1f} min ({len(df):,} rows)")
            all_dfs.append(df)
        except Exception as e:
            log.error(f"  {config.name} FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Combine all into one master rank dataframe
    rank_df = pd.concat(all_dfs, ignore_index=True)
    rank_df.to_parquet(RANK_DATA_DIR / "per_query_ranks.parquet", index=False)
    log.info(f"\nMaster rank table: {len(rank_df):,} rows -> per_query_ranks.parquet")

    # Step 4: Fit LME per model
    log.info("\n" + "="*70)
    log.info("Phase 3: LME fitting per model")
    log.info("="*70)

    all_fe = []
    all_summaries = []
    text_summaries = []

    for config in MODELS:
        log.info(f"\nFitting LME: {config.name}")
        t0 = time.time()
        try:
            result_dict, text = fit_lme_for_model(rank_df, config.name)
            if result_dict:
                all_fe.extend(result_dict["fixed_effects"])
                all_summaries.append(result_dict["summary"])
                text_summaries.append(f"\n{'='*80}\n{config.name}\n{'='*80}\n{text}\n")
                elapsed = (time.time() - t0)
                log.info(f"  Done in {elapsed:.1f}s | ICC={result_dict['summary']['ICC']:.3f}")
            else:
                log.warning(f"  {config.name}: no LME result")
                text_summaries.append(f"\n{'='*80}\n{config.name}\n{'='*80}\n{text}\n")
        except Exception as e:
            log.error(f"  {config.name} LME FAILED: {e}")

    fe_df = pd.DataFrame(all_fe)
    summary_df = pd.DataFrame(all_summaries)
    fe_df.to_parquet(LME_RESULTS_DIR / "per_model_fixed_effects.parquet", index=False)
    summary_df.to_parquet(LME_RESULTS_DIR / "per_model_summary.parquet", index=False)
    with open(LME_RESULTS_DIR / "lme_text_summaries.txt", "w") as f:
        f.writelines(text_summaries)

    log.info(f"\nSaved fixed_effects: {len(fe_df)} rows")
    log.info(f"Saved summary: {len(summary_df)} rows")
    log.info(f"Saved text summaries: {LME_RESULTS_DIR / 'lme_text_summaries.txt'}")

    log.info("\n" + "="*70)
    log.info("STREAM C COMPLETE")
    log.info(f"  Rank data:    {RANK_DATA_DIR}")
    log.info(f"  LME results:  {LME_RESULTS_DIR}")
    log.info("="*70)


if __name__ == "__main__":
    main()


# ==CELL 10: OPTIONAL — SYNC TO DRIVE==
# from google.colab import drive
# drive.mount('/content/drive')
# import shutil
# from pathlib import Path
# drive_target = Path("/content/drive/MyDrive/Content/paper2_layer_analysis/revision/stream_c")
# drive_target.mkdir(parents=True, exist_ok=True)
# shutil.copytree('/content/paper12_revision/stream_c', str(drive_target), dirs_exist_ok=True)
# print(f"Synced to: {drive_target}")