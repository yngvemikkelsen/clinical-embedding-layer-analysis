"""
Paper 12 Revision — Stream A: Matched Comparisons (v6)
========================================================
SELF-CONTAINED. Paste entire file into Colab.

v6 change: Meditron-7B replaced by BioMistral/BioMistral-7B
  Reason: Meditron-7B is a gated HuggingFace repo (requires access request).
  BioMistral-7B is the methodologically equivalent ungated replacement:
    - 7B parameters (same scale)
    - Biomedical pretraining (PubMed Central full-text)
    - Mistral-7B base architecture (current, clean tokenizer integration)
    - No retrieval training (matches Phi-3-mini in matched-comparison design)
    - Published Labrak et al., ACL Findings 2024
    - Ungated, currently maintained

Inputs (at /content/sample_data/):
  - mtsamples_sample.csv
  - metadata_queries.json
  - synthetic_notes.csv

Outputs (at /content/paper12_revision/):
  - results/BERT-base-uncased.parquet
  - results/BioMistral-7B.parquet
  - interventions/interventions_new_models.parquet

Sanity check runs FIRST. If BioMistral-7B tokenizer fails to load, script
aborts in ~1-2 minutes BEFORE committing to any long compute.
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

# rank_bm25 and sentencepiece are commonly missing; the others are usually
# pre-installed in Colab but we list them for safety
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

REVISION_DIR = Path("/content/paper12_revision")
RESULTS_DIR = REVISION_DIR / "results"
INTERVENTIONS_DIR = REVISION_DIR / "interventions"
LOGS_DIR = REVISION_DIR / "logs"

for d in [REVISION_DIR, RESULTS_DIR, INTERVENTIONS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Files in {INPUT_DIR}:")
for f in sorted(INPUT_DIR.iterdir()):
    print(f"  {f.name}")

assert MTSAMPLES_SAMPLE_PATH.exists(), f"Missing: {MTSAMPLES_SAMPLE_PATH}"
assert QUERIES_PATH.exists(), f"Missing: {QUERIES_PATH}"
assert SYNTH_NOTES_PATH.exists(), f"Missing: {SYNTH_NOTES_PATH}"
print("\nAll input files present.\n")

SEED = 42
N_EVAL_QUERIES = 100
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# ==CELL 3: MODEL REGISTRY==
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

log.info(f"Models to process: {len(MODELS)}")
for m in MODELS:
    log.info(f"  - {m.name} ({m.category}) | {m.hf_id} | {m.dtype} | {m.n_layers} layers")


# ==CELL 4: DATA LOADING==
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
        log.warning(f"  BM25 MRR@10 = {mrr:.3f} — alignment may be wrong!")

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

        if len(docs) < n_queries:
            log.warning(f"  {corpus_name}: only {len(docs)} docs but {n_queries} queries. Truncating.")

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
        log.info(f"  {corpus_name}: {n} docs x {n} queries x 2 formats")

    return eval_data


def run_bm25_sanity_check(eval_data: Dict):
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
            status = "OK" if mrr > 0.5 else "FAIL"
            log.info(f"  [{status}] {corpus_name}/{qf_name}: BM25 MRR@10={mrr:.3f}, Recall@10={recall:.3f}")
            if mrr < 0.5:
                all_ok = False

    if not all_ok:
        raise ValueError("BM25 alignment check failed.")

    log.info("  All corpora pass alignment check.\n")


# ==CELL 5: LAYER EXTRACTION ENGINE==
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


def _supports_output_hidden_states(model) -> bool:
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
        if any(pattern in name for pattern in ["embeddings", "embed_tokens"]) and embed_module is None:
            if hasattr(module, 'forward') and not any(sub in name for sub in ["LayerNorm", "dropout", "position", "token_type", "norm"]):
                if name.count('.') <= 1:
                    embed_module = (name, module)
        if any(pattern in name for pattern in [
            "encoder.layers.", "transformer.layers.",
            "encoder.layer.", "transformer.layer.",
            "model.layers.",
        ]):
            parts = name.split(".")
            is_top_level = False
            for i, part in enumerate(parts):
                if part.isdigit():
                    remaining = ".".join(parts[i+1:])
                    if not remaining:
                        is_top_level = True
                        break
            if is_top_level:
                transformer_layers.append((name, module))

    if embed_module is None or len(transformer_layers) == 0:
        raise RuntimeError(
            f"Could not identify embedding + transformer layers. "
            f"Model: {type(model).__name__}. "
            f"Modules: {[n for n, _ in model.named_modules() if n.count('.') <= 2]}"
        )

    log.info(f"  Hook extraction: 1 embed + {len(transformer_layers)} transformer layers")

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
        else:
            log.warning(f"  Layer {layer_idx} captured nothing")

    layer_embs = {k: v for k, v in layer_embs.items() if isinstance(v, np.ndarray)}
    return layer_embs


def extract_all_layers(texts, model, tokenizer, config, batch_size=32, pool_fn_override=None):
    pool_fn = pool_fn_override or POOL_FN[config.pooling]

    if not _supports_output_hidden_states(model):
        log.info(f"  Model {type(model).__name__} doesn't support output_hidden_states — using hooks")
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
        except TypeError as e:
            if start == 0:
                log.info(f"  output_hidden_states failed ({e}) — falling back to hooks")
                return _extract_via_hooks(texts, model, tokenizer, config, batch_size, pool_fn)
            else:
                raise

        hidden_states = outputs.hidden_states
        attn_mask = inputs["attention_mask"]

        if len(hidden_states) != n_layers:
            if start == 0:
                log.warning(f"  Expected {n_layers} hidden states, got {len(hidden_states)}. Adjusting.")
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


# ==CELL 6: METRIC COMPUTATION==
def compute_anisotropy_svd(embeddings: np.ndarray) -> float:
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
    q_emb = q_emb.astype(np.float32)
    d_emb = d_emb.astype(np.float32)
    sim_matrix = q_emb @ d_emb.T
    n = sim_matrix.shape[0]
    ranks = np.zeros(n)
    for i in range(n):
        scores = sim_matrix[i]
        ranks[i] = (scores > scores[i]).sum() + 1
    rr = np.where(ranks <= k, 1.0 / ranks, 0.0)
    return {"MRR@10": float(rr.mean()), "Recall@10": float((ranks <= k).mean())}, ranks


def compute_all_layer_metrics(doc_layer_embs, query_layer_embs, doc_token_counts=None):
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

        geo = {
            "anisotropy_svd": compute_anisotropy_svd(d_emb),
            "participation_ratio": compute_participation_ratio(d_emb),
            "avg_cosine": compute_avg_cosine(d_emb),
        }
        ret, ranks = compute_retrieval_metrics(q_emb, d_emb)
        rows.append({"layer": layer_idx, "length_tercile": "all", **geo, **ret})

        if tercile_labels is not None:
            for tercile in ["short", "medium", "long"]:
                mask = tercile_labels == tercile
                if mask.sum() < 5:
                    continue
                d_sub = d_emb[mask]

                geo_t = {
                    "anisotropy_svd": compute_anisotropy_svd(d_sub),
                    "participation_ratio": compute_participation_ratio(d_sub),
                    "avg_cosine": compute_avg_cosine(d_sub, n_pairs=min(5000, d_sub.shape[0]*100)),
                }

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
                ret_t = {"MRR@10": float(rr.mean()), "Recall@10": float((tercile_ranks <= 10).mean())}
                rows.append({"layer": layer_idx, "length_tercile": tercile, **geo_t, **ret_t})

    return rows


# ==CELL 7: MODEL LOADING + PROCESS_SINGLE_MODEL==
def load_hf_model(hf_id: str, dtype: str = "fp32", trust_remote: bool = True):
    """
    Load HF model and tokenizer via standard Auto* path.
    BioMistral-7B and BERT-base both work with this standard path.
    """
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


def _log_condition_summary(metrics: List[Dict], model_name: str, corpus: str, qf: str):
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
        f"-> best@L{best_layer}:{mrr_values[best_idx]:.3f} -> final:{mrr_values[-1]:.3f} | "
        f"aniso: {aniso_values[0]:.3f}->{aniso_values[-1]:.3f}"
    )


def process_single_model(config: ModelConfig, eval_data: Dict) -> pd.DataFrame:
    log.info(f"\n{'='*70}")
    log.info(f"Processing: {config.name} ({config.category})")
    log.info(f"  HF: {config.hf_id} | Pooling: {config.pooling} | Layers: {config.n_layers}")
    log.info(f"{'='*70}")

    all_rows = []
    batch_size = 2 if config.dtype == "fp16" else 16

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

            log.info(f"    [{(time.time()-t0)/60:.1f} min]")

    free_model(model)
    df = pd.DataFrame(all_rows)
    return df


# ==CELL 8: INTERVENTIONS==
def interventions_on_new_models(eval_data: Dict) -> pd.DataFrame:
    """ZCA whitening (transductive, eps=1e-5) and mean centering on final-layer embeddings."""
    rows = []

    for config in MODELS:
        log.info(f"\nInterventions: {config.name}")
        batch_size = 2 if config.dtype == "fp16" else 16
        model, tokenizer = load_hf_model(config.hf_id, config.dtype)

        for corpus_name, corpus_data in eval_data.items():
            documents = corpus_data["documents"]
            for qf_name, queries in corpus_data["queries"].items():
                log.info(f"  {corpus_name}/{qf_name}")

                prefixed_q = apply_prefixes(queries, config.query_prefix, config.instruction)
                prefixed_d = apply_prefixes(documents, config.doc_prefix)
                q_layers = extract_all_layers(prefixed_q, model, tokenizer, config, batch_size)
                d_layers = extract_all_layers(prefixed_d, model, tokenizer, config, batch_size)

                final_layer = max(d_layers.keys())
                q_emb = q_layers[final_layer]
                d_emb = d_layers[final_layer]

                baseline_ret, _ = compute_retrieval_metrics(q_emb, d_emb)
                q_emb_f32 = q_emb.astype(np.float32)
                d_emb_f32 = d_emb.astype(np.float32)

                # Mean centering
                d_mean = d_emb_f32.mean(axis=0, keepdims=True)
                q_centered = q_emb_f32 - d_mean
                d_centered = d_emb_f32 - d_mean
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

                # ZCA whitening (transductive — fit on combined q+d, eps=1e-5)
                try:
                    all_emb = np.vstack([d_emb_f32, q_emb_f32])
                    mean_vec = all_emb.mean(axis=0)
                    centered_all = all_emb - mean_vec
                    cov = centered_all.T @ centered_all / (len(centered_all) - 1)
                    U, S, Vt = np.linalg.svd(cov)
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
                    log.warning(f"    ZCA failed: {e}")

        free_model(model)

    return pd.DataFrame(rows)


# ==CELL 9: FAST PREFLIGHT — tokenizer-only test (fails in 30s if broken)==
def preflight_tokenizers():
    """
    Load ONLY tokenizers (no model weights). Fails in ~30s if anything is broken,
    BEFORE committing to ~14GB BioMistral-7B download + 60+ min compute.
    """
    log.info("\n" + "="*70)
    log.info("PREFLIGHT: tokenizer-only test (no model weights)")
    log.info("="*70)

    from transformers import AutoTokenizer

    for config in MODELS:
        log.info(f"\nTesting {config.name} tokenizer ({config.hf_id})...")
        try:
            tok = AutoTokenizer.from_pretrained(
                config.hf_id, trust_remote_code=True, token=HF_TOKEN or None
            )
            test_text = "patient with chest pain and elevated troponin"
            ids = tok(test_text, return_tensors="pt")
            log.info(f"  OK | {type(tok).__name__} | tokens: {ids['input_ids'].shape}")
            del tok
            gc.collect()
        except Exception as e:
            log.error(f"  FAILED: {type(e).__name__}: {e}")
            raise

    log.info("\nAll tokenizers OK. Proceeding to model load + pipeline.\n")


# ==CELL 10: FULL SANITY CHECK — loads model weights and runs forward pass==
def sanity_check_new_models():
    log.info("\n" + "="*70)
    log.info("FULL SANITY CHECK: load model + forward pass")
    log.info("="*70)

    from transformers import AutoModel, AutoTokenizer

    for config in MODELS:
        log.info(f"\nTesting {config.name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.hf_id, trust_remote_code=True, token=HF_TOKEN or None
            )
            log.info(f"  Tokenizer OK ({type(tokenizer).__name__})")

            load_kwargs = {"trust_remote_code": True, "token": HF_TOKEN or None}
            if config.dtype == "fp16":
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"

            model = AutoModel.from_pretrained(config.hf_id, **load_kwargs)
            if config.dtype != "fp16":
                model = model.to(DEVICE)
            model.eval()

            test_text = "patient with chest pain and elevated troponin"
            inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)

            n_hidden = len(out.hidden_states)
            log.info(f"  Forward pass OK | {n_hidden} hidden states (expected ~{config.n_layers + 1})")
            log.info(f"  Hidden state shape: {out.hidden_states[-1].shape}")

            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            log.error(f"  {config.name} FAILED: {type(e).__name__}: {e}")
            raise

    log.info("\nAll new models load successfully.\n")


# ==CELL 11: MAIN==
def main():
    log.info("="*70)
    log.info("Paper 12 Revision — Stream A: Matched Comparisons (v6)")
    log.info("="*70)
    log.info(f"Output: {REVISION_DIR}")

    # Step 1: Fast preflight — tokenizer-only test (~30s)
    preflight_tokenizers()

    # Step 2: Full sanity check — model + forward pass (~3-5 min)
    sanity_check_new_models()

    # Step 3: Load eval data
    log.info("\nLoading evaluation data...")
    eval_data = build_evaluation_data()
    run_bm25_sanity_check(eval_data)

    # Step 4: Process each new model
    log.info("\n" + "="*70)
    log.info("Processing new models")
    log.info("="*70)

    for config in MODELS:
        ckpt = RESULTS_DIR / f"{config.name}.parquet"
        if ckpt.exists():
            log.info(f"  {config.name}: already processed -> {ckpt.name} (skipping)")
            continue

        t0 = time.time()
        try:
            df = process_single_model(config, eval_data)
            df.to_parquet(ckpt, index=False)
            elapsed = (time.time() - t0) / 60
            log.info(f"  {config.name} done in {elapsed:.1f} min -> {ckpt.name}")
        except Exception as e:
            log.error(f"  {config.name} FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Step 5: Interventions on new models
    log.info("\n" + "="*70)
    log.info("Interventions on new models")
    log.info("="*70)

    intv_ckpt = INTERVENTIONS_DIR / "interventions_new_models.parquet"
    if intv_ckpt.exists():
        log.info(f"  Interventions already complete -> {intv_ckpt.name} (skipping)")
    else:
        intv_df = interventions_on_new_models(eval_data)
        intv_df.to_parquet(intv_ckpt, index=False)
        log.info(f"  Saved {len(intv_df)} intervention rows -> {intv_ckpt.name}")

    log.info("\n" + "="*70)
    log.info("STREAM A COMPLETE")
    log.info(f"  Results:       {RESULTS_DIR}")
    log.info(f"  Interventions: {INTERVENTIONS_DIR}")
    log.info("="*70)


if __name__ == "__main__":
    main()


# ==CELL 12: OPTIONAL — SYNC TO DRIVE==
# Run after Stream A completes to copy outputs to Drive:
#
# from google.colab import drive
# drive.mount('/content/drive')
# import shutil
# drive_target = Path("/content/drive/MyDrive/Content/paper2_layer_analysis/revision")
# drive_target.mkdir(parents=True, exist_ok=True)
# shutil.copytree('/content/paper12_revision', str(drive_target), dirs_exist_ok=True)
# print(f"Outputs synced to: {drive_target}")