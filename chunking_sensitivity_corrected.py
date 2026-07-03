"""
Chunking sensitivity re-run with corrected token-level truncation.
Fixes the word-level approximation in the original stream_e3 pipeline that
produced spurious differences between "first" and "full" strategies for
512-ceiling models (Editorial Comment #1, round-3 revision).

Original bug:
    approx_word_limit = approx_max_tokens // 2  # too imprecise
    return " ".join(words[:approx_word_limit])  # word-level, not token-level

Fix:
    Token-level truncation using the model's own tokenizer.

Verification:
    For 512-ceiling models (BERT-base-uncased, BGE-base, MedCPT) whose
    max_length equals or exceeds the longest evaluation document,
    max |first - full| < 1e-6 across all 6 corpus × query-format conditions.

Usage:
    Set HF_TOKEN below, ensure inputs are present, then execute end-to-end.
    Colab: paste into a single cell; local: run as `python chunking_sensitivity_corrected.py`.

Expected inputs:
    /content/mtsamples_sample.csv     (or ./inputs/mtsamples_sample.csv)
    /content/metadata_queries.json    (or ./inputs/metadata_queries.json)
    /content/synthetic_notes.csv      (or ./inputs/synthetic_notes.csv)

Output:
    /content/chunking_corrected/chunking_sensitivity_CORRECTED.parquet  (72 rows)
    /content/chunking_corrected/chunking_sensitivity_CORRECTED_table4.csv  (aggregate)

Runtime: ~15 min on Colab A100 / H100 with fp16 for BioMistral-7B.
"""

HF_TOKEN = "INPUT HF-TOKEN HERE"

import os, gc, json, time, logging, warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED); torch.manual_seed(SEED)
log.info(f"Device: {DEVICE}")
if torch.cuda.is_available():
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")

# HF auth
from huggingface_hub import login
login(token=HF_TOKEN, add_to_git_credential=False)
log.info("HF authenticated.")

# Paths
INPUT_DIR = Path("/content")
MTSAMPLES_PATH = INPUT_DIR / "mtsamples_sample.csv"
QUERIES_PATH = INPUT_DIR / "metadata_queries.json"
SYNTH_NOTES_PATH = INPUT_DIR / "synthetic_notes.csv"
OUTPUT_DIR = Path("/content/chunking_corrected")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PARQUET = OUTPUT_DIR / "chunking_sensitivity_CORRECTED.parquet"

for p in [MTSAMPLES_PATH, QUERIES_PATH, SYNTH_NOTES_PATH]:
    assert p.exists(), f"Missing input file: {p}"
log.info("All input files present.")


# Model configs (matched to original Stream E.3 subset)
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


MODELS_SUBSET: List[ModelConfig] = [
    ModelConfig(name="BERT-base-uncased", hf_id="bert-base-uncased",
                category="General Encoder", pooling="mean", n_layers=12,
                dtype="fp32", max_length=512),
    ModelConfig(name="BGE-base", hf_id="BAAI/bge-base-en-v1.5",
                category="General Embedding", n_layers=12,
                dtype="fp32", max_length=512),
    ModelConfig(name="MedCPT", hf_id="ncbi/MedCPT-Query-Encoder",
                category="Biomedical Retriever", pooling="cls",
                is_dual_encoder=True,
                hf_id_query="ncbi/MedCPT-Query-Encoder",
                hf_id_doc="ncbi/MedCPT-Article-Encoder",
                n_layers=12, dtype="fp32", max_length=512),
    ModelConfig(name="BioMistral-7B", hf_id="BioMistral/BioMistral-7B",
                category="Domain LLM", pooling="mean",
                dtype="fp16", max_length=2048, n_layers=32),
]


# ────────────────────────────────────────────────────────────────────
# THE FIX: token-level truncation
# ────────────────────────────────────────────────────────────────────
def chunk_text_corrected(text, strategy, tokenizer, max_tokens):
    """Token-level chunking. For docs that fit in max_tokens, all strategies
    return the same text (=> first==full for 512-ceiling models on short docs)."""
    if strategy == "full":
        return text
    token_ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= max_tokens:
        return text
    if strategy == "first":
        kept = token_ids[:max_tokens]
    elif strategy == "last":
        kept = token_ids[-max_tokens:]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return tokenizer.decode(kept, skip_special_tokens=True)


# Data loading
def load_mtsamples():
    df = pd.read_csv(MTSAMPLES_PATH).dropna(subset=["text"])
    return df["text"].tolist()[:100]


def load_pmc_patients():
    from datasets import load_dataset
    try:
        ds = load_dataset("zhengyun21/PMC-Patients", split="train", token=HF_TOKEN)
        df = ds.to_pandas()
    except Exception:
        from huggingface_hub import hf_hub_download
        json_path = hf_hub_download(
            repo_id="zhengyun21/PMC-Patients",
            filename="PMC-Patients-V2.json",
            repo_type="dataset",
            token=HF_TOKEN,
        )
        df = pd.read_json(json_path, lines=False)
    text_col = None
    for col in ["patient", "text", "patient_text", "summary"]:
        if col in df.columns: text_col = col; break
    if text_col is None:
        for c in df.columns:
            if df[c].dtype == object and df[c].str.len().median() > 100:
                text_col = c; break
    df = df.rename(columns={text_col: "text"}).dropna(subset=["text"])
    df = df[df["text"].str.len() > 50].reset_index(drop=True)
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(df), size=100, replace=False)
    return df.iloc[idx].reset_index(drop=True)["text"].tolist()


def load_synthetic(queries_kw):
    from rank_bm25 import BM25Okapi
    df = pd.read_csv(SYNTH_NOTES_PATH).dropna(subset=["text"])
    all_docs = df["text"].tolist()
    tokenized = [d.lower().split() for d in all_docs]
    bm25 = BM25Okapi(tokenized)
    mapping = [int(np.argmax(bm25.get_scores(q.lower().split()))) for q in queries_kw]
    return [all_docs[i] for i in mapping]


def load_queries():
    with open(QUERIES_PATH) as f:
        return json.load(f)


log.info("Building eval data...")
queries = load_queries()
mt_docs = load_mtsamples()
pmc_docs = load_pmc_patients()
syn_docs = load_synthetic(queries["Synthetic"]["keyword"])
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
    log.info(f"  {corpus_name}: {n} docs, {n} queries x 2 formats")


# Metrics
def compute_per_query_ranks(q_emb, d_emb):
    q_emb = q_emb.astype(np.float32); d_emb = d_emb.astype(np.float32)
    sim = q_emb @ d_emb.T
    n = sim.shape[0]
    ranks = np.zeros(n, dtype=np.int32)
    for i in range(n):
        ranks[i] = (sim[i] > sim[i][i]).sum() + 1
    return ranks


def compute_mrr_recall(ranks, k=10):
    rr = np.where(ranks <= k, 1.0 / ranks, 0.0)
    return float(rr.mean()), float((ranks <= k).mean())


# Pooling
def mean_pool(hs, am):
    m = am.unsqueeze(-1).expand(hs.size()).float()
    return torch.sum(hs * m, dim=1) / m.sum(dim=1).clamp(min=1e-9)


def cls_pool(hs, am):
    return hs[:, 0, :]


def extract_final_layer(texts, model, tokenizer, max_length, batch_size=8, pool="mean"):
    pool_fn = mean_pool if pool == "mean" else cls_pool
    embs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt")
        input_device = next(model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        last = outputs.last_hidden_state
        pooled = pool_fn(last, inputs["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
        embs.append(pooled.cpu().numpy().astype(np.float32))
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return np.concatenate(embs, axis=0)


# Sanity check the fix BEFORE running the long extraction
log.info("Sanity-checking chunk_text_corrected on BERT-base...")
from transformers import AutoTokenizer
_check_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
_short_doc = eval_data["MTSamples"]["documents"][0]
_n_tok = len(_check_tok(_short_doc, truncation=False, add_special_tokens=False)["input_ids"])
log.info(f"  Short MTSamples doc[0] = {_n_tok} tokens")
_first = chunk_text_corrected(_short_doc, "first", _check_tok, 512)
_full = chunk_text_corrected(_short_doc, "full", _check_tok, 512)
if _n_tok <= 512:
    assert _first == _full, "FIX FAILED: first != full on short doc"
    log.info("  [OK] first == full for doc fitting in 512 tokens")
_long_doc = next(
    (d for d in eval_data["PMC-Patients"]["documents"]
     if len(_check_tok(d, truncation=False, add_special_tokens=False)["input_ids"]) > 512),
    None,
)
if _long_doc is not None:
    _ln = len(_check_tok(_long_doc, truncation=False, add_special_tokens=False)["input_ids"])
    _lf = chunk_text_corrected(_long_doc, "first", _check_tok, 512)
    _lfn = len(_check_tok(_lf, truncation=False, add_special_tokens=False)["input_ids"])
    log.info(f"  Long PMC doc = {_ln} tokens, after first-512 = {_lfn} tokens")
    assert _lfn <= 512, f"FIX FAILED: first truncation produced {_lfn} > 512 tokens"
    log.info("  [OK] first truncates to <= max_tokens on long doc")
del _check_tok
log.info("Sanity check passed. Starting extraction.\n")


# Main extraction
from transformers import AutoModel
rows = []
t_start = time.time()
for config in MODELS_SUBSET:
    log.info(f"=== {config.name} ({config.hf_id}, max_length={config.max_length}) ===")
    if config.is_dual_encoder:
        d_tok = AutoTokenizer.from_pretrained(config.hf_id_doc, trust_remote_code=True, token=HF_TOKEN)
        d_model = AutoModel.from_pretrained(config.hf_id_doc, trust_remote_code=True, token=HF_TOKEN).to(DEVICE).eval()
        if d_tok.pad_token is None: d_tok.pad_token = d_tok.eos_token
        q_tok = AutoTokenizer.from_pretrained(config.hf_id_query, trust_remote_code=True, token=HF_TOKEN)
        q_model = AutoModel.from_pretrained(config.hf_id_query, trust_remote_code=True, token=HF_TOKEN).to(DEVICE).eval()
        if q_tok.pad_token is None: q_tok.pad_token = q_tok.eos_token
    else:
        d_tok = AutoTokenizer.from_pretrained(config.hf_id, trust_remote_code=True, token=HF_TOKEN)
        load_kwargs = {"trust_remote_code": True, "token": HF_TOKEN}
        if config.dtype == "fp16":
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        d_model = AutoModel.from_pretrained(config.hf_id, **load_kwargs)
        if config.dtype != "fp16":
            d_model = d_model.to(DEVICE)
        d_model.eval()
        if d_tok.pad_token is None: d_tok.pad_token = d_tok.eos_token
        q_model, q_tok = d_model, d_tok
    pool = config.pooling
    bs = 2 if config.dtype == "fp16" else 16

    for strategy in ["first", "last", "full"]:
        log.info(f"  strategy={strategy}")
        for corpus_name, corpus_data in eval_data.items():
            documents = corpus_data["documents"]
            chunked_docs = [chunk_text_corrected(d, strategy, d_tok, config.max_length) for d in documents]
            prefixed_d = ([config.doc_prefix + d for d in chunked_docs]
                          if config.doc_prefix else chunked_docs)
            d_emb = extract_final_layer(prefixed_d, d_model, d_tok, config.max_length, bs, pool)
            for qf_name, q_list in corpus_data["queries"].items():
                if config.instruction:
                    prefixed_q = [f"Instruct: {config.instruction}\nQuery: {q}" for q in q_list]
                elif config.query_prefix:
                    prefixed_q = [config.query_prefix + q for q in q_list]
                else:
                    prefixed_q = q_list
                q_emb = extract_final_layer(prefixed_q, q_model, q_tok, config.max_length, bs, pool)
                ranks = compute_per_query_ranks(q_emb, d_emb)
                mrr, recall = compute_mrr_recall(ranks)
                rows.append({
                    "model": config.name,
                    "chunking": strategy,
                    "corpus": corpus_name,
                    "query_format": qf_name,
                    "MRR@10": mrr,
                    "Recall@10": recall,
                    "median_rank": float(np.median(ranks)),
                })

    del d_model
    if config.is_dual_encoder: del q_model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

log.info(f"\nExtraction complete. Elapsed: {(time.time() - t_start) / 60:.1f} min")
df_corrected = pd.DataFrame(rows)
df_corrected.to_parquet(OUTPUT_PARQUET, index=False)
log.info(f"Saved {len(df_corrected)} rows -> {OUTPUT_PARQUET}")


# Verification: for 512-ceiling models, first == full should hold
print("\n" + "=" * 75)
print("VERIFICATION: first vs full MRR@10 (512-ceiling models should be identical)")
print("=" * 75)
for model_name in df_corrected["model"].unique():
    cfg = next(m for m in MODELS_SUBSET if m.name == model_name)
    sub = df_corrected[df_corrected["model"] == model_name]
    pivot = sub.pivot_table(index=["corpus", "query_format"],
                            columns="chunking", values="MRR@10").round(6)
    print(f"\n{model_name} (max_length={cfg.max_length}):")
    print(pivot)
    if cfg.max_length == 512:
        diff = (pivot["first"] - pivot["full"]).abs().max()
        status = "OK" if diff < 1e-6 else f"FAIL diff={diff:.6f}"
        print(f"  max |first - full|: {status}")


# Summary table for manuscript Table 4
print("\n" + "=" * 75)
print("MANUSCRIPT-READY SUMMARY (mean MRR@10 across 6 corpus x query_format conditions)")
print("=" * 75)
wide = (df_corrected.groupby(["model", "chunking"])["MRR@10"].mean()
        .unstack()[["first", "last", "full"]].round(4))
print(wide.to_string())

wide_csv = OUTPUT_DIR / "chunking_sensitivity_CORRECTED_table4.csv"
wide.to_csv(wide_csv)
log.info(f"Saved Table 4 -> {wide_csv}")

print("\nDONE.")
