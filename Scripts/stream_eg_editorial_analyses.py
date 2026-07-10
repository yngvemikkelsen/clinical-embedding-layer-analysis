"""
Paper 12 Revision — Streams E + G Combined
============================================
SELF-CONTAINED. Paste entire file into Colab.

Stream E (Decision 6): Limitations analyses
  - E.1: Lexical overlap (BM25 vs embedding-MRR correlation, per model)
  - E.2: Synthetic corpus audit (distribution comparison vs MTSamples/PMC-Patients)
  - E.3: Chunking sensitivity (first-512 vs last-512 vs full document at fixed max_length)

Stream G (Decision 8): Editorial analyses
  - G.1: MedCPT MRR with expanded candidate set (full corpus per condition)
  - G.2: Full participation ratio + avg cosine across all 13 models × conditions
  - G.3: E5-Mistral-7B-ablation intervention sweep (mean centering, ZCA variants)

ALL ANALYSES USE CACHED EMBEDDINGS FROM STREAMS A/B.
Only E.3 (chunking sensitivity) requires new model loads — done in Phase 2.

Inputs needed:
  - /content/drive/MyDrive/Content/paper2_layer_analysis/revision/stream_b/embeddings/*.npz
  - /content/sample_data/{mtsamples_sample.csv, metadata_queries.json, synthetic_notes.csv}

Outputs (at /content/paper12_revision/stream_eg/):
  - stream_e/
    - lexical_overlap.parquet         (BM25 rank vs embedding rank correlation)
    - synthetic_audit.parquet         (lexical/length/topic distribution comparison)
    - chunking_sensitivity.parquet    (3 chunking strategies × subset of models)
  - stream_g/
    - medcpt_expanded.parquet         (MedCPT MRR over expanded candidate set)
    - geometry_full.parquet           (PR + avg cosine, all models × conditions × layers)
    - e5_ablation_intervention_sweep.parquet  (5 intervention variants on E5-Mistral-abl)

Runtime on RTX PRO 6000 Blackwell:
  - Phase 1 (E.1, E.2, G.1, G.2, G.3 — embedding-only): ~5-15 min
  - Phase 2 (E.3 chunking — requires 4 model re-loads): ~30-45 min
  - Total: ~35-60 min
"""

# ==CELL 1: INSTALL==
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
    "scipy",
])
print("Dependencies ready.\n")


# ==CELL 2: IMPORTS, PATHS, CONFIG==
import os
import gc
import json
import time
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Mount Drive and sync revision dir to local for fast access
from google.colab import drive
drive.mount('/content/drive')

import shutil
DRIVE_REVISION = Path('/content/drive/MyDrive/Content/paper2_layer_analysis/revision')
LOCAL_REVISION = Path('/content/paper12_revision')

if DRIVE_REVISION.exists() and not (LOCAL_REVISION / 'stream_b' / 'embeddings').exists():
    log.info(f"Syncing Drive → local: {DRIVE_REVISION} → {LOCAL_REVISION}")
    shutil.copytree(str(DRIVE_REVISION), str(LOCAL_REVISION), dirs_exist_ok=True)

# Inputs
INPUT_DIR = Path("/content/sample_data")
MTSAMPLES_SAMPLE_PATH = INPUT_DIR / "mtsamples_sample.csv"
QUERIES_PATH = INPUT_DIR / "metadata_queries.json"
SYNTH_NOTES_PATH = INPUT_DIR / "synthetic_notes.csv"

STREAM_B_EMB_DIR = LOCAL_REVISION / "stream_b" / "embeddings"

# Outputs
STREAM_EG_DIR = LOCAL_REVISION / "stream_eg"
STREAM_E_DIR = STREAM_EG_DIR / "stream_e"
STREAM_G_DIR = STREAM_EG_DIR / "stream_g"
for d in [STREAM_EG_DIR, STREAM_E_DIR, STREAM_G_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Files in {INPUT_DIR}:")
for f in sorted(INPUT_DIR.iterdir()):
    print(f"  {f.name}")

assert MTSAMPLES_SAMPLE_PATH.exists()
assert QUERIES_PATH.exists()
assert SYNTH_NOTES_PATH.exists()
assert STREAM_B_EMB_DIR.exists(), f"Stream B embeddings missing at {STREAM_B_EMB_DIR}"

cached_npz = sorted(STREAM_B_EMB_DIR.glob('*.npz'))
print(f"\nCached embeddings: {len(cached_npz)}/13")
for f in cached_npz:
    print(f"  {f.name}: {f.stat().st_size/1e6:.1f} MB")

print("\nAll inputs present.\n")

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


# ==CELL 3: MODEL CONFIGS (verbatim from Stream B)==
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

MODEL_BY_NAME = {m.name: m for m in MODELS}


# ==CELL 4: DATA LOADING==
def load_mtsamples():
    df = pd.read_csv(MTSAMPLES_SAMPLE_PATH).dropna(subset=["text"])
    return df["text"].tolist()[:100]


def load_pmc_patients():
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


def build_eval_data():
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
    return eval_data


# ==CELL 5: SHARED METRICS==
def compute_per_query_ranks(q_emb, d_emb):
    q_emb = q_emb.astype(np.float32)
    d_emb = d_emb.astype(np.float32)
    sim = q_emb @ d_emb.T
    n = sim.shape[0]
    ranks = np.zeros(n, dtype=np.int32)
    for i in range(n):
        ranks[i] = (sim[i] > sim[i][i]).sum() + 1
    return ranks


def compute_mrr_recall(ranks, k=10):
    rr = np.where(ranks <= k, 1.0 / ranks, 0.0)
    return float(rr.mean()), float((ranks <= k).mean())


def fit_zca_corpus_only(d_emb, eps=DEFAULT_EPSILON):
    d_emb = d_emb.astype(np.float32)
    mean_vec = d_emb.mean(axis=0)
    centered = d_emb - mean_vec
    cov = centered.T @ centered / (len(centered) - 1)
    U, S, _ = np.linalg.svd(cov)
    W = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T
    return W, mean_vec


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


def apply_transform(emb, W, mean_vec):
    emb_f32 = emb.astype(np.float32)
    transformed = (emb_f32 - mean_vec) @ W.T
    return transformed / np.linalg.norm(transformed, axis=1, keepdims=True).clip(1e-9)


def compute_participation_ratio(emb):
    emb = emb.astype(np.float32)
    centered = emb - emb.mean(axis=0)
    n = centered.shape[0]
    if n > 1000:
        idx = np.random.RandomState(SEED).choice(n, 1000, replace=False)
        centered = centered[idx]
    try:
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        S_sq = S ** 2
        return float((S_sq.sum())**2 / (S_sq**2).sum())
    except Exception:
        return float("nan")


def compute_avg_cosine(emb, n_pairs=10000):
    emb = emb.astype(np.float32)
    n = emb.shape[0]
    rng = np.random.RandomState(SEED)
    a = rng.randint(0, n, n_pairs)
    b = rng.randint(0, n, n_pairs)
    mask = a != b
    a, b = a[mask], b[mask]
    return float(np.sum(emb[a] * emb[b], axis=1).mean())


def compute_anisotropy(emb):
    emb = emb.astype(np.float32)
    centered = emb - emb.mean(axis=0)
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


# ==CELL 6: STREAM E.1 — Lexical overlap (BM25 vs embedding correlation)==
def stream_e1_lexical_overlap(eval_data):
    """
    For each (model, corpus, query_format):
      - Compute BM25 ranks for all queries
      - Compute embedding ranks (baseline, final layer)
      - Pearson + Spearman correlation between BM25 rank and embedding rank
    Tells us how much of the embedding model's behavior is captured by lexical matching.
    """
    from rank_bm25 import BM25Okapi

    log.info("\n" + "="*70)
    log.info("Stream E.1: Lexical overlap (BM25 vs embedding correlation)")
    log.info("="*70)

    rows = []
    for corpus_name, corpus_data in eval_data.items():
        docs = corpus_data["documents"]
        tokenized = [d.lower().split() for d in docs]
        bm25 = BM25Okapi(tokenized)

        for qf_name, queries in corpus_data["queries"].items():
            # BM25 ranks for each query
            bm25_ranks = []
            for i, q in enumerate(queries):
                scores = bm25.get_scores(q.lower().split())
                bm25_ranks.append((scores > scores[i]).sum() + 1)
            bm25_ranks = np.array(bm25_ranks)
            bm25_mrr = float(np.where(bm25_ranks <= 10, 1.0/bm25_ranks, 0.0).mean())

            for model_name in MODEL_BY_NAME.keys():
                npz_path = STREAM_B_EMB_DIR / f"{model_name}.npz"
                if not npz_path.exists():
                    continue
                with np.load(npz_path) as data:
                    qkey = f"{corpus_name}_{qf_name}_q"
                    dkey = f"{corpus_name}_{qf_name}_d"
                    if qkey not in data.files or dkey not in data.files:
                        continue
                    q_emb = data[qkey].copy()
                    d_emb = data[dkey].copy()

                emb_ranks = compute_per_query_ranks(q_emb, d_emb)
                emb_mrr = float(np.where(emb_ranks <= 10, 1.0/emb_ranks, 0.0).mean())

                pearson_r, pearson_p = stats.pearsonr(bm25_ranks, emb_ranks)
                spearman_r, spearman_p = stats.spearmanr(bm25_ranks, emb_ranks)

                rows.append({
                    "model": model_name,
                    "corpus": corpus_name,
                    "query_format": qf_name,
                    "bm25_MRR@10": bm25_mrr,
                    "embedding_MRR@10": emb_mrr,
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_r": float(spearman_r),
                    "spearman_p": float(spearman_p),
                })

    df = pd.DataFrame(rows)
    df.to_parquet(STREAM_E_DIR / "lexical_overlap.parquet", index=False)
    log.info(f"  Saved {len(df)} rows -> lexical_overlap.parquet")
    return df


# ==CELL 7: STREAM E.2 — Synthetic corpus audit==
def stream_e2_synthetic_audit(eval_data):
    """
    Compare synthetic corpus to MTSamples/PMC-Patients on:
      - Document length distribution (word count)
      - Query length distribution
      - Vocabulary overlap with real corpora
      - Token type-to-token ratio (TTR) as lexical diversity measure
    """
    log.info("\n" + "="*70)
    log.info("Stream E.2: Synthetic corpus audit")
    log.info("="*70)

    rows = []
    corpus_vocabs = {}
    for corpus_name, corpus_data in eval_data.items():
        docs = corpus_data["documents"]
        doc_lengths = [len(d.split()) for d in docs]
        all_tokens = [t.lower() for d in docs for t in d.split()]
        vocab = set(all_tokens)
        corpus_vocabs[corpus_name] = vocab
        ttr = len(vocab) / len(all_tokens) if all_tokens else 0

        for qf_name, queries in corpus_data["queries"].items():
            query_lengths = [len(q.split()) for q in queries]
            rows.append({
                "corpus": corpus_name,
                "query_format": qf_name,
                "n_docs": len(docs),
                "doc_length_mean": float(np.mean(doc_lengths)),
                "doc_length_median": float(np.median(doc_lengths)),
                "doc_length_std": float(np.std(doc_lengths)),
                "doc_length_min": int(np.min(doc_lengths)),
                "doc_length_max": int(np.max(doc_lengths)),
                "query_length_mean": float(np.mean(query_lengths)),
                "query_length_median": float(np.median(query_lengths)),
                "vocab_size": len(vocab),
                "total_tokens": len(all_tokens),
                "type_token_ratio": ttr,
            })

    # Vocabulary overlap (jaccard)
    overlap_rows = []
    corpora = list(corpus_vocabs.keys())
    for i, ci in enumerate(corpora):
        for cj in corpora[i+1:]:
            vi, vj = corpus_vocabs[ci], corpus_vocabs[cj]
            jaccard = len(vi & vj) / len(vi | vj) if (vi | vj) else 0
            overlap_rows.append({
                "corpus_a": ci,
                "corpus_b": cj,
                "jaccard": float(jaccard),
                "shared_terms": len(vi & vj),
                "total_unique": len(vi | vj),
            })

    df = pd.DataFrame(rows)
    overlap_df = pd.DataFrame(overlap_rows)
    df.to_parquet(STREAM_E_DIR / "synthetic_audit_descriptive.parquet", index=False)
    overlap_df.to_parquet(STREAM_E_DIR / "synthetic_audit_vocab_overlap.parquet", index=False)
    log.info(f"  Saved descriptive ({len(df)} rows) + vocab overlap ({len(overlap_df)} rows)")
    return df, overlap_df


# ==CELL 8: STREAM G.1 — MedCPT MRR with expanded candidate set==
def stream_g1_medcpt_expanded(eval_data):
    """
    Original Paper 12 evaluates MedCPT against 100-doc candidate set per corpus.
    Reviewer asked: how does MedCPT perform with expanded candidate set?

    Here we evaluate MedCPT queries against ALL documents from ALL corpora
    (300 total: 100 MTSamples + 100 PMC-Patients + 100 Synthetic).
    Tests cross-corpus retrieval challenge.
    """
    log.info("\n" + "="*70)
    log.info("Stream G.1: MedCPT MRR with expanded candidate set (300 docs)")
    log.info("="*70)

    npz_path = STREAM_B_EMB_DIR / "MedCPT.npz"
    if not npz_path.exists():
        log.warning("MedCPT.npz not found, skipping G.1")
        return pd.DataFrame()

    with np.load(npz_path) as data:
        emb_dict = {k: data[k].copy() for k in data.files}

    rows = []
    corpora = ["MTSamples", "PMC-Patients", "Synthetic"]
    query_formats = ["keyword", "natural_language"]

    for qf_name in query_formats:
        # Build expanded doc set
        all_docs = []
        doc_corpus_labels = []
        for corpus_name in corpora:
            dkey = f"{corpus_name}_{qf_name}_d"
            if dkey not in emb_dict:
                continue
            all_docs.append(emb_dict[dkey])
            doc_corpus_labels.extend([corpus_name] * emb_dict[dkey].shape[0])
        if not all_docs:
            continue
        all_docs_emb = np.vstack(all_docs)
        doc_corpus_labels = np.array(doc_corpus_labels)

        # For each corpus's queries, evaluate against expanded set
        for corpus_name in corpora:
            qkey = f"{corpus_name}_{qf_name}_q"
            if qkey not in emb_dict:
                continue
            q_emb = emb_dict[qkey]

            # The correct doc for query i in this corpus is at index i within that corpus's doc range
            # In the expanded set, find that index
            corpus_start = int(np.where(doc_corpus_labels == corpus_name)[0][0])

            sim = q_emb.astype(np.float32) @ all_docs_emb.astype(np.float32).T
            ranks = []
            cross_corpus_top1 = []
            for i in range(q_emb.shape[0]):
                true_doc_idx = corpus_start + i
                scores = sim[i]
                rank = int((scores > scores[true_doc_idx]).sum() + 1)
                ranks.append(rank)
                # What corpus did the top-1 retrieved doc come from?
                top1_idx = int(np.argmax(scores))
                cross_corpus_top1.append(doc_corpus_labels[top1_idx])

            ranks = np.array(ranks)
            mrr, recall = compute_mrr_recall(ranks)

            # Cross-corpus confusion: % of top-1 from each corpus
            top1_corpus_dist = pd.Series(cross_corpus_top1).value_counts(normalize=True).to_dict()

            rows.append({
                "query_corpus": corpus_name,
                "query_format": qf_name,
                "candidate_set_size": int(all_docs_emb.shape[0]),
                "MRR@10": mrr,
                "Recall@10": recall,
                "median_rank": float(np.median(ranks)),
                "top1_from_correct_corpus_pct": float(top1_corpus_dist.get(corpus_name, 0.0)),
                "top1_distribution": json.dumps(top1_corpus_dist),
            })

    df = pd.DataFrame(rows)
    df.to_parquet(STREAM_G_DIR / "medcpt_expanded.parquet", index=False)
    log.info(f"  Saved {len(df)} rows -> medcpt_expanded.parquet")
    return df


# ==CELL 9: STREAM G.2 — Full PR + cosine across all models × conditions==
def stream_g2_geometry_full():
    """
    For every (model, corpus, query_format), compute:
      - Participation ratio (effective dimensionality)
      - Average pairwise cosine similarity
      - Anisotropy (SVD-based)
    For both DOC and QUERY embeddings separately.
    """
    log.info("\n" + "="*70)
    log.info("Stream G.2: Full PR + cosine + anisotropy across all models")
    log.info("="*70)

    rows = []
    for model_name in MODEL_BY_NAME.keys():
        npz_path = STREAM_B_EMB_DIR / f"{model_name}.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path) as data:
            emb_dict = {k: data[k].copy() for k in data.files}

        for corpus_name in ["MTSamples", "PMC-Patients", "Synthetic"]:
            for qf_name in ["keyword", "natural_language"]:
                qkey = f"{corpus_name}_{qf_name}_q"
                dkey = f"{corpus_name}_{qf_name}_d"
                if qkey not in emb_dict or dkey not in emb_dict:
                    continue
                q_emb = emb_dict[qkey]
                d_emb = emb_dict[dkey]

                for emb_type, emb in [("doc", d_emb), ("query", q_emb)]:
                    rows.append({
                        "model": model_name,
                        "corpus": corpus_name,
                        "query_format": qf_name,
                        "embedding_type": emb_type,
                        "participation_ratio": compute_participation_ratio(emb),
                        "avg_cosine": compute_avg_cosine(emb),
                        "anisotropy": compute_anisotropy(emb),
                        "n_samples": int(emb.shape[0]),
                        "embedding_dim": int(emb.shape[1]),
                    })

    df = pd.DataFrame(rows)
    df.to_parquet(STREAM_G_DIR / "geometry_full.parquet", index=False)
    log.info(f"  Saved {len(df)} rows -> geometry_full.parquet")
    return df


# ==CELL 10: STREAM G.3 — E5-Mistral-7B-ablation intervention sweep==
def stream_g3_e5_intervention_sweep(eval_data):
    """
    For E5-Mistral-7B-ablation (the model showing strongest baseline degradation),
    apply 5 intervention variants:
      1. Baseline (no intervention)
      2. Mean centering only
      3. Corpus-only ZCA @ eps=1e-5
      4. Transductive ZCA @ eps=1e-5
      5. Transductive ZCA @ eps=1e-3 (larger regularization)
    """
    log.info("\n" + "="*70)
    log.info("Stream G.3: E5-Mistral-7B-ablation intervention sweep")
    log.info("="*70)

    npz_path = STREAM_B_EMB_DIR / "E5-Mistral-7B-ablation.npz"
    if not npz_path.exists():
        log.warning("E5-Mistral-7B-ablation.npz not found, skipping G.3")
        return pd.DataFrame()

    with np.load(npz_path) as data:
        emb_dict = {k: data[k].copy() for k in data.files}

    rows = []
    for corpus_name in ["MTSamples", "PMC-Patients", "Synthetic"]:
        for qf_name in ["keyword", "natural_language"]:
            qkey = f"{corpus_name}_{qf_name}_q"
            dkey = f"{corpus_name}_{qf_name}_d"
            if qkey not in emb_dict or dkey not in emb_dict:
                continue
            q_emb = emb_dict[qkey].astype(np.float32)
            d_emb = emb_dict[dkey].astype(np.float32)

            # 1. Baseline
            ranks_base = compute_per_query_ranks(q_emb, d_emb)
            base_mrr, base_recall = compute_mrr_recall(ranks_base)
            rows.append({"intervention": "baseline", "corpus": corpus_name,
                        "query_format": qf_name, "MRR@10": base_mrr,
                        "Recall@10": base_recall, "median_rank": float(np.median(ranks_base))})

            # 2. Mean centering only
            try:
                d_mean = d_emb.mean(axis=0, keepdims=True)
                q_c = (q_emb - d_mean) / np.linalg.norm(q_emb - d_mean, axis=1, keepdims=True).clip(1e-9)
                d_c = (d_emb - d_mean) / np.linalg.norm(d_emb - d_mean, axis=1, keepdims=True).clip(1e-9)
                ranks = compute_per_query_ranks(q_c, d_c)
                mrr, recall = compute_mrr_recall(ranks)
                rows.append({"intervention": "mean_centering", "corpus": corpus_name,
                            "query_format": qf_name, "MRR@10": mrr,
                            "Recall@10": recall, "median_rank": float(np.median(ranks))})
            except Exception as e:
                log.warning(f"  Mean centering failed: {e}")

            # 3. Corpus-only ZCA eps=1e-5
            for eps_label, eps in [("corpus_only_ZCA_eps1e-5", 1e-5)]:
                try:
                    W, mu = fit_zca_corpus_only(d_emb, eps)
                    q_t = apply_transform(q_emb, W, mu)
                    d_t = apply_transform(d_emb, W, mu)
                    ranks = compute_per_query_ranks(q_t, d_t)
                    mrr, recall = compute_mrr_recall(ranks)
                    rows.append({"intervention": eps_label, "corpus": corpus_name,
                                "query_format": qf_name, "MRR@10": mrr,
                                "Recall@10": recall, "median_rank": float(np.median(ranks))})
                except Exception as e:
                    log.warning(f"  {eps_label} failed: {e}")

            # 4-5. Transductive ZCA at two eps values
            for eps_label, eps in [("transductive_ZCA_eps1e-5", 1e-5),
                                    ("transductive_ZCA_eps1e-3", 1e-3)]:
                try:
                    W, mu = fit_zca_transductive(d_emb, q_emb, eps)
                    q_t = apply_transform(q_emb, W, mu)
                    d_t = apply_transform(d_emb, W, mu)
                    ranks = compute_per_query_ranks(q_t, d_t)
                    mrr, recall = compute_mrr_recall(ranks)
                    rows.append({"intervention": eps_label, "corpus": corpus_name,
                                "query_format": qf_name, "MRR@10": mrr,
                                "Recall@10": recall, "median_rank": float(np.median(ranks))})
                except Exception as e:
                    log.warning(f"  {eps_label} failed: {e}")

    df = pd.DataFrame(rows)
    df["delta_vs_baseline"] = df.groupby(["corpus","query_format"])["MRR@10"].transform(
        lambda x: x - x.iloc[0]
    )
    df.to_parquet(STREAM_G_DIR / "e5_ablation_intervention_sweep.parquet", index=False)
    log.info(f"  Saved {len(df)} rows -> e5_ablation_intervention_sweep.parquet")
    return df


# ==CELL 11: STREAM E.3 — Chunking sensitivity (REQUIRES MODEL RELOAD)==
def stream_e3_chunking_sensitivity(eval_data):
    """
    For 4 representative models, re-extract document embeddings with 3 chunking strategies:
      - first_512: only the first 512 tokens
      - last_512: only the last 512 tokens
      - full: full document (up to model's max_length)

    Tests whether retrieval is dominated by document beginnings vs full content.
    Subset of models chosen for time: BERT-base (fast), BioMistral (LLM), BGE-base (retrieval), MedCPT (dual-encoder).
    """
    log.info("\n" + "="*70)
    log.info("Stream E.3: Chunking sensitivity (4 models × 3 strategies)")
    log.info("="*70)

    from transformers import AutoModel, AutoTokenizer

    def mean_pool(hs, am):
        m = am.unsqueeze(-1).expand(hs.size()).float()
        return torch.sum(hs * m, dim=1) / m.sum(dim=1).clamp(min=1e-9)

    def cls_pool(hs, am):
        return hs[:, 0, :]

    def extract_final_layer(texts, model, tokenizer, max_length, batch_size=8, pool="mean"):
        pool_fn = mean_pool if pool == "mean" else cls_pool
        embs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start+batch_size]
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
            torch.cuda.empty_cache()
        return np.concatenate(embs, axis=0)

    def chunk_text(text, strategy, approx_max_tokens):
        words = text.split()
        approx_word_limit = approx_max_tokens // 2  # rough words-to-tokens
        if strategy == "first":
            return " ".join(words[:approx_word_limit])
        elif strategy == "last":
            return " ".join(words[-approx_word_limit:]) if len(words) > approx_word_limit else text
        else:  # full
            return text

    subset = ["BERT-base-uncased", "BGE-base", "MedCPT", "BioMistral-7B"]
    rows = []

    for model_name in subset:
        config = MODEL_BY_NAME[model_name]
        log.info(f"\n  {model_name} ({config.hf_id})")

        # For each chunking strategy, extract docs only (queries unchanged)
        for strategy in ["first", "last", "full"]:
            log.info(f"    chunking={strategy}")
            try:
                if config.is_dual_encoder:
                    # Use article encoder for docs, query encoder for queries
                    d_tok = AutoTokenizer.from_pretrained(config.hf_id_doc, trust_remote_code=True, token=HF_TOKEN or None)
                    d_model = AutoModel.from_pretrained(config.hf_id_doc, trust_remote_code=True, token=HF_TOKEN or None).to(DEVICE).eval()
                    if d_tok.pad_token is None:
                        d_tok.pad_token = d_tok.eos_token
                    pool = config.pooling

                    # Load query encoder once for queries (full text — queries don't get chunked)
                    q_tok = AutoTokenizer.from_pretrained(config.hf_id_query, trust_remote_code=True, token=HF_TOKEN or None)
                    q_model = AutoModel.from_pretrained(config.hf_id_query, trust_remote_code=True, token=HF_TOKEN or None).to(DEVICE).eval()
                    if q_tok.pad_token is None:
                        q_tok.pad_token = q_tok.eos_token
                else:
                    d_tok = AutoTokenizer.from_pretrained(config.hf_id, trust_remote_code=True, token=HF_TOKEN or None)
                    load_kwargs = {"trust_remote_code": True, "token": HF_TOKEN or None}
                    if config.dtype == "fp16":
                        load_kwargs["torch_dtype"] = torch.float16
                        load_kwargs["device_map"] = "auto"
                    d_model = AutoModel.from_pretrained(config.hf_id, **load_kwargs)
                    if config.dtype != "fp16":
                        d_model = d_model.to(DEVICE)
                    d_model.eval()
                    if d_tok.pad_token is None:
                        d_tok.pad_token = d_tok.eos_token
                    q_model, q_tok = d_model, d_tok
                    pool = config.pooling

                bs = 2 if config.dtype == "fp16" else 16

                for corpus_name, corpus_data in eval_data.items():
                    documents = corpus_data["documents"]
                    chunked_docs = [chunk_text(d, strategy, config.max_length) for d in documents]
                    prefixed_d = [config.doc_prefix + d for d in chunked_docs] if config.doc_prefix else chunked_docs
                    d_emb = extract_final_layer(prefixed_d, d_model, d_tok, config.max_length, bs, pool)

                    for qf_name, queries in corpus_data["queries"].items():
                        prefixed_q = queries
                        if config.instruction:
                            prefixed_q = [f"Instruct: {config.instruction}\nQuery: {q}" for q in queries]
                        elif config.query_prefix:
                            prefixed_q = [config.query_prefix + q for q in queries]

                        q_emb = extract_final_layer(prefixed_q, q_model, q_tok, config.max_length, bs, pool)
                        ranks = compute_per_query_ranks(q_emb, d_emb)
                        mrr, recall = compute_mrr_recall(ranks)

                        rows.append({
                            "model": model_name,
                            "chunking": strategy,
                            "corpus": corpus_name,
                            "query_format": qf_name,
                            "MRR@10": mrr,
                            "Recall@10": recall,
                            "median_rank": float(np.median(ranks)),
                        })

                # Free
                del d_model, d_tok
                if config.is_dual_encoder:
                    del q_model, q_tok
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                log.error(f"    {model_name} / {strategy} FAILED: {type(e).__name__}: {e}")
                continue

    df = pd.DataFrame(rows)
    df.to_parquet(STREAM_E_DIR / "chunking_sensitivity.parquet", index=False)
    log.info(f"  Saved {len(df)} rows -> chunking_sensitivity.parquet")
    return df


# ==CELL 12: MAIN==
def main():
    log.info("="*70)
    log.info("Paper 12 Revision — Streams E + G Combined")
    log.info("="*70)

    log.info("\nLoading evaluation data...")
    eval_data = build_eval_data()

    # Phase 1: cached-embedding-only analyses (~5-15 min)
    log.info("\n" + "="*70)
    log.info("PHASE 1: cached-embedding-only analyses (~5-15 min)")
    log.info("="*70)

    stream_e1_lexical_overlap(eval_data)
    stream_e2_synthetic_audit(eval_data)
    stream_g1_medcpt_expanded(eval_data)
    stream_g2_geometry_full()
    stream_g3_e5_intervention_sweep(eval_data)

    # Phase 2: chunking sensitivity (requires 4 model reloads, ~30-45 min)
    log.info("\n" + "="*70)
    log.info("PHASE 2: chunking sensitivity (4 model reloads, ~30-45 min)")
    log.info("="*70)

    stream_e3_chunking_sensitivity(eval_data)

    log.info("\n" + "="*70)
    log.info("STREAMS E + G COMPLETE")
    log.info(f"  Stream E: {STREAM_E_DIR}")
    log.info(f"  Stream G: {STREAM_G_DIR}")
    log.info("="*70)


if __name__ == "__main__":
    main()


# ==CELL 13: SYNC TO DRIVE==
# Run after main() completes:
#
# import shutil
# drive_target = Path('/content/drive/MyDrive/Content/paper2_layer_analysis/revision')
# shutil.copytree('/content/paper12_revision', str(drive_target), dirs_exist_ok=True)
# print(f"Synced to: {drive_target}")