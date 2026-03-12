"""
Paper 2 — Supplementary Experiments
====================================
1. Non-transductive (corpus-only) whitening vs transductive whitening
2. Validation corpus: 500 PMC-Patients with different seed

Run in Colab after the main experiment is complete.
Requires: all model checkpoints from main run, metadata_queries infrastructure.
"""

# ============================================================
# CELL 0: INSTALL & SETUP
# ============================================================
# !pip install -q transformers datasets huggingface_hub accelerate rank-bm25
# !pip install -q scipy scikit-learn pandas numpy tqdm pyarrow
# !git clone https://github.com/yngvemikkelsen/clinical-rag-retrieval-benchmark /content/paper1_repo 2>/dev/null || true

import os, gc, json, time, logging, warnings
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

from google.colab import drive
drive.mount('/content/drive')

DRIVE_BASE = Path("/content/drive/MyDrive/Content")
OUTPUT_DIR = DRIVE_BASE / "paper2_layer_analysis"
RESULTS_DIR = OUTPUT_DIR / "results"
SUPP_DIR = OUTPUT_DIR / "supplementary"
SUPP_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log.info(f"Device: {DEVICE}")
if torch.cuda.is_available():
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# PART 1: NON-TRANSDUCTIVE WHITENING
# ============================================================
# Compare three whitening variants:
#   A) Transductive: fit on docs + queries (existing result)
#   B) Corpus-only: fit on docs only, apply to both
#   C) Held-out corpus: fit on 80% of docs, apply to all docs + queries
# ============================================================

from dataclasses import dataclass

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

# Only the degraded models that benefit from whitening
DEGRADED_MODELS = [
    ModelConfig(name="BioBERT", hf_id="dmis-lab/biobert-v1.1", category="Domain Encoder", n_layers=12),
    ModelConfig(name="ClinicalBERT", hf_id="medicalai/ClinicalBERT", category="Domain Encoder", n_layers=6),
    ModelConfig(name="E5-Mistral-7B", hf_id="intfloat/e5-mistral-7b-instruct", category="General LLM",
                pooling="eos", instruction="Given a clinical note, retrieve the most relevant clinical document.",
                dtype="fp16", max_length=4096, n_layers=32),
    ModelConfig(name="Phi-3-mini", hf_id="microsoft/Phi-3-mini-4k-instruct", category="General LLM",
                pooling="mean", dtype="fp16", max_length=4096, n_layers=32),
]


def mean_pool(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    return (torch.sum(hidden_states * mask, dim=1) / mask.sum(dim=1).clamp(min=1e-9))

def eos_pool(hidden_states, attention_mask):
    seq_lens = attention_mask.sum(dim=1) - 1
    return hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), seq_lens]

def cls_pool(hidden_states, attention_mask):
    return hidden_states[:, 0, :]

POOL_FN = {"mean": mean_pool, "cls": cls_pool, "eos": eos_pool}


def load_model(config):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.hf_id, trust_remote_code=True, token=HF_TOKEN or None)
    kwargs = {"trust_remote_code": True, "token": HF_TOKEN or None}
    if config.dtype == "fp16":
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
    model = AutoModel.from_pretrained(config.hf_id, **kwargs)
    if config.dtype != "fp16":
        model = model.to(DEVICE)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def encode_final_layer(texts, model, tokenizer, config, batch_size=16):
    """Extract final-layer embeddings only (faster than all layers)."""
    pool_fn = POOL_FN[config.pooling]
    all_embs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                          max_length=config.max_length, return_tensors="pt")
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        pooled = pool_fn(hidden, inputs["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
        all_embs.append(pooled.cpu().numpy())
        del outputs, hidden
        torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0).astype(np.float32)


def apply_prefixes(texts, prefix="", instruction=""):
    if instruction:
        return [f"Instruct: {instruction}\nQuery: {t}" for t in texts]
    elif prefix:
        return [prefix + t for t in texts]
    return texts


def compute_mrr_recall(q_emb, d_emb, k=10):
    sim = q_emb @ d_emb.T
    n = sim.shape[0]
    ranks = np.array([(sim[i] > sim[i, i]).sum() + 1 for i in range(n)], dtype=float)
    rr = np.where(ranks <= k, 1.0 / ranks, 0.0)
    return {"MRR@10": float(rr.mean()), "Recall@10": float((ranks <= k).mean())}


def zca_whitening(train_emb, apply_embs: List[np.ndarray]):
    """
    Fit ZCA on train_emb, apply to each array in apply_embs.
    Returns list of whitened arrays.
    """
    train_emb = train_emb.astype(np.float32)
    mean_vec = train_emb.mean(axis=0)
    centered = train_emb - mean_vec
    cov = centered.T @ centered / (len(centered) - 1)
    U, S, Vt = np.linalg.svd(cov)
    W = U @ np.diag(1.0 / np.sqrt(S + 1e-5)) @ U.T

    results = []
    for emb in apply_embs:
        emb = emb.astype(np.float32)
        whitened = (emb - mean_vec) @ W.T
        whitened = whitened / np.linalg.norm(whitened, axis=1, keepdims=True).clip(1e-9)
        results.append(whitened)
    return results


def run_whitening_experiment(eval_data):
    """
    For each degraded model × corpus × query format:
      - Baseline: final-layer cosine retrieval
      - Transductive: ZCA fit on docs+queries
      - Corpus-only: ZCA fit on docs only
      - Held-out: ZCA fit on 80% of docs
    """
    rows = []

    for config in DEGRADED_MODELS:
        log.info(f"\n{'='*60}")
        log.info(f"Whitening variants: {config.name}")
        log.info(f"{'='*60}")

        batch_size = 2 if config.dtype == "fp16" else 16
        model, tokenizer = load_model(config)

        for corpus_name, corpus_data in eval_data.items():
            documents = corpus_data["documents"]
            for qf_name, queries in corpus_data["queries"].items():
                log.info(f"  {corpus_name}/{qf_name}")

                # Encode
                prefixed_q = apply_prefixes(queries, config.query_prefix, config.instruction)
                prefixed_d = apply_prefixes(documents, config.doc_prefix)
                q_emb = encode_final_layer(prefixed_q, model, tokenizer, config, batch_size)
                d_emb = encode_final_layer(prefixed_d, model, tokenizer, config, batch_size)

                # A) Baseline
                baseline = compute_mrr_recall(q_emb, d_emb)

                # B) Transductive: fit on docs + queries
                combined = np.vstack([d_emb, q_emb])
                d_trans, q_trans = zca_whitening(combined, [d_emb, q_emb])
                transductive = compute_mrr_recall(q_trans, d_trans)

                # C) Corpus-only: fit on docs only
                d_corpus, q_corpus = zca_whitening(d_emb, [d_emb, q_emb])
                corpus_only = compute_mrr_recall(q_corpus, d_corpus)

                # D) Held-out: fit on 80% of docs, apply to all
                rng = np.random.RandomState(SEED)
                n_docs = len(d_emb)
                n_train = int(0.8 * n_docs)
                train_idx = rng.choice(n_docs, n_train, replace=False)
                d_train = d_emb[train_idx]
                d_held, q_held = zca_whitening(d_train, [d_emb, q_emb])
                held_out = compute_mrr_recall(q_held, d_held)

                row = {
                    "model": config.name,
                    "corpus": corpus_name,
                    "query_format": qf_name,
                    "baseline_MRR@10": baseline["MRR@10"],
                    "baseline_Recall@10": baseline["Recall@10"],
                    "transductive_MRR@10": transductive["MRR@10"],
                    "transductive_Recall@10": transductive["Recall@10"],
                    "corpus_only_MRR@10": corpus_only["MRR@10"],
                    "corpus_only_Recall@10": corpus_only["Recall@10"],
                    "held_out_MRR@10": held_out["MRR@10"],
                    "held_out_Recall@10": held_out["Recall@10"],
                    # Deltas
                    "delta_transductive": transductive["MRR@10"] - baseline["MRR@10"],
                    "delta_corpus_only": corpus_only["MRR@10"] - baseline["MRR@10"],
                    "delta_held_out": held_out["MRR@10"] - baseline["MRR@10"],
                    # Retention: what fraction of transductive gain is retained?
                    "retention_corpus_only": (
                        (corpus_only["MRR@10"] - baseline["MRR@10"]) /
                        (transductive["MRR@10"] - baseline["MRR@10"])
                        if (transductive["MRR@10"] - baseline["MRR@10"]) > 0.01 else float("nan")
                    ),
                    "retention_held_out": (
                        (held_out["MRR@10"] - baseline["MRR@10"]) /
                        (transductive["MRR@10"] - baseline["MRR@10"])
                        if (transductive["MRR@10"] - baseline["MRR@10"]) > 0.01 else float("nan")
                    ),
                }
                rows.append(row)

                log.info(f"    Baseline:      {baseline['MRR@10']:.3f}")
                log.info(f"    Transductive:  {transductive['MRR@10']:.3f} (Δ={row['delta_transductive']:+.3f})")
                log.info(f"    Corpus-only:   {corpus_only['MRR@10']:.3f} (Δ={row['delta_corpus_only']:+.3f}, "
                         f"retention={row['retention_corpus_only']:.0%})")
                log.info(f"    Held-out 80%:  {held_out['MRR@10']:.3f} (Δ={row['delta_held_out']:+.3f}, "
                         f"retention={row['retention_held_out']:.0%})")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    result_df = pd.DataFrame(rows)
    result_df.to_parquet(SUPP_DIR / "whitening_variants.parquet", index=False)
    result_df.to_csv(SUPP_DIR / "whitening_variants.csv", index=False)
    return result_df


# ============================================================
# PART 2: VALIDATION CORPUS — 500 PMC-Patients (seed=123)
# ============================================================
# Different seed → different 500 docs (vs main experiment's 100)
# Generate queries via BM25-matching to existing query templates,
# or generate new metadata queries.
# Strategy: sample 500, use the same query generation approach.
# ============================================================

def load_pmc_validation(n=500, seed=123):
    """Load 500 PMC-Patients with a different seed than the main experiment."""
    log.info(f"Loading PMC-Patients validation set (n={n}, seed={seed})...")

    try:
        from datasets import load_dataset
        ds = load_dataset("zhengyun21/PMC-Patients", split="train", token=HF_TOKEN or None)
        df = ds.to_pandas()
    except Exception:
        from huggingface_hub import hf_hub_download
        json_path = hf_hub_download(repo_id="zhengyun21/PMC-Patients",
                                     filename="PMC-Patients-V2.json", repo_type="dataset",
                                     token=HF_TOKEN or None)
        df = pd.read_json(json_path, lines=False)

    text_col = next(c for c in ["patient", "text"] if c in df.columns)
    df = df.rename(columns={text_col: "text"})
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.len() > 50].reset_index(drop=True)
    log.info(f"  Total PMC-Patients: {len(df)}")

    # Sample with different seed
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    sampled = df.iloc[idx].reset_index(drop=True)
    log.info(f"  Sampled: {len(sampled)} docs (seed={seed})")

    # Exclude any overlap with main experiment (seed=42, n=100)
    rng_main = np.random.RandomState(42)
    main_idx = set(rng_main.choice(len(df), size=100, replace=False).tolist())
    val_idx = set(idx.tolist())
    overlap = val_idx & main_idx
    log.info(f"  Overlap with main experiment: {len(overlap)} docs")

    return sampled


def generate_heuristic_queries(df):
    """
    Generate keyword and NL queries from document text using heuristics.
    Matches Paper 1's approach: capitalized medical terms for KW,
    first 1-2 sentences for NL.
    """
    kw_queries = []
    nl_queries = []

    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        words = text.split()

        # Keyword: extract capitalized medical terms
        skip = {"this","that","with","from","were","have","been","patient","year",
                "years","history","presented","admission","hospital","examination","showed",
                "the","was","for","and","had","she","his","her","our"}
        med_terms = []
        for w in words:
            clean = w.strip(".,;:()")
            if len(clean) > 3 and (clean[0].isupper() or clean.isupper()):
                if clean.lower() not in skip:
                    med_terms.append(clean)
            if len(med_terms) >= 6:
                break
        if not med_terms:
            med_terms = words[:5]
        kw_queries.append(" ".join(med_terms))

        # NL: first 1-2 sentences
        sents = text.replace("\n", " ").split(". ")
        nl_q = ". ".join(sents[:2])
        if len(nl_q.split()) > 100:
            nl_q = " ".join(nl_q.split()[:100])
        nl_queries.append(nl_q)

    return kw_queries, nl_queries


def run_validation_experiment(val_docs, val_kw, val_nl):
    """
    Run the core layer-level analysis on the validation corpus.
    Extract all layers, compute metrics — same pipeline as main experiment.
    """
    # Import extraction functions from main script or redefine
    from transformers import AutoModel, AutoTokenizer

    ALL_MODELS = [
        ModelConfig(name="BioBERT", hf_id="dmis-lab/biobert-v1.1", category="Domain Encoder", n_layers=12),
        ModelConfig(name="ClinicalBERT", hf_id="medicalai/ClinicalBERT", category="Domain Encoder", n_layers=6),
        ModelConfig(name="BioLORD-2023", hf_id="FremyCompany/BioLORD-2023", category="Biomedical Retriever", n_layers=12),
        ModelConfig(name="BGE-base", hf_id="BAAI/bge-base-en-v1.5", category="General Embedding", n_layers=12),
        ModelConfig(name="GTE-base", hf_id="thenlper/gte-base", category="General Embedding", n_layers=12),
        ModelConfig(name="Nomic-embed-text", hf_id="nomic-ai/nomic-embed-text-v1.5", category="General Embedding",
                    query_prefix="search_query: ", doc_prefix="search_document: ", n_layers=12),
        ModelConfig(name="E5-Mistral-7B", hf_id="intfloat/e5-mistral-7b-instruct", category="General LLM",
                    pooling="eos", instruction="Given a clinical note, retrieve the most relevant clinical document.",
                    dtype="fp16", max_length=4096, n_layers=32),
        ModelConfig(name="Phi-3-mini", hf_id="microsoft/Phi-3-mini-4k-instruct", category="General LLM",
                    pooling="mean", dtype="fp16", max_length=4096, n_layers=32),
    ]

    eval_data = {
        "PMC-Val-500": {
            "documents": val_docs,
            "queries": {"keyword": val_kw, "natural_language": val_nl},
        }
    }

    all_rows = []

    for config in ALL_MODELS:
        checkpoint = SUPP_DIR / f"val_{config.name.replace(' ','_')}_layers.parquet"
        if checkpoint.exists():
            log.info(f"SKIP {config.name} — checkpoint exists")
            continue

        log.info(f"\nValidation: {config.name}")
        batch_size = 2 if config.dtype == "fp16" else 16
        model, tokenizer = load_model(config)
        model_rows = []

        for qf_name, queries in eval_data["PMC-Val-500"]["queries"].items():
            documents = eval_data["PMC-Val-500"]["documents"]
            prefixed_q = apply_prefixes(queries, config.query_prefix, config.instruction)
            prefixed_d = apply_prefixes(documents, config.doc_prefix)

            # Extract all layers
            pool_fn = POOL_FN[config.pooling]
            n_expected = config.n_layers + 1
            q_layers = {i: [] for i in range(n_expected)}
            d_layers = {i: [] for i in range(n_expected)}

            for texts, layer_dict in [(prefixed_q, q_layers), (prefixed_d, d_layers)]:
                for start in range(0, len(texts), batch_size):
                    batch = texts[start:start+batch_size]
                    inputs = tokenizer(batch, padding=True, truncation=True,
                                      max_length=config.max_length, return_tensors="pt")
                    dev = next(model.parameters()).device
                    inputs = {k: v.to(dev) for k, v in inputs.items()}

                    try:
                        with torch.no_grad():
                            outputs = model(**inputs, output_hidden_states=True)
                        hidden_states = outputs.hidden_states
                    except TypeError:
                        with torch.no_grad():
                            outputs = model(**inputs)
                        hidden_states = [outputs.last_hidden_state]  # fallback

                    if start == 0 and len(hidden_states) != n_expected:
                        n_expected = len(hidden_states)
                        layer_dict.clear()
                        for i in range(n_expected):
                            layer_dict[i] = []

                    for li in range(len(hidden_states)):
                        pooled = pool_fn(hidden_states[li], inputs["attention_mask"])
                        pooled = F.normalize(pooled, p=2, dim=1)
                        layer_dict[li].append(pooled.cpu().numpy().astype(np.float32))

                    del outputs
                    torch.cuda.empty_cache()

                for li in layer_dict:
                    if layer_dict[li]:
                        layer_dict[li] = np.concatenate(layer_dict[li], axis=0)

            # Compute metrics per layer
            for li in sorted(d_layers.keys()):
                if not isinstance(d_layers[li], np.ndarray):
                    continue
                d_e = d_layers[li]
                q_e = q_layers[li]

                # Retrieval
                ret = compute_mrr_recall(q_e, d_e)

                # Geometry
                d_f32 = d_e.astype(np.float32)
                centered = d_f32 - d_f32.mean(axis=0)
                try:
                    _, S, _ = np.linalg.svd(centered, full_matrices=False)
                    S_sq = S ** 2
                    pr = float((S_sq.sum())**2 / (S_sq**2).sum())
                except:
                    pr = float("nan")

                # Avg cosine
                rng = np.random.RandomState(SEED)
                n = d_f32.shape[0]
                ia = rng.randint(0, n, 10000)
                ib = rng.randint(0, n, 10000)
                mask = ia != ib
                cos = float(np.sum(d_f32[ia[mask]] * d_f32[ib[mask]], axis=1).mean())

                model_rows.append({
                    "model": config.name, "category": config.category,
                    "corpus": "PMC-Val-500", "query_format": qf_name,
                    "layer": li, "MRR@10": ret["MRR@10"], "Recall@10": ret["Recall@10"],
                    "participation_ratio": pr, "avg_cosine": cos,
                })

            qf_short = "KW" if qf_name == "keyword" else "NL"
            mrrs = [r["MRR@10"] for r in model_rows if r["query_format"] == qf_name]
            if mrrs:
                best_idx = np.argmax(mrrs)
                log.info(f"  {qf_short}: L0={mrrs[0]:.3f} → best@L{best_idx}={mrrs[best_idx]:.3f} → final={mrrs[-1]:.3f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

        df_model = pd.DataFrame(model_rows)
        df_model.to_parquet(checkpoint, index=False)
        all_rows.extend(model_rows)
        log.info(f"  Saved {len(model_rows)} rows")

    # Combine all
    if all_rows:
        combined = pd.DataFrame(all_rows)
    else:
        # Load from checkpoints
        dfs = []
        for f in SUPP_DIR.glob("val_*_layers.parquet"):
            dfs.append(pd.read_parquet(f))
        combined = pd.concat(dfs, ignore_index=True)

    combined.to_parquet(SUPP_DIR / "validation_500_combined.parquet", index=False)
    combined.to_csv(SUPP_DIR / "validation_500_combined.csv", index=False)
    log.info(f"\nValidation complete: {len(combined)} rows")
    return combined


def analyze_validation(val_df, main_df=None):
    """Compare validation results with main experiment patterns."""
    log.info("\n" + "="*70)
    log.info("VALIDATION ANALYSIS")
    log.info("="*70)

    # 1. Does the U-shape hold?
    log.info("\n1. U-shape check (avg MRR by relative depth):")
    for model in sorted(val_df["model"].unique()):
        sub = val_df[val_df["model"] == model].groupby("layer")["MRR@10"].mean().sort_index()
        n = len(sub)
        first = sub.iloc[0]
        mid = sub.iloc[n//3:2*n//3].mean()
        final = sub.iloc[-1]
        shape = "U-SHAPE" if (first > mid * 1.3 and final > mid * 1.3) else "OTHER"
        log.info(f"  {model:25s} | L0={first:.3f} mid={mid:.3f} final={final:.3f} | {shape}")

    # 2. Does PR correlate with MRR at final layer?
    from scipy.stats import spearmanr
    final = val_df.groupby("model").apply(lambda g: g[g["layer"]==g["layer"].max()]).reset_index(drop=True)
    summary = final.groupby("model").agg(mrr=("MRR@10","mean"), pr=("participation_ratio","mean")).reset_index()
    r, p = spearmanr(summary["pr"], summary["mrr"])
    log.info(f"\n2. Final-layer PR vs MRR: ρ={r:.3f}, p={p:.3f}")
    for _, row in summary.sort_values("mrr", ascending=False).iterrows():
        log.info(f"  {row['model']:25s} | MRR={row['mrr']:.3f} PR={row['pr']:.1f}")

    # 3. Recovery ratio
    log.info(f"\n3. Recovery ratios (validation):")
    for model in sorted(val_df["model"].unique()):
        sub = val_df[val_df["model"] == model].groupby("layer")["MRR@10"].mean().sort_index()
        n = len(sub)
        q = max(2, n//4)
        early = sub.iloc[:q].max()
        trough = sub.iloc[q:3*q].min()
        final = sub.iloc[-1]
        depth = early - trough
        recovery = (final - trough) / depth if depth > 0.01 else float("nan")
        log.info(f"  {model:25s} | peak={early:.3f} trough={trough:.3f} final={final:.3f} recovery={recovery:.2f}")

    # 4. If main experiment data available, compare rankings
    if main_df is not None:
        log.info(f"\n4. Rank correlation: main vs validation (final-layer MRR):")
        main_final = main_df[main_df["length_tercile"]=="all"].groupby("model").apply(
            lambda g: g[g["layer"]==g["layer"].max()]
        ).reset_index(drop=True)
        main_rank = main_final.groupby("model")["MRR@10"].mean()
        val_rank = final.groupby("model")["MRR@10"].mean()
        common = main_rank.index.intersection(val_rank.index)
        if len(common) >= 4:
            from scipy.stats import kendalltau
            tau, p = kendalltau(main_rank[common].values, val_rank[common].values)
            log.info(f"  Kendall τ = {tau:.3f} (p={p:.3f})")
            rho, p2 = spearmanr(main_rank[common].values, val_rank[common].values)
            log.info(f"  Spearman ρ = {rho:.3f} (p={p2:.3f})")


# ============================================================
# MAIN: Run both experiments
# ============================================================
def load_main_eval_data():
    """Load the same eval data as the main experiment."""
    QUERIES_PATH = Path("/content/paper1_repo/metadata_queries.json")
    SYNTH_PATH = Path("/content/paper1_repo/synthetic_notes.csv")
    MT_PATH = DRIVE_BASE / "mtsamples_sample.csv"

    with open(QUERIES_PATH) as f:
        queries = json.load(f)

    # MTSamples
    mt = pd.read_csv(MT_PATH)
    mt_docs = mt["text"].dropna().tolist()[:100]

    # PMC-Patients (seed=42)
    from huggingface_hub import hf_hub_download
    try:
        from datasets import load_dataset
        ds = load_dataset("zhengyun21/PMC-Patients", split="train", token=HF_TOKEN or None)
        pmc_df = ds.to_pandas()
    except:
        pmc_path = hf_hub_download(repo_id="zhengyun21/PMC-Patients",
                                    filename="PMC-Patients-V2.json", repo_type="dataset",
                                    token=HF_TOKEN or None)
        pmc_df = pd.read_json(pmc_path, lines=False)
    text_col = next(c for c in ["patient","text"] if c in pmc_df.columns)
    pmc_df = pmc_df.rename(columns={text_col: "text"}).dropna(subset=["text"])
    pmc_df = pmc_df[pmc_df["text"].str.len() > 50].reset_index(drop=True)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(pmc_df), 100, replace=False)
    pmc_docs = pmc_df.iloc[idx]["text"].tolist()

    # Synthetic (BM25 alignment)
    from rank_bm25 import BM25Okapi
    synth = pd.read_csv(SYNTH_PATH).dropna(subset=["text"])
    all_synth = synth["text"].tolist()
    tokenized = [d.lower().split() for d in all_synth]
    bm25 = BM25Okapi(tokenized)
    syn_kw = queries["Synthetic"]["keyword"]
    mapping = [int(np.argmax(bm25.get_scores(q.lower().split()))) for q in syn_kw]
    syn_docs = [all_synth[i] for i in mapping]

    return {
        "MTSamples": {"documents": mt_docs, "queries": {
            "keyword": queries["MTSamples"]["keyword"][:100],
            "natural_language": queries["MTSamples"]["natural_language"][:100]}},
        "PMC-Patients": {"documents": pmc_docs, "queries": {
            "keyword": queries["PMC-Patients"]["keyword"][:100],
            "natural_language": queries["PMC-Patients"]["natural_language"][:100]}},
        "Synthetic": {"documents": syn_docs, "queries": {
            "keyword": queries["Synthetic"]["keyword"][:100],
            "natural_language": queries["Synthetic"]["natural_language"][:100]}},
    }


def main():
    log.info("="*70)
    log.info("SUPPLEMENTARY EXPERIMENTS")
    log.info("="*70)

    # ── Part 1: Non-transductive whitening ──
    log.info("\n" + "="*70)
    log.info("PART 1: Whitening variants (transductive vs corpus-only vs held-out)")
    log.info("="*70)
    eval_data = load_main_eval_data()
    wv_df = run_whitening_experiment(eval_data)

    # Summary
    log.info("\nWhitening variant summary (mean across conditions):")
    summary = wv_df.groupby("model").agg(
        baseline=("baseline_MRR@10", "mean"),
        transductive=("transductive_MRR@10", "mean"),
        corpus_only=("corpus_only_MRR@10", "mean"),
        held_out=("held_out_MRR@10", "mean"),
        retention_corpus=("retention_corpus_only", "mean"),
        retention_held=("retention_held_out", "mean"),
    )
    for model, row in summary.iterrows():
        log.info(f"  {model:20s} | base={row['baseline']:.3f} trans={row['transductive']:.3f} "
                 f"corpus={row['corpus_only']:.3f} ({row['retention_corpus']:.0%}) "
                 f"held={row['held_out']:.3f} ({row['retention_held']:.0%})")

    # ── Part 2: Validation corpus ──
    log.info("\n" + "="*70)
    log.info("PART 2: Validation corpus (500 PMC-Patients, seed=123)")
    log.info("="*70)
    val_df = load_pmc_validation(n=500, seed=123)
    val_docs = val_df["text"].tolist()
    val_kw, val_nl = generate_heuristic_queries(val_df)
    log.info(f"  Generated {len(val_kw)} keyword + {len(val_nl)} NL queries")

    # BM25 sanity check
    from rank_bm25 import BM25Okapi
    tokenized = [d.lower().split() for d in val_docs]
    bm25 = BM25Okapi(tokenized)
    ranks = [(bm25.get_scores(q.lower().split()) > bm25.get_scores(q.lower().split())[i]).sum() + 1
             for i, q in enumerate(val_kw)]
    bm25_mrr = np.mean([1/r if r <= 10 else 0 for r in ranks])
    log.info(f"  BM25 sanity check: MRR@10 = {bm25_mrr:.3f}")

    # Run layer analysis
    val_results = run_validation_experiment(val_docs, val_kw, val_nl)

    # Load main results for comparison
    try:
        main_results = pd.read_parquet(RESULTS_DIR / "all_models_combined.parquet")
    except:
        main_results = None

    analyze_validation(val_results, main_results)

    log.info("\n" + "="*70)
    log.info("✅ SUPPLEMENTARY EXPERIMENTS COMPLETE")
    log.info(f"Output: {SUPP_DIR}")
    log.info("="*70)


if __name__ == "__main__":
    main()
