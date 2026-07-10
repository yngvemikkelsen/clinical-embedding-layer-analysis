#!/usr/bin/env python3
"""
Paper 12 v19 — Prepare input data
=================================
ONE script. Run it, get the four parquet files the v19 validation script
needs. No patches, no follow-up steps.

Pipeline:
  1. Read three uploaded files from INPUT_DIR:
       - pmc_patients_sample.csv         (>=500 rows, column 'text')
       - mtsamples_sample.csv            (>=500 rows, column 'text')
       - metadata_queries_v17.json       (existing 100 queries per corpus,
                                          nested shape {corpus: {keyword:[],
                                          natural_language:[]}})
  2. For each corpus (PMC-Patients, MTSamples):
       For positions 100..499 of the CSV:
         a. Extract metadata via GPT-4o (temperature=0)
         b. Generate keyword and natural-language queries via GPT-4o
            (temperature=0.3)
       Prompts and temperatures are byte-identical to the v17 generator
       (metadata_query_colab.py).
  3. Merge the new 400 query pairs per corpus with the existing 100 from
     v17 to make 500 per corpus. v17 queries occupy positions 0..99 of the
     merged list; extension queries occupy positions 100..499. This is
     positional 1-to-1 alignment with the CSV rows.
  4. Write the four parquet files the v19 script reads:
       pmc500_docs.parquet
       mtsamples500_docs.parquet
       pmc500_queries.parquet
       mtsamples500_queries.parquet
     Schema:
       docs   : doc_id (int 0..499), document_text (str)
       queries: query_id (int 0..499), query_keyword (str),
                query_natural_language (str), relevant_doc_id (int 0..499)
  5. Also write the merged JSON (metadata_queries_500.json) for archival
     reproducibility — same nested shape as v17 input.

Requirements:
  - OPENAI_API_KEY environment variable set
  - pandas, openai, tqdm, pyarrow

Cost (gpt-4o):
  - 800 docs * 3 calls each (1 metadata + 2 query generations)
  - ~$10-15 total

Runtime: ~30-45 minutes on a stable network.

Run:
  export OPENAI_API_KEY=sk-...
  python paper12_v19_prepare_inputs.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
INPUT_DIR = Path("/content/sample_data")
OUTPUT_DIR = Path("/content/sample_data/v19_inputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PMC_CSV   = INPUT_DIR / "pmc_patients_sample.csv"
MT_CSV    = INPUT_DIR / "mtsamples_sample.csv"
V17_JSON  = INPUT_DIR / "metadata_queries_v17.json"

N_DOCS = 500
EXT_START = 100  # extension generates positions 100..499

CORPORA = {
    "PMC-Patients": {
        "csv": PMC_CSV,
        "docs_out":    OUTPUT_DIR / "pmc500_docs.parquet",
        "queries_out": OUTPUT_DIR / "pmc500_queries.parquet",
    },
    "MTSamples": {
        "csv": MT_CSV,
        "docs_out":    OUTPUT_DIR / "mtsamples500_docs.parquet",
        "queries_out": OUTPUT_DIR / "mtsamples500_queries.parquet",
    },
}

MODEL = "gpt-4o"
META_TEMP = 0.0
QUERY_TEMP = 0.3
META_MAX_TOKENS = 300
QUERY_MAX_TOKENS = 150
RETRY_ATTEMPTS = 3
SLEEP_BETWEEN_CALLS = 0.3

# ============================================================
# PROMPTS — verbatim from metadata_query_colab.py (v17 generator)
# ============================================================
META_PROMPT = """Extract metadata from this clinical document as JSON:
{{"specialty":"...","note_type":"...","primary_diagnosis":"...","secondary_diagnoses":["..."],"patient_demographics":"..."}}
Return ONLY valid JSON.

Document:
{text}"""

NL_TPL = """Based ONLY on this metadata, write a natural language clinical question (1-2 sentences):
Specialty: {sp} | Note type: {nt} | Diagnosis: {dx} | Other: {sec} | Patient: {dem}
Query:"""

KW_TPL = """Based ONLY on this metadata, output 3-6 clinical search keywords (space-separated):
Specialty: {sp} | Note type: {nt} | Diagnosis: {dx} | Other: {sec} | Patient: {dem}
Keywords:"""

# ============================================================
# OPENAI CLIENT
# ============================================================
if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("ERROR: OPENAI_API_KEY environment variable is not set.")
client = OpenAI()


def get_metadata(text):
    for attempt in range(RETRY_ATTEMPTS):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                temperature=META_TEMP,
                max_tokens=META_MAX_TOKENS,
                messages=[{"role": "user",
                           "content": META_PROMPT.format(text=text[:3000])}],
            )
            raw = r.choices[0].message.content.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            return json.loads(raw)
        except Exception:
            time.sleep(2 ** attempt)
    return {"specialty": "Unknown", "note_type": "Clinical Note",
            "primary_diagnosis": "Unknown", "secondary_diagnoses": [],
            "patient_demographics": "adult"}


def gen_q(meta, tpl):
    sec = ", ".join(meta.get("secondary_diagnoses", []) or []) or "none"
    prompt = tpl.format(
        sp=meta.get("specialty", "Unknown"),
        nt=meta.get("note_type", "Note"),
        dx=meta.get("primary_diagnosis", "Unknown"),
        sec=sec,
        dem=meta.get("patient_demographics", "adult"),
    )
    for attempt in range(RETRY_ATTEMPTS):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                temperature=QUERY_TEMP,
                max_tokens=QUERY_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.choices[0].message.content.strip()
        except Exception:
            time.sleep(2 ** attempt)
    return meta.get("primary_diagnosis", "clinical query")


# ============================================================
# PIPELINE
# ============================================================
def generate_extension_queries(corpus_name, csv_path):
    """Generate 400 new query pairs (positions EXT_START..N_DOCS-1)."""
    print(f"\n--- Extension query generation: {corpus_name} "
          f"(positions {EXT_START}..{N_DOCS - 1}) ---")
    df = pd.read_csv(csv_path)
    if len(df) < N_DOCS:
        sys.exit(f"ERROR: {csv_path} has {len(df)} rows, need {N_DOCS}.")
    if "text" not in df.columns:
        sys.exit(f"ERROR: {csv_path} has no 'text' column. "
                 f"Columns: {list(df.columns)}")
    slice_df = df.iloc[EXT_START:N_DOCS].reset_index(drop=True)

    kw_queries = []
    nl_queries = []
    for i, row in tqdm(list(slice_df.iterrows()),
                       desc=f"{corpus_name}", total=len(slice_df)):
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        meta = get_metadata(text)
        time.sleep(SLEEP_BETWEEN_CALLS)
        nl = gen_q(meta, NL_TPL)
        time.sleep(SLEEP_BETWEEN_CALLS)
        kw = gen_q(meta, KW_TPL)
        time.sleep(SLEEP_BETWEEN_CALLS)
        kw_queries.append(kw)
        nl_queries.append(nl)
    print(f"  Generated {len(kw_queries)} keyword + {len(nl_queries)} NL "
          f"queries for {corpus_name}")
    return kw_queries, nl_queries


def load_v17_queries():
    if not V17_JSON.exists():
        sys.exit(f"ERROR: {V17_JSON} not found.")
    with open(V17_JSON) as f:
        v17 = json.load(f)
    for corpus in CORPORA:
        if corpus not in v17:
            sys.exit(f"ERROR: '{corpus}' missing in v17 JSON.")
        kw = v17[corpus].get("keyword", [])
        nl = v17[corpus].get("natural_language", [])
        if len(kw) != EXT_START or len(nl) != EXT_START:
            print(f"  WARN: {corpus} v17 has {len(kw)} kw / {len(nl)} nl "
                  f"(expected {EXT_START}/{EXT_START}). Proceeding.")
    return v17


def write_docs_parquet(corpus_name, csv_path, out_path):
    df = pd.read_csv(csv_path)
    sub = df.iloc[:N_DOCS].reset_index(drop=True)
    out = pd.DataFrame({
        "doc_id": list(range(N_DOCS)),
        "document_text": sub["text"].astype(str).tolist(),
    })
    out.to_parquet(out_path, index=False)
    print(f"  Docs parquet -> {out_path} ({len(out)} rows)")


def write_queries_parquet(corpus_name, kw_all, nl_all, out_path):
    if len(kw_all) != N_DOCS or len(nl_all) != N_DOCS:
        sys.exit(f"ERROR: {corpus_name} merged length = "
                 f"{len(kw_all)} kw / {len(nl_all)} nl (expected "
                 f"{N_DOCS}/{N_DOCS}).")
    out = pd.DataFrame({
        "query_id": list(range(N_DOCS)),
        "query_keyword": kw_all,
        "query_natural_language": nl_all,
        "relevant_doc_id": list(range(N_DOCS)),
    })
    out.to_parquet(out_path, index=False)
    print(f"  Queries parquet -> {out_path} ({len(out)} rows)")


def write_merged_json(merged):
    out_path = OUTPUT_DIR / "metadata_queries_500.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"  Merged JSON -> {out_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("Paper 12 v19 — single-script input preparation")
    print(f"Input dir : {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Model: {MODEL} | metadata temp={META_TEMP} "
          f"| query temp={QUERY_TEMP}")
    for name, cfg in CORPORA.items():
        if not cfg["csv"].exists():
            sys.exit(f"ERROR: missing input CSV {cfg['csv']}")

    v17 = load_v17_queries()

    merged_json = {}
    for corpus_name, cfg in CORPORA.items():
        ext_kw, ext_nl = generate_extension_queries(corpus_name, cfg["csv"])
        all_kw = list(v17[corpus_name]["keyword"]) + ext_kw
        all_nl = list(v17[corpus_name]["natural_language"]) + ext_nl
        merged_json[corpus_name] = {
            "keyword": all_kw,
            "natural_language": all_nl,
        }
        write_docs_parquet(corpus_name, cfg["csv"], cfg["docs_out"])
        write_queries_parquet(corpus_name, all_kw, all_nl, cfg["queries_out"])

    write_merged_json(merged_json)

    print("\nDone. Point the v19 script at these:")
    for corpus_name, cfg in CORPORA.items():
        print(f"  {corpus_name}: docs   = {cfg['docs_out']}")
        print(f"  {corpus_name}: queries = {cfg['queries_out']}")


if __name__ == "__main__":
    main()