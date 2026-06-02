"""
LME grouping diagnostic: positional vs corpus-unique query indexing
=====================================================================

Demonstrates why per_model_summary_CORRECTED.parquet uses corpus-unique
grouping rather than positional query_idx. Runs both specifications side
by side and reports convergence and ICC for each.

Background
----------
In the long-format per_query_ranks.parquet, the column query_idx runs 0-99
within each corpus and is reused across the three corpora (MTSamples,
PMC-Patients, Synthetic). A random intercept keyed on query_idx alone
therefore pools three distinct clinical queries under each value (one
from each corpus). The appropriate grouping prepends the corpus name:

    corpus_query_idx = corpus + "_" + str(query_idx)   # 300 groups per model

What this diagnostic shows
--------------------------
Running both groupings on identical input data:

- POSITIONAL grouping: positive variance components (~0.05-0.23), but
  conceptually wrong (pools across corpora)
- CORPUS-UNIQUE grouping: positive variance components (~0.06-0.55),
  conceptually correct, all 13 models converge

The originally submitted per_model_summary.parquet showed exact-zero
variance components and non-finite log-likelihoods across all models.
That was a degenerate fit independent of grouping choice — a clean
refit under either specification produces well-behaved results, but
the corpus-unique grouping is the correct one for the analysis.

See ../../docs/analysis_decisions.md §1 for the full story.
"""

import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

DEFAULT_INPUT = "/content/drive/MyDrive/Content/paper2_layer_analysis/revision/stream_c/rank_data/per_query_ranks.parquet"

FORMULA = "log_rank ~ C(intervention) + C(corpus) + C(query_format) + rel_layer"


def fit_one(subset: pd.DataFrame, groups: pd.Series, label: str) -> dict:
    """Fit the LME and return summary stats."""
    try:
        res = smf.mixedlm(
            FORMULA, data=subset, groups=groups,
            re_formula="~rel_layer",
        ).fit(reml=True, method="lbfgs")

        gv = float(res.cov_re.iloc[0, 0])
        resid = float(res.scale)
        icc = gv / (gv + resid) if (gv + resid) > 0 else float("nan")
        return {
            "spec": label,
            "n_groups": groups.nunique(),
            "group_var": round(gv, 4),
            "residual_var": round(resid, 4),
            "ICC": round(icc, 4),
            "log_likelihood": round(res.llf, 1),
            "converged": bool(res.converged),
        }
    except Exception as e:
        return {
            "spec": label,
            "n_groups": groups.nunique(),
            "group_var": None, "residual_var": None, "ICC": None,
            "log_likelihood": None, "converged": False,
            "error": str(e)[:80],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    df["log_rank"] = np.log(df["rank"].astype(float) + 1.0)
    df["corpus_query_idx"] = df["corpus"].astype(str) + "_" + df["query_idx"].astype(str)

    print(f"Loaded {len(df):,} rows, {df['model'].nunique()} models")
    print(f"Positional query_idx unique values: {df['query_idx'].nunique()}")
    print(f"Corpus-unique values: {df['corpus_query_idx'].nunique()}")
    print()

    results = []
    for model in sorted(df["model"].unique()):
        s = df[df["model"] == model]
        pos = fit_one(s, s["query_idx"], "positional")
        uniq = fit_one(s, s["corpus_query_idx"], "corpus-unique")

        for r in (pos, uniq):
            r["model"] = model
            results.append(r)

        print(f"{model:<24} "
              f"positional: ICC={pos['ICC']}, conv={pos['converged']} | "
              f"corpus-unique: ICC={uniq['ICC']}, conv={uniq['converged']}")

    out = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(out.groupby("spec")[["ICC", "converged"]].agg(
        {"ICC": ["min", "max", "median"], "converged": "sum"}
    ))


if __name__ == "__main__":
    main()
