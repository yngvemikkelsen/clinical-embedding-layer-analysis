"""
Per-query layer-wise linear mixed-effects refit (corrected grouping)
=====================================================================

Produces the authoritative LME outputs for the v45 manuscript:
  - per_model_summary_CORRECTED.parquet  (variance components, ICC, convergence)
  - per_model_fixed_effects_CORRECTED.parquet  (fixed-effect coefficients, SEs, P-values)

These supersede the earlier per_model_summary.parquet and
per_model_fixed_effects.parquet, which reflected a degenerate fit
(zero variance components, non-finite log-likelihoods).

See ../../docs/analysis_decisions.md §1 for the full story.

Inputs
------
per_query_ranks.parquet  (long-format, ~437,400 rows)
    Columns: model, intervention, corpus, query_format, query_idx,
             rel_layer, layer, rank

Statistical specification
-------------------------
For each of the 13 model configurations, fit:

    log(rank + 1) ~ C(intervention) + C(corpus) + C(query_format) + rel_layer
                  + (rel_layer | corpus_query_idx)

where `corpus_query_idx = corpus + "_" + str(query_idx)` is the
corpus-unique query identifier (300 groups per model = 100 queries × 3 corpora).

This grouping treats each query as a distinct group within its source corpus.
The earlier specification used positional query_idx (0–99) without corpus
prefix, which pooled three distinct clinical queries under each index value.

Compute
-------
- Runtime: ~5-10 minutes for all 13 models on CPU
- Memory: <2 GB
- statsmodels 0.14 with method='lbfgs', reml=True
"""

import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# Default paths — override via CLI args for non-Colab environments
DEFAULT_INPUT = "/content/drive/MyDrive/Content/paper2_layer_analysis/revision/stream_c/rank_data/per_query_ranks.parquet"
DEFAULT_OUTPUT_DIR = "/content/drive/MyDrive/Content/paper2_layer_analysis/revision/stream_c/rank_data/"

FORMULA = "log_rank ~ C(intervention) + C(corpus) + C(query_format) + rel_layer"


def refit_all_models(input_path: str, output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit the per-query LME for all configurations with corpus-unique grouping.

    Returns
    -------
    summary : DataFrame
        One row per model with variance components, ICC, and convergence flag.
    fixed_effects : DataFrame
        One row per (model, term) with estimate, SE, z-value, p-value.
    """
    df = pd.read_parquet(input_path)
    df["log_rank"] = np.log(df["rank"].astype(float) + 1.0)
    df["corpus_query_idx"] = df["corpus"].astype(str) + "_" + df["query_idx"].astype(str)

    summary_rows = []
    fe_rows = []

    for model in sorted(df["model"].unique()):
        subset = df[df["model"] == model]

        try:
            res = smf.mixedlm(
                FORMULA,
                data=subset,
                groups=subset["corpus_query_idx"],
                re_formula="~rel_layer",
            ).fit(reml=True, method="lbfgs")

            group_var = float(res.cov_re.iloc[0, 0])
            resid_var = float(res.scale)
            icc = group_var / (group_var + resid_var)

            summary_rows.append({
                "model": model,
                "n_obs": len(subset),
                "n_groups": subset["corpus_query_idx"].nunique(),
                "random_effect_var": round(group_var, 4),
                "residual_var": round(resid_var, 4),
                "ICC": round(icc, 4),
                "log_likelihood": round(res.llf, 1),
                "converged": bool(res.converged),
            })

            for term, est, se, z, p in zip(
                res.params.index, res.params.values,
                res.bse.values, res.tvalues.values, res.pvalues.values,
            ):
                if term in ("Group Var", "rel_layer Var", "rel_layer Cov"):
                    continue
                fe_rows.append({
                    "model": model, "term": term,
                    "estimate": round(est, 4), "std_error": round(se, 4),
                    "z_value": round(z, 3), "p_value": p,
                })

            print(f"{model:<24} ICC={icc:.3f}  GrpVar={group_var:.4f}  "
                  f"resid={resid_var:.4f}  LL={res.llf:.0f}  conv={res.converged}")

        except Exception as e:
            print(f"{model:<24} FAILED: {str(e)[:80]}")
            summary_rows.append({
                "model": model,
                "n_obs": len(subset),
                "n_groups": subset["corpus_query_idx"].nunique(),
                "random_effect_var": None, "residual_var": None, "ICC": None,
                "log_likelihood": None, "converged": False,
            })

    summary = pd.DataFrame(summary_rows)
    fixed_effects = pd.DataFrame(fe_rows)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(out / "per_model_summary_CORRECTED.parquet")
    fixed_effects.to_parquet(out / "per_model_fixed_effects_CORRECTED.parquet")

    return summary, fixed_effects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Path to per_query_ranks.parquet")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write CORRECTED parquets")
    args = parser.parse_args()

    summary, fixed_effects = refit_all_models(args.input, args.output_dir)

    print("\n=== CORRECTED SUMMARY ===")
    print(summary.to_string(index=False))

    print(f"\nICC range: {summary['ICC'].min():.3f} to {summary['ICC'].max():.3f}, "
          f"median {summary['ICC'].median():.3f}")
    print(f"All converged: {summary['converged'].all()}")


if __name__ == "__main__":
    main()
