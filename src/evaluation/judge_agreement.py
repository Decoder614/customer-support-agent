"""Statistical Inter-Annotator Agreement Harness for LLM Judge Calibration.

Measures alignment between human-verified ratings and automated judge ratings across
rubric dimensions using Quadratic-Weighted Cohen's Kappa, Pearson r, and Spearman rho.
Outputs structured metrics to artifacts/judge_agreement.json.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

from src.evaluation.llm_judge import JUDGE_DIMENSIONS, evaluate_with_judge


def compute_judge_agreement(
    human_df: pd.DataFrame,
    judge_df: Optional[pd.DataFrame] = None,
    dimensions: Optional[List[str]] = None,
    artifacts_dir: Union[str, Path] = "artifacts",
) -> Dict[str, Any]:
    """Computes inter-annotator agreement statistics between human and automated judge ratings.

    Args:
        human_df: DataFrame containing human expert ratings (1 to 5).
        judge_df: Optional DataFrame with automated judge scores. If None, computes on human_df.
        dimensions: List of rubric dimension names to evaluate.
        artifacts_dir: Directory to persist agreement results.

    Returns:
        Structured agreement metrics dictionary.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    eval_dimensions = dimensions or [d for d in JUDGE_DIMENSIONS if d in human_df.columns]

    # If judge_df is not supplied, run automated judge scoring on the human evaluation samples
    if judge_df is None or len(judge_df) == 0:
        logging.info("Generating automated judge ratings for human evaluation sample...")
        judge_rows = []
        for _, row in human_df.iterrows():
            pred = {
                "reply": str(row.get("reply", "")),
                "action": str(row.get("action", "AUTO_HANDLE")),
                "intent": "lost_and_found",
                "intent_confidence": 0.90,
                "escalation_reason": str(row.get("escalation_reason", "")),
                "retrieved_examples": json.loads(row.get("evidence", "[]")) if isinstance(row.get("evidence"), str) and row.get("evidence").startswith("[") else [],
            }
            score_dict = evaluate_with_judge(
                customer_message=str(row.get("customer_message", "")),
                conversation_context=str(row.get("conversation_context", "")),
                prediction=pred,
            )
            judge_rows.append({
                "example_id": str(row.get("example_id", "")),
                "system": str(row.get("system", "")),
                "reply_sha256": str(row.get("reply_sha256", "")),
                **{d: score_dict.get(d, 4) for d in eval_dimensions},
                "overall_score": score_dict.get("overall_score", 4.0),
                "source": "llm_judge",
            })
        judge_df = pd.DataFrame(judge_rows)
        judge_df.to_csv(artifacts_dir / "llm_judge_scores.csv", index=False)

    # Validate schema
    for col in eval_dimensions:
        if col not in human_df.columns or col not in judge_df.columns:
            raise ValueError(f"Dimension '{col}' missing from human or judge DataFrame.")

    results: Dict[str, Any] = {
        "n_samples": len(human_df),
        "dimensions": {},
    }

    kappa_list = []
    pearson_list = []
    exact_list = []
    within_one_list = []

    import warnings

    for dim in eval_dimensions:
        a = pd.to_numeric(human_df[dim], errors="coerce").fillna(4).to_numpy().astype(int)
        b = pd.to_numeric(judge_df[dim], errors="coerce").fillna(4).to_numpy().astype(int)

        exact_agr = float(np.mean(a == b))
        within_one_agr = float(np.mean(np.abs(a - b) <= 1))

        # Check for non-constant arrays for correlation and kappa calculations
        if len(set(a)) == 1 or len(set(b)) == 1:
            if np.array_equal(a, b):
                p_val = 1.0
                s_val = 1.0
                qwk_val = 1.0
            else:
                p_val = 0.0
                s_val = 0.0
                qwk_val = 0.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_val = float(pearsonr(a, b).statistic)
                s_val = float(spearmanr(a, b).statistic)
                qwk_val = float(cohen_kappa_score(a, b, labels=[1, 2, 3, 4, 5], weights="quadratic"))

        exact_list.append(exact_agr)
        within_one_list.append(within_one_agr)
        if qwk_val is not None and not np.isnan(qwk_val):
            kappa_list.append(qwk_val)
        if p_val is not None and not np.isnan(p_val):
            pearson_list.append(p_val)

        results["dimensions"][dim] = {
            "exact_agreement": round(exact_agr, 4),
            "within_one_agreement": round(within_one_agr, 4),
            "pearson_r": round(p_val, 4) if p_val is not None else None,
            "spearman_rho": round(s_val, 4) if s_val is not None else None,
            "quadratic_weighted_kappa": round(qwk_val, 4) if qwk_val is not None else None,
        }

    results["macro_averages"] = {
        "mean_exact_agreement": round(float(np.mean(exact_list)), 4) if exact_list else 0.0,
        "mean_within_one": round(float(np.mean(within_one_list)), 4) if within_one_list else 0.0,
        "mean_quadratic_kappa": round(float(np.mean(kappa_list)), 4) if kappa_list else 0.0,
        "mean_pearson_r": round(float(np.mean(pearson_list)), 4) if pearson_list else 0.0,
    }

    # Save to artifacts
    out_file = artifacts_dir / "judge_agreement.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Saved inter-annotator agreement metrics to {out_file}")

    # Print summary table
    print(f"\n{'='*85}")
    print(f"  Inter-Annotator Agreement: Human vs. LLM Judge Calibration (N = {len(human_df)})")
    print(f"{'='*85}")
    print(f"  {'Dimension':<30} | {'Exact (%)':>10} | {'Within +-1':>10} | {'Pearson r':>10} | {'Quad Kappa':>12}")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
    for dim, stats in results["dimensions"].items():
        ex_str = f"{stats['exact_agreement']*100:.1f}%"
        w1_str = f"{stats['within_one_agreement']*100:.1f}%"
        pr_str = f"{stats['pearson_r']:.3f}" if stats["pearson_r"] is not None else "N/A"
        kp_str = f"{stats['quadratic_weighted_kappa']:.3f}" if stats["quadratic_weighted_kappa"] is not None else "N/A"
        print(f"  {dim:<30} | {ex_str:>10} | {w1_str:>10} | {pr_str:>10} | {kp_str:>12}")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
    macro = results["macro_averages"]
    print(f"  {'Macro Average':<30} | {macro['mean_exact_agreement']*100:>9.1f}% | {macro['mean_within_one']*100:>9.1f}% | {macro['mean_pearson_r']:>10.3f} | {macro['mean_quadratic_kappa']:>12.3f}")
    print(f"{'='*85}\n")

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", default="data/eval/judge_human_eval_completed.csv", help="Completed human evaluation CSV")
    parser.add_argument("--judge", default="artifacts/llm_judge_scores.csv", help="Automated judge scores CSV")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    args = parser.parse_args()

    human_path = Path(args.human)
    if not human_path.exists():
        raise FileNotFoundError(f"Human evaluation benchmark file not found at {human_path}")

    human_df = pd.read_csv(human_path, dtype=str, keep_default_na=False)

    judge_path = Path(args.judge)
    judge_df = pd.read_csv(judge_path, dtype=str, keep_default_na=False) if judge_path.exists() else None

    compute_judge_agreement(
        human_df=human_df,
        judge_df=judge_df,
        artifacts_dir=args.artifacts_dir,
    )


if __name__ == "__main__":
    main()
