"""Failure Analysis and Diagnostic Extractor for Uber Support Agent.

Performs systematic error categorization on model predictions, grouping failure modes
into structured diagnostic buckets with root cause hypotheses, remediation strategies,
and representative examples. Outputs structured logs to artifacts/failure_cases.json.
"""

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.agent.support_agent import SupportAgent, load_agent


FAILURE_DIAGNOSTICS: Dict[str, Tuple[str, str]] = {
    "ambiguous_or_multi_intent": (
        "Multiple lexical topics compete in customer query (e.g. overcharge and driver conduct); priority rules disagree with intent classifier.",
        "Refine intent disambiguation hierarchy and have humans label primary requested action.",
    ),
    "context_dependent_intent": (
        "Customer utterance in latest turn relies on critical antecedent context in earlier thread turns.",
        "Incorporate multi-turn dialogue context vectors directly into classification features.",
    ),
    "intent_confusion": (
        "Sparse text or out-of-distribution wording maps utterance to an incorrect intent category.",
        "Expand training dataset with diverse paraphrases and increase character n-gram capacity.",
    ),
    "unsupported_retrieval": (
        "Customer inquiry is legitimate but no matching historical precedent passed the 0.30 cosine similarity threshold.",
        "Expand reviewed historical knowledge base and incorporate dense semantic embeddings.",
    ),
    "over_escalation": (
        "Conservative policy threshold or missing retrieval step escalated an otherwise safe customer query.",
        "Calibrate confidence thresholds on annotated validation sets without compromising safety.",
    ),
    "unsafe_auto_handle": (
        "Agent auto-handled a query that ground truth annotation marked for mandatory human escalation.",
        "Add explicit safety keyword triggers and reduce auto-handling confidence thresholds for sensitive domains.",
    ),
}


def extract_failure_cases(
    df: pd.DataFrame,
    predictions: List[Dict[str, Any]],
    intent_col: str = "intent_gold",
    action_col: str = "action_gold",
    artifacts_dir: Union[str, Path] = "artifacts",
) -> Dict[str, Any]:
    """Categorizes model and policy failures into structured diagnostic buckets.

    Args:
        df: Input benchmark DataFrame with ground truth labels.
        predictions: List of SupportAgent prediction dictionaries.
        intent_col: Column name for gold intent.
        action_col: Column name for gold action.
        artifacts_dir: Directory to save failure logs.

    Returns:
        Structured failure analysis summary dictionary.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    actual_intent_col = intent_col if intent_col in df.columns else ("intent" if "intent" in df.columns else "intent_gold")
    actual_action_col = action_col if action_col in df.columns else ("action" if "action" in df.columns else "action_gold")

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for (_, row), pred in zip(df.iterrows(), predictions):
        gold_intent = str(row.get(actual_intent_col, ""))
        gold_action = str(row.get(actual_action_col, ""))
        pred_intent = str(pred.get("intent", ""))
        pred_action = str(pred.get("action", ""))
        is_ambiguous = str(row.get("ambiguity", "")).lower() == "true"
        context = str(row.get("conversation_context", ""))
        cust_msg = str(row.get("customer_message", ""))

        categories: List[str] = []

        # Intent classification mismatch
        if pred_intent != gold_intent:
            if is_ambiguous:
                categories.append("ambiguous_or_multi_intent")
            elif context and len(context.strip()) > 0:
                categories.append("context_dependent_intent")
            else:
                categories.append("intent_confusion")

        # Retrieval grounding shortfall
        if not pred.get("retrieved_examples"):
            categories.append("unsupported_retrieval")

        # Action / Policy mismatch
        if pred_action != gold_action:
            if pred_action == "AUTO_HANDLE" and gold_action == "ESCALATE":
                categories.append("unsafe_auto_handle")
            elif pred_action == "ESCALATE" and gold_action == "AUTO_HANDLE":
                categories.append("over_escalation")

        for cat in categories:
            hypothesis, possible_fix = FAILURE_DIAGNOSTICS.get(
                cat,
                ("Unclassified error mode.", "Perform manual error analysis."),
            )
            buckets[cat].append({
                "example_id": str(row.get("example_id", "")),
                "customer_message": cust_msg,
                "conversation_context": context,
                "gold_intent": gold_intent,
                "predicted_intent": pred_intent,
                "gold_action": gold_action,
                "predicted_action": pred_action,
                "agent_reply": pred.get("reply", ""),
                "escalation_reason": pred.get("escalation_reason", ""),
                "hypothesis": hypothesis,
                "possible_fix": possible_fix,
            })

    ranked_categories = sorted(buckets.keys(), key=lambda c: (-len(buckets[c]), c))

    top_five_summary = [
        {
            "category": cat,
            "error_count": len(buckets[cat]),
            "hypothesis": FAILURE_DIAGNOSTICS[cat][0],
            "recommended_fix": FAILURE_DIAGNOSTICS[cat][1],
            "representative_examples": buckets[cat][:3],
        }
        for cat in ranked_categories[:5]
    ]

    failure_analysis_report = {
        "total_evaluated_samples": len(df),
        "total_failures_observed": sum(len(b) for b in buckets.values()),
        "category_counts": {cat: len(buckets[cat]) for cat in ranked_categories},
        "top_failure_modes": top_five_summary,
        "all_failure_details": {cat: items for cat, items in buckets.items()},
    }

    # Save to artifacts
    cases_path = artifacts_dir / "failure_cases.json"
    examples_path = artifacts_dir / "failure_examples.json"

    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump(failure_analysis_report, f, indent=2)

    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(failure_analysis_report, f, indent=2)

    logging.info(f"Saved failure analysis artifacts to {cases_path} and {examples_path}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"  Top Failure Modes & Error Diagnostics (N = {len(df):,})")
    print(f"{'='*80}")
    for idx, top in enumerate(top_five_summary, 1):
        print(f"  {idx}. {top['category']} (Count = {top['error_count']})")
        print(f"     Hypothesis: {top['hypothesis']}")
        print(f"     Remedy:     {top['recommended_fix']}")
    print(f"{'='*80}\n")

    return failure_analysis_report


def analyze_failures(
    dataset_path: Union[str, Path] = "data/golden/golden_eval_verified.csv",
    system_tier: str = "main",
    artifacts_dir: Union[str, Path] = "artifacts",
) -> Dict[str, Any]:
    """Loads dataset, generates predictions, and executes full failure analysis."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path, dtype=str, keep_default_na=False)
    agent = load_agent(system_tier=system_tier)

    logging.info(f"Generating predictions on {len(df)} samples from {dataset_path}...")
    predictions = [
        agent.process(
            message=row.customer_message,
            context=getattr(row, "conversation_context", ""),
        )
        for row in df.itertuples()
    ]

    return extract_failure_cases(
        df=df,
        predictions=predictions,
        artifacts_dir=artifacts_dir,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/golden/golden_eval_verified.csv", help="Evaluation dataset CSV")
    parser.add_argument("--system", choices=["main", "simple", "trivial"], default="main", help="System tier")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    args = parser.parse_args()

    analyze_failures(
        dataset_path=args.dataset,
        system_tier=args.system,
        artifacts_dir=args.artifacts_dir,
    )


if __name__ == "__main__":
    main()
