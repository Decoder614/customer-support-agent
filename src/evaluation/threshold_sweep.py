"""Parameter Sweep Engine for Confidence & Grounding Thresholds.

Evaluates trade-offs between automation coverage (Auto-Handle Rate) and safety/trust
(False Auto-Handle Rate, Unsafe Rate Among Auto) across confidence thresholds (0.0 to 0.95).
Outputs structured results to artifacts/threshold_sweep.json and artifacts/threshold_sweep.csv.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from src.agent.support_agent import SupportAgent, load_agent
from src.evaluation.metrics import evaluate_agent_predictions, calculate_safety_and_policy_metrics


DEFAULT_THRESHOLDS: List[float] = [0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95]


def run_threshold_sweep(
    dataset_path: Union[str, Path] = "data/processed/dev.csv",
    system_tier: str = "main",
    thresholds: Optional[Sequence[float]] = None,
    artifacts_dir: Union[str, Path] = "artifacts",
    intent_col: str = "intent",
    action_col: str = "action",
) -> List[Dict[str, Any]]:
    """Runs systematic threshold sweep over the provided evaluation dataset.

    Args:
        dataset_path: Path to development or evaluation CSV.
        system_tier: Model tier ('main', 'simple', 'trivial').
        thresholds: List of confidence thresholds to sweep.
        artifacts_dir: Output directory for sweep artifacts.
        intent_col: Name of gold intent column in CSV.
        action_col: Name of gold action column in CSV.

    Returns:
        List of per-threshold metric dictionaries.
    """
    threshold_list = list(thresholds) if thresholds is not None else DEFAULT_THRESHOLDS
    dataset_path = Path(dataset_path)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset for threshold sweep not found at {dataset_path}")

    logging.info(f"Loading dataset from {dataset_path} for threshold sweep...")
    df = pd.read_csv(dataset_path, dtype=str, keep_default_na=False)

    # Detect gold column names
    actual_intent_col = intent_col if intent_col in df.columns else ("intent_gold" if "intent_gold" in df.columns else "intent")
    actual_action_col = action_col if action_col in df.columns else ("action_gold" if "action_gold" in df.columns else "action")

    gold_intents = df[actual_intent_col].tolist()
    gold_actions = df[actual_action_col].tolist()

    agent = load_agent(system_tier=system_tier)
    sweep_results: List[Dict[str, Any]] = []

    logging.info(f"Sweeping {len(threshold_list)} thresholds over {len(df)} samples...")

    for thresh in threshold_list:
        agent.settings["intent_confidence_threshold"] = thresh

        predictions = [
            agent.process(
                message=row.customer_message,
                context=getattr(row, "conversation_context", ""),
            )
            for row in df.itertuples()
        ]

        pred_intents = [p["intent"] for p in predictions]
        pred_actions = [p["action"] for p in predictions]

        safety_metrics = calculate_safety_and_policy_metrics(
            gold_actions=gold_actions,
            pred_actions=pred_actions,
            gold_intents=gold_intents,
            pred_intents=pred_intents,
        )

        row_result = {
            "confidence_threshold": round(float(thresh), 2),
            "system": system_tier,
            "dataset": dataset_path.name,
            "total_samples": len(df),
            "auto_handle_rate": safety_metrics["auto_handle_rate"],
            "escalation_rate": safety_metrics["escalation_rate"],
            "false_auto_handle_count": safety_metrics["false_auto_handle_count"],
            "false_auto_handle_rate": safety_metrics["false_auto_handle_rate"],
            "unsafe_among_auto": safety_metrics["unsafe_among_auto"],
            "intent_accuracy_when_auto": safety_metrics["intent_accuracy_when_auto"],
            "joint_correct_when_auto": safety_metrics["joint_correct_when_auto"],
        }
        sweep_results.append(row_result)

    # Save artifacts
    json_path = artifacts_dir / "threshold_sweep.json"
    csv_path = artifacts_dir / "threshold_sweep.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    pd.DataFrame(sweep_results).to_csv(csv_path, index=False)
    logging.info(f"Saved threshold sweep artifacts to {json_path} and {csv_path}")

    # Print summary table
    print(f"\n{'='*95}")
    print(f"  Confidence Threshold Sweep Summary ({dataset_path.name}, N = {len(df):,})")
    print(f"{'='*95}")
    print(f"  {'Threshold':>10} | {'Auto-Handle %':>14} | {'Escalate %':>12} | {'False Autos':>12} | {'Unsafe in Auto':>15} | {'Joint Correct':>14}")
    print(f"  {'-'*10}-+-{'-'*14}-+-{'-'*12}-+-{'-'*12}-+-{'-'*15}-+-{'-'*14}")
    for r in sweep_results:
        auto_pct = f"{r['auto_handle_rate']*100:.1f}%"
        esc_pct = f"{r['escalation_rate']*100:.1f}%"
        unsafe_pct = f"{r['unsafe_among_auto']*100:.1f}%" if r["unsafe_among_auto"] is not None else "N/A"
        joint_pct = f"{r['joint_correct_when_auto']*100:.1f}%" if r["joint_correct_when_auto"] is not None else "N/A"
        print(f"  {r['confidence_threshold']:>10.2f} | {auto_pct:>14} | {esc_pct:>12} | {r['false_auto_handle_count']:>12} | {unsafe_pct:>15} | {joint_pct:>14}")
    print(f"{'='*95}\n")

    return sweep_results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/dev.csv", help="Dataset path for threshold sweep")
    parser.add_argument("--system", choices=["main", "simple", "trivial"], default="main", help="System tier")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    args = parser.parse_args()

    run_threshold_sweep(
        dataset_path=args.dataset,
        system_tier=args.system,
        artifacts_dir=args.artifacts_dir,
    )


if __name__ == "__main__":
    main()
