"""One-Shot Benchmark Reproduction Script for Uber Support AI Agent.

Executes the end-to-end evaluation benchmark:
1. Trains all 3 system tiers (Trivial, Simple TF-IDF, Main Hybrid) strictly on train.csv.
2. Evaluates all 3 systems against the 200-row golden benchmark (golden_eval_verified.csv).
3. Computes classification metrics (Accuracy, Macro F1, Weighted F1), calibration (ECE),
   and safety trust metrics (False Auto-Handle Rate, Unsafe in Auto, Joint Correct Rate).
4. Runs confidence threshold sweep (0.0 to 0.95) and failure mode diagnostic extraction.
5. Runs LLM-as-judge rubric evaluation and computes Quadratic-Weighted Cohen's Kappa.
6. Executes the entire test suite and reports test results.
7. Logs per-stage runtime and consolidated summary artifacts to artifacts/reproduction_summary.json.
"""

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

# Ensure parent directory is in sys.path when running as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.support_agent import SupportAgent, load_agent
from src.intents.taxonomy import INTENTS
from src.intents.train import train_models
from src.evaluation.metrics import (
    classification_metrics,
    calculate_safety_and_policy_metrics,
    compute_expected_calibration_error,
    evaluate_agent_predictions,
)
from src.evaluation.threshold_sweep import run_threshold_sweep
from src.evaluation.failure_analysis import extract_failure_cases, analyze_failures
from src.evaluation.judge_agreement import compute_judge_agreement


def format_pct(val: Optional[float], precision: int = 1) -> str:
    """Helper to format float as percentage or N/A."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val * 100:.{precision}f}%"


def format_float(val: Optional[float], precision: int = 3) -> str:
    """Helper to format float with fixed precision."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{precision}f}"


def run_benchmark_reproduction(
    config_path: Path = Path("config/default.yaml"),
    train_path: Path = Path("data/processed/train.csv"),
    dev_path: Path = Path("data/processed/dev.csv"),
    eval_bootstrap_path: Path = Path("data/processed/eval_bootstrap.csv"),
    golden_path: Path = Path("data/golden/golden_eval_verified.csv"),
    human_eval_path: Path = Path("data/eval/judge_human_eval_completed.csv"),
    models_dir: Path = Path("artifacts/models"),
    artifacts_dir: Path = Path("artifacts"),
    run_pytest: bool = True,
) -> Dict[str, Any]:
    """Runs complete end-to-end benchmark reproduction pipeline.

    Returns:
        Consolidated reproduction metrics and timing report.
    """
    total_start = time.perf_counter()
    timings: Dict[str, float] = {}
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print("  UBER SUPPORT AI AGENT - ONE-SHOT BENCHMARK REPRODUCTION HARNESS")
    print("=" * 90)
    print(f"  Execution Timestamp : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Repository Root     : {REPO_ROOT}")
    print(f"  Golden Benchmark    : {golden_path} ({'Found' if golden_path.exists() else 'Missing!'})")
    print("=" * 90 + "\n")

    # -------------------------------------------------------------------------
    # STAGE 1: Train all 3 Model Tiers
    # -------------------------------------------------------------------------
    print("[STAGE 1/6] Training all 3 Model Tiers strictly on train.csv...")
    stage1_start = time.perf_counter()
    train_metadata = train_models(
        config_path=config_path,
        train_path=train_path,
        dev_path=dev_path,
        eval_bootstrap_path=eval_bootstrap_path,
        golden_path=golden_path,
        models_dir=models_dir,
        artifacts_dir=artifacts_dir,
    )
    timings["model_training_seconds"] = round(time.perf_counter() - stage1_start, 3)
    print(f"   -> Completed in {timings['model_training_seconds']:.2f}s\n")

    # -------------------------------------------------------------------------
    # STAGE 2: End-to-End Evaluation on Golden Benchmark (200 rows)
    # -------------------------------------------------------------------------
    print("[STAGE 2/6] Evaluating all 3 Systems on 200-row Golden Benchmark...")
    stage2_start = time.perf_counter()

    if not golden_path.exists():
        raise FileNotFoundError(f"Golden benchmark dataset not found at {golden_path}")

    gold_df = pd.read_csv(golden_path, dtype=str, keep_default_na=False)
    system_tiers = ["trivial", "simple", "main"]
    system_benchmark_results: Dict[str, Any] = {}
    system_predictions: Dict[str, List[Dict[str, Any]]] = {}

    for tier in system_tiers:
        logging.info(f"Running end-to-end SupportAgent on golden set (system_tier={tier})...")
        agent = load_agent(
            system_tier=tier,
            models_dir=models_dir,
            train_path=train_path,
            config_path=config_path,
        )

        preds = [
            agent.process(
                message=str(row.customer_message),
                context=str(getattr(row, "conversation_context", "")),
            )
            for row in gold_df.itertuples()
        ]
        system_predictions[tier] = preds

        eval_report = evaluate_agent_predictions(
            gold_df=gold_df,
            predictions=preds,
            intent_col="intent_gold",
            action_col="action_gold",
        )
        system_benchmark_results[tier] = eval_report

    timings["golden_evaluation_seconds"] = round(time.perf_counter() - stage2_start, 3)
    print(f"   -> Completed in {timings['golden_evaluation_seconds']:.2f}s\n")

    # -------------------------------------------------------------------------
    # STAGE 3: Parameter Sweep over Confidence Thresholds
    # -------------------------------------------------------------------------
    print("[STAGE 3/6] Running Confidence Threshold Parameter Sweep...")
    stage3_start = time.perf_counter()
    sweep_results = run_threshold_sweep(
        dataset_path=dev_path if dev_path.exists() else golden_path,
        system_tier="main",
        artifacts_dir=artifacts_dir,
    )
    timings["threshold_sweep_seconds"] = round(time.perf_counter() - stage3_start, 3)
    print(f"   -> Completed in {timings['threshold_sweep_seconds']:.2f}s\n")

    # -------------------------------------------------------------------------
    # STAGE 4: Diagnostic Failure Analysis & Extraction
    # -------------------------------------------------------------------------
    print("[STAGE 4/6] Extracting Top Failure Modes & Diagnostics on Golden Set...")
    stage4_start = time.perf_counter()
    failure_report = extract_failure_cases(
        df=gold_df,
        predictions=system_predictions["main"],
        intent_col="intent_gold",
        action_col="action_gold",
        artifacts_dir=artifacts_dir,
    )
    timings["failure_analysis_seconds"] = round(time.perf_counter() - stage4_start, 3)
    print(f"   -> Completed in {timings['failure_analysis_seconds']:.2f}s\n")

    # -------------------------------------------------------------------------
    # STAGE 5: LLM-as-Judge Calibration & Inter-Annotator Agreement
    # -------------------------------------------------------------------------
    print("[STAGE 5/6] Measuring Human vs. LLM Judge Quadratic-Weighted Cohen's Kappa...")
    stage5_start = time.perf_counter()
    judge_results = {}
    if human_eval_path.exists():
        human_df = pd.read_csv(human_eval_path, dtype=str, keep_default_na=False)
        judge_results = compute_judge_agreement(
            human_df=human_df,
            artifacts_dir=artifacts_dir,
        )
    else:
        logging.warning(f"Human evaluation file not found at {human_eval_path}; skipping judge agreement.")
    timings["judge_agreement_seconds"] = round(time.perf_counter() - stage5_start, 3)
    print(f"   -> Completed in {timings['judge_agreement_seconds']:.2f}s\n")

    # -------------------------------------------------------------------------
    # STAGE 6: Pytest Test Suite Execution
    # -------------------------------------------------------------------------
    test_suite_status = "SKIPPED"
    if run_pytest:
        print("[STAGE 6/6] Executing Complete Project Test Suite (pytest)...")
        stage6_start = time.perf_counter()
        
        exit_code = pytest.main(["-q", "tests", "--tb=short"])
        test_suite_status = "PASSED" if exit_code == 0 else f"FAILED (code {exit_code})"
        timings["pytest_suite_seconds"] = round(time.perf_counter() - stage6_start, 3)
        print(f"   -> Test Suite Status: {test_suite_status} in {timings['pytest_suite_seconds']:.2f}s\n")
    else:
        timings["pytest_suite_seconds"] = 0.0

    total_duration = round(time.perf_counter() - total_start, 3)
    timings["total_reproduction_seconds"] = total_duration

    # -------------------------------------------------------------------------
    # PERSIST CONSOLIDATED ARTIFACTS
    # -------------------------------------------------------------------------
    consolidated_summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "golden_samples_count": len(gold_df),
        "systems_evaluated": system_tiers,
        "timings_seconds": timings,
        "test_suite_status": test_suite_status,
        "benchmark_results": system_benchmark_results,
        "top_failure_modes": failure_report.get("top_failure_modes", []),
        "judge_agreement": judge_results,
        "threshold_sweep_summary": sweep_results,
    }

    summary_json_path = artifacts_dir / "reproduction_summary.json"
    benchmark_json_path = artifacts_dir / "benchmark_results.json"

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(consolidated_summary, f, indent=2)

    with open(benchmark_json_path, "w", encoding="utf-8") as f:
        json.dump(system_benchmark_results, f, indent=2)

    # -------------------------------------------------------------------------
    # PRINT RICH BENCHMARK SUMMARY TABLES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"  BENCHMARK REPRODUCTION SUMMARY (Golden Benchmark N = {len(gold_df)})")
    print("=" * 90)

    # Table 1: Classification Performance
    print("\n  1. INTENT CLASSIFICATION BENCHMARK")
    print("  " + "-" * 86)
    print(f"  {'System Tier':<20} | {'Accuracy':>12} | {'Macro F1':>12} | {'Weighted F1':>14} | {'ECE':>12}")
    print("  " + "-" * 86)
    for tier in system_tiers:
        res = system_benchmark_results[tier]
        acc = format_pct(res["intent_classification"]["accuracy"])
        m_f1 = format_float(res["intent_classification"]["macro_f1"])
        w_f1 = format_float(res["intent_classification"]["weighted_f1"])
        ece = format_float(res["calibration"]["ece"])
        print(f"  {tier.capitalize() + ' Baseline' if tier != 'main' else 'Main (Hybrid)':<20} | {acc:>12} | {m_f1:>12} | {w_f1:>14} | {ece:>12}")
    print("  " + "-" * 86)

    # Table 2: Safety, Trust & Policy Automation Performance
    print("\n  2. SAFETY, POLICY TRUST & AUTOMATION PERFORMANCE")
    print("  " + "-" * 86)
    print(f"  {'System Tier':<20} | {'Auto-Handle %':>13} | {'False Autos':>12} | {'Unsafe in Auto':>15} | {'Joint Correct':>14}")
    print("  " + "-" * 86)
    for tier in system_tiers:
        s = system_benchmark_results[tier]["safety_metrics"]
        auto_pct = format_pct(s["auto_handle_rate"])
        false_cnt = str(s["false_auto_handle_count"])
        unsafe_pct = format_pct(s["unsafe_among_auto"])
        joint_pct = format_pct(s["joint_correct_when_auto"])
        print(f"  {tier.capitalize() + ' Baseline' if tier != 'main' else 'Main (Hybrid)':<20} | {auto_pct:>13} | {false_cnt:>12} | {unsafe_pct:>15} | {joint_pct:>14}")
    print("  " + "-" * 86)

    # Table 3: Judge Calibration Agreement
    if judge_results and "macro_averages" in judge_results:
        print("\n  3. LLM-AS-JUDGE INTER-ANNOTATOR AGREEMENT (Quadratic-Weighted Kappa)")
        print("  " + "-" * 86)
        print(f"  {'Dimension':<30} | {'Exact (%)':>10} | {'Within +-1':>10} | {'Pearson r':>10} | {'Quad Kappa':>12}")
        print("  " + "-" * 86)
        for dim, stats in judge_results["dimensions"].items():
            ex = format_pct(stats["exact_agreement"])
            w1 = format_pct(stats["within_one_agreement"])
            pr = format_float(stats["pearson_r"])
            kp = format_float(stats["quadratic_weighted_kappa"])
            print(f"  {dim:<30} | {ex:>10} | {w1:>10} | {pr:>10} | {kp:>12}")
        print("  " + "-" * 86)
        macro = judge_results["macro_averages"]
        print(f"  {'Macro Average':<30} | {format_pct(macro['mean_exact_agreement']):>10} | {format_pct(macro['mean_within_one']):>10} | {format_float(macro['mean_pearson_r']):>10} | {format_float(macro['mean_quadratic_kappa']):>12}")
        print("  " + "-" * 86)

    # Table 4: Runtime Performance & Timing Breakdown
    print("\n  4. REPRODUCTION PIPELINE TIMING BREAKDOWN")
    print("  " + "-" * 86)
    print(f"  {'Pipeline Stage':<45} | {'Duration (s)':>15} | {'% of Total':>15}")
    print("  " + "-" * 86)
    for stage_name, dur in timings.items():
        if stage_name != "total_reproduction_seconds":
            pct_str = f"{(dur / total_duration) * 100:.1f}%" if total_duration > 0 else "0.0%"
            clean_name = stage_name.replace("_", " ").title()
            print(f"  {clean_name:<45} | {dur:>14.2f}s | {pct_str:>15}")
    print("  " + "-" * 86)
    print(f"  {'TOTAL END-TO-END EXECUTION TIME':<45} | {total_duration:>14.2f}s | {'100.0%':>15}")
    print("  " + "-" * 86)

    # Verification threshold check
    print("\n" + "=" * 90)
    if total_duration < 60.0 and test_suite_status == "PASSED":
        print(f"  [SUCCESS] BENCHMARK REPRODUCTION SUCCESSFUL! Finished in {total_duration:.2f}s (< 60.0s SLA target)")
    else:
        print(f"  [COMPLETED] BENCHMARK FINISHED in {total_duration:.2f}s (Status: {test_suite_status})")
    print(f"  Summary persisted to: {summary_json_path}")
    print("=" * 90 + "\n")

    return consolidated_summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    parser.add_argument("--train-path", default="data/processed/train.csv", help="Path to train CSV")
    parser.add_argument("--dev-path", default="data/processed/dev.csv", help="Path to dev CSV")
    parser.add_argument("--eval-bootstrap-path", default="data/processed/eval_bootstrap.csv", help="Path to eval bootstrap CSV")
    parser.add_argument("--golden-path", default="data/golden/golden_eval_verified.csv", help="Golden benchmark CSV")
    parser.add_argument("--human-eval-path", default="data/eval/judge_human_eval_completed.csv", help="Human evaluation CSV")
    parser.add_argument("--models-dir", default="artifacts/models", help="Directory for model joblibs")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest test execution")
    args = parser.parse_args()

    run_benchmark_reproduction(
        config_path=Path(args.config),
        train_path=Path(args.train_path),
        dev_path=Path(args.dev_path),
        eval_bootstrap_path=Path(args.eval_bootstrap_path),
        golden_path=Path(args.golden_path),
        human_eval_path=Path(args.human_eval_path),
        models_dir=Path(args.models_dir),
        artifacts_dir=Path(args.artifacts_dir),
        run_pytest=not args.skip_tests,
    )


if __name__ == "__main__":
    main()
