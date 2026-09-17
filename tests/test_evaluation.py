"""Tests for evaluation metrics, calibration (ECE), safety trust metrics, threshold sweeps, LLM judge, and agreement."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    classification_metrics,
    compute_expected_calibration_error,
    calculate_safety_and_policy_metrics,
    evaluate_agent_predictions,
)
from src.evaluation.threshold_sweep import run_threshold_sweep
from src.evaluation.failure_analysis import extract_failure_cases, analyze_failures
from src.evaluation.llm_judge import (
    JUDGE_DIMENSIONS,
    deterministic_rubric_judge,
    evaluate_with_judge,
    offline_quality_checks,
)
from src.evaluation.judge_agreement import compute_judge_agreement
from src.intents.taxonomy import Intent, INTENTS


def test_classification_metrics():
    """Verify multi-class classification accuracy, F1, and confusion matrix calculation."""
    gold = ["lost_and_found", "safety_and_conduct", "cancellation_fee", "lost_and_found"]
    pred = ["lost_and_found", "safety_and_conduct", "fare_dispute_overcharge", "lost_and_found"]

    res = classification_metrics(gold, pred)
    assert res["accuracy"] == 0.75
    assert 0.0 <= res["macro_f1"] <= 1.0
    assert 0.0 <= res["weighted_f1"] <= 1.0
    assert len(res["confusion_matrix"]) > 0
    assert res["per_class"]["lost_and_found"]["recall"] == 1.0


def test_classification_metrics_empty():
    """Verify handling of empty inputs."""
    res = classification_metrics([], [])
    assert res["accuracy"] == 0.0
    assert res["macro_f1"] == 0.0
    assert res["per_class"] == {}


def test_compute_expected_calibration_error_perfect():
    """Verify ECE is 0.0 when predictions are perfectly calibrated."""
    y_true = [True] * 10
    confs = [1.0] * 10

    ece_res = compute_expected_calibration_error(y_true, confs, n_bins=10)
    assert ece_res["ece"] == 0.0
    assert len(ece_res["bins"]) == 1
    assert ece_res["bins"][0]["accuracy"] == 1.0


def test_compute_expected_calibration_error_miscalibrated():
    """Verify ECE accurately measures gap when predictions are overconfident."""
    y_true = [False] * 10
    confs = [1.0] * 10

    ece_res = compute_expected_calibration_error(y_true, confs, n_bins=10)
    assert ece_res["ece"] == 1.0
    assert ece_res["bins"][0]["accuracy"] == 0.0
    assert ece_res["bins"][0]["mean_confidence"] == 1.0


def test_calculate_safety_and_policy_metrics():
    """Verify false auto-handle rate and unsafe rate among auto calculations."""
    gold_actions = ["ESCALATE", "ESCALATE", "AUTO_HANDLE", "AUTO_HANDLE"]
    pred_actions = ["ESCALATE", "AUTO_HANDLE", "AUTO_HANDLE", "ESCALATE"]
    gold_intents = ["safety_and_conduct", "fare_dispute_overcharge", "lost_and_found", "app_and_account_access"]
    pred_intents = ["safety_and_conduct", "fare_dispute_overcharge", "lost_and_found", "app_and_account_access"]

    safety = calculate_safety_and_policy_metrics(
        gold_actions=gold_actions,
        pred_actions=pred_actions,
        gold_intents=gold_intents,
        pred_intents=pred_intents,
    )

    assert safety["total_samples"] == 4
    assert safety["false_auto_handle_count"] == 1
    assert safety["false_auto_handle_rate"] == 0.50
    assert safety["unsafe_among_auto"] == 0.50
    assert safety["auto_handle_rate"] == 0.50
    assert safety["escalation_rate"] == 0.50
    assert safety["intent_accuracy_when_auto"] == 1.0
    assert safety["joint_correct_when_auto"] == 0.50


def test_calculate_safety_metrics_zero_autos():
    """Verify division by zero handling when no predictions are auto-handled."""
    gold_actions = ["ESCALATE", "AUTO_HANDLE"]
    pred_actions = ["ESCALATE", "ESCALATE"]

    safety = calculate_safety_and_policy_metrics(gold_actions, pred_actions)
    assert safety["false_auto_handle_count"] == 0
    assert safety["false_auto_handle_rate"] == 0.0
    assert safety["unsafe_among_auto"] == 0.0
    assert safety["auto_handle_rate"] == 0.0
    assert safety["escalation_rate"] == 1.0


def test_evaluate_agent_predictions_e2e():
    """Verify end-to-end evaluation report assembly on sample dataset."""
    gold_df = pd.DataFrame({
        "example_id": ["1", "2"],
        "intent_gold": ["lost_and_found", "safety_and_conduct"],
        "action_gold": ["AUTO_HANDLE", "ESCALATE"],
    })

    predictions = [
        {
            "intent": "lost_and_found",
            "intent_confidence": 0.95,
            "action": "AUTO_HANDLE",
            "retrieved_examples": [{"similarity_score": 0.85}],
        },
        {
            "intent": "safety_and_conduct",
            "intent_confidence": 0.90,
            "action": "ESCALATE",
            "retrieved_examples": [],
        },
    ]

    report = evaluate_agent_predictions(gold_df, predictions)
    assert report["n_samples"] == 2
    assert report["intent_classification"]["accuracy"] == 1.0
    assert report["escalation_classification"]["accuracy"] == 1.0
    assert report["safety_metrics"]["false_auto_handle_count"] == 0
    assert report["calibration"]["ece"] >= 0.0
    assert report["retrieval"]["retrieval_coverage"] == 0.5
    assert report["retrieval"]["mean_top_similarity"] == 0.85


def test_extract_failure_cases(tmp_path):
    """Verify failure extraction groups errors into diagnostic buckets."""
    df = pd.DataFrame([
        {
            "example_id": "1",
            "customer_message": "Driver was threatening me and speeding",
            "conversation_context": "",
            "intent_gold": "safety_and_conduct",
            "action_gold": "ESCALATE",
            "ambiguity": "true",
        },
        {
            "example_id": "2",
            "customer_message": "I forgot my phone",
            "conversation_context": "",
            "intent_gold": "lost_and_found",
            "action_gold": "AUTO_HANDLE",
            "ambiguity": "false",
        },
    ])

    predictions = [
        {
            "intent": "other_general_feedback",
            "action": "ESCALATE",
            "reply": "Escalation notice",
            "escalation_reason": "General feedback",
            "retrieved_examples": [],
        },
        {
            "intent": "lost_and_found",
            "action": "AUTO_HANDLE",
            "reply": "Lost item steps",
            "escalation_reason": "",
            "retrieved_examples": [{"similarity_score": 0.8}],
        },
    ]

    result = extract_failure_cases(df, predictions, artifacts_dir=tmp_path)
    assert result["total_evaluated_samples"] == 2
    assert "ambiguous_or_multi_intent" in result["category_counts"]
    assert "unsupported_retrieval" in result["category_counts"]
    assert (tmp_path / "failure_cases.json").exists()
    assert (tmp_path / "failure_examples.json").exists()


def test_run_threshold_sweep(tmp_path):
    """Verify threshold sweep runs across specified thresholds and writes artifacts."""
    sample_df = pd.DataFrame([
        {
            "example_id": "1",
            "customer_message": "I left my wallet in the car",
            "conversation_context": "",
            "intent": "lost_and_found",
            "action": "AUTO_HANDLE",
        },
        {
            "example_id": "2",
            "customer_message": "Driver was drunk and crashed",
            "conversation_context": "",
            "intent": "safety_and_conduct",
            "action": "ESCALATE",
        },
    ])
    data_file = tmp_path / "dev_sample.csv"
    sample_df.to_csv(data_file, index=False)

    sweep_res = run_threshold_sweep(
        dataset_path=data_file,
        system_tier="main",
        thresholds=[0.0, 0.50, 0.95],
        artifacts_dir=tmp_path,
    )

    assert len(sweep_res) == 3
    assert (tmp_path / "threshold_sweep.json").exists()
    assert (tmp_path / "threshold_sweep.csv").exists()


def test_llm_judge_scoring():
    """Verify 6-dimension judge rubric scoring produces integers 1-5 and overall average."""
    prediction = {
        "intent": "lost_and_found",
        "intent_confidence": 0.95,
        "action": "AUTO_HANDLE",
        "escalation_reason": "Low-risk self service",
        "reply": "Hi! Thanks for reaching out. If you left an item behind in a vehicle, you can contact your driver directly through 'Your Trips' > 'Find lost item' in the Uber app.",
        "retrieved_examples": [{"similarity_score": 0.85}],
    }

    judge_res = evaluate_with_judge(
        customer_message="I forgot my wallet in the car yesterday",
        conversation_context="",
        prediction=prediction,
    )

    for dim in JUDGE_DIMENSIONS:
        assert dim in judge_res
        assert isinstance(judge_res[dim], int)
        assert 1 <= judge_res[dim] <= 5

    assert 1.0 <= judge_res["overall_score"] <= 5.0
    assert len(judge_res["reasoning_summary"]) > 0


def test_llm_judge_unsafe_reply_penalty():
    """Verify judge penalizes unsafe draft responses with safety score of 1."""
    unsafe_pred = {
        "intent": "fare_dispute_overcharge",
        "intent_confidence": 0.90,
        "action": "AUTO_HANDLE",
        "escalation_reason": "",
        "reply": "Your refund has been approved and we have refunded $50 to your credit card.",
        "retrieved_examples": [],
    }

    judge_res = evaluate_with_judge(
        customer_message="I was overcharged $50",
        conversation_context="",
        prediction=unsafe_pred,
    )

    assert judge_res["safety"] == 1


def test_offline_quality_checks():
    """Verify offline heuristic checks on prediction payload."""
    clean_pred = {
        "reply": "Hi! We can help with your lost item.",
        "escalation_reason": "Self service",
        "retrieved_examples": [{"similarity_score": 0.8}],
    }
    checks = offline_quality_checks(clean_pred)
    assert checks["reply_nonempty"] is True
    assert checks["risky_pattern_detected"] is False
    assert checks["evidence_present"] is True
    assert checks["reason_present"] is True


def test_compute_judge_agreement(tmp_path):
    """Verify agreement calculation computes exact agreement, within-one, and quadratic kappa."""
    human_df = pd.DataFrame([
        {
            "example_id": "1",
            "system": "main",
            "reply_sha256": "hash1",
            "customer_message": "I lost my phone",
            "conversation_context": "",
            "reply": "Hi! Check Your Trips > Find lost item",
            "action": "AUTO_HANDLE",
            "evidence": "[]",
            "relevance": 5,
            "groundedness": 5,
            "helpfulness": 5,
            "safety": 5,
            "style": 5,
            "escalation_appropriateness": 5,
        },
        {
            "example_id": "2",
            "system": "main",
            "reply_sha256": "hash2",
            "customer_message": "Driver crashed",
            "conversation_context": "",
            "reply": "Specialist will review",
            "action": "ESCALATE",
            "evidence": "[]",
            "relevance": 4,
            "groundedness": 4,
            "helpfulness": 4,
            "safety": 5,
            "style": 4,
            "escalation_appropriateness": 5,
        },
    ])

    judge_df = pd.DataFrame([
        {
            "example_id": "1",
            "system": "main",
            "reply_sha256": "hash1",
            "relevance": 5,
            "groundedness": 5,
            "helpfulness": 5,
            "safety": 5,
            "style": 5,
            "escalation_appropriateness": 5,
        },
        {
            "example_id": "2",
            "system": "main",
            "reply_sha256": "hash2",
            "relevance": 4,
            "groundedness": 5,
            "helpfulness": 4,
            "safety": 5,
            "style": 4,
            "escalation_appropriateness": 5,
        },
    ])

    res = compute_judge_agreement(
        human_df=human_df,
        judge_df=judge_df,
        dimensions=["relevance", "groundedness", "helpfulness", "safety", "style", "escalation_appropriateness"],
        artifacts_dir=tmp_path,
    )

    assert res["n_samples"] == 2
    assert "dimensions" in res
    assert "macro_averages" in res
    assert (tmp_path / "judge_agreement.json").exists()
