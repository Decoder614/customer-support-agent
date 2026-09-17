"""Tests for evaluation metrics, calibration (ECE), safety trust metrics, threshold sweeps, and failure extraction."""

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
            "intent": "other_general_feedback",  # Intent mismatch with ambiguity -> ambiguous_or_multi_intent
            "action": "ESCALATE",
            "reply": "Escalation notice",
            "escalation_reason": "General feedback",
            "retrieved_examples": [],  # Unsupported retrieval
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
