"""Tests for evaluation metrics, calibration (ECE), and safety trust metrics."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    classification_metrics,
    compute_expected_calibration_error,
    calculate_safety_and_policy_metrics,
    evaluate_agent_predictions,
)
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
    # 10 correct predictions at confidence 1.0
    y_true = [True] * 10
    confs = [1.0] * 10

    ece_res = compute_expected_calibration_error(y_true, confs, n_bins=10)
    assert ece_res["ece"] == 0.0
    assert len(ece_res["bins"]) == 1
    assert ece_res["bins"][0]["accuracy"] == 1.0


def test_compute_expected_calibration_error_miscalibrated():
    """Verify ECE accurately measures gap when predictions are overconfident."""
    # 10 completely wrong predictions at confidence 1.0 -> gap = 1.0
    y_true = [False] * 10
    confs = [1.0] * 10

    ece_res = compute_expected_calibration_error(y_true, confs, n_bins=10)
    assert ece_res["ece"] == 1.0
    assert ece_res["bins"][0]["accuracy"] == 0.0
    assert ece_res["bins"][0]["mean_confidence"] == 1.0


def test_calculate_safety_and_policy_metrics():
    """Verify false auto-handle rate and unsafe rate among auto calculations."""
    gold_actions = ["ESCALATE", "ESCALATE", "AUTO_HANDLE", "AUTO_HANDLE"]
    # 1 false auto-handle (idx 1: gold ESCALATE, pred AUTO_HANDLE)
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
    # 1 false auto out of 2 that should escalate -> 50%
    assert safety["false_auto_handle_rate"] == 0.50
    # 1 false auto out of 2 total auto-handled -> 50%
    assert safety["unsafe_among_auto"] == 0.50
    assert safety["auto_handle_rate"] == 0.50
    assert safety["escalation_rate"] == 0.50
    assert safety["intent_accuracy_when_auto"] == 1.0
    # Joint correct when auto: 1 correct auto out of 2 total autos = 0.50
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
