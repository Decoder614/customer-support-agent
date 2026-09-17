"""Comprehensive Evaluation Metrics for Intent Classification, Calibration, and Safety.

Calculates:
1. Classification Metrics: Multi-Class Accuracy, Macro F1, Weighted F1, Per-class stats, Confusion Matrix.
2. Calibration Metrics: Expected Calibration Error (ECE) over equal-width bins with reliability stats.
3. Safety & Trust Metrics: False Auto-Handle Count, False Auto-Handle Rate, Unsafe Rate Among Auto,
   Automation Coverage, Joint Correct Rate on Auto-Handled Traffic, and Retrieval Grounding Stats.
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.intents.taxonomy import INTENTS


def classification_metrics(
    gold: Sequence[str],
    predicted: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Computes multi-class classification accuracy, macro/weighted F1, and per-class reports."""
    if len(gold) == 0 or len(predicted) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class": {},
            "confusion_matrix": [],
        }

    label_list = list(labels) if labels is not None else sorted(list(set(gold) | set(predicted)))
    report = classification_report(
        gold,
        predicted,
        labels=label_list,
        output_dict=True,
        zero_division=0,
    )
    conf_mat = confusion_matrix(gold, predicted, labels=label_list).tolist()

    return {
        "accuracy": round(float(accuracy_score(gold, predicted)), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "per_class": {
            lbl: {
                "precision": round(float(report[lbl]["precision"]), 4),
                "recall": round(float(report[lbl]["recall"]), 4),
                "f1": round(float(report[lbl]["f1-score"]), 4),
                "support": int(report[lbl]["support"]),
            }
            for lbl in label_list
            if lbl in report
        },
        "labels": label_list,
        "confusion_matrix": conf_mat,
    }


def compute_expected_calibration_error(
    y_true: Sequence[bool],
    confidences: Sequence[float],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Calculates Expected Calibration Error (ECE) across confidence bins.

    Formula: ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        y_true: Sequence of booleans indicating correctness of each prediction (y_pred == y_gold).
        confidences: Sequence of model confidence scores (0.0 to 1.0).
        n_bins: Number of equal-width calibration bins (default: 10).

    Returns:
        Dictionary with ECE score and detailed bin diagnostics.
    """
    if len(y_true) == 0 or len(confidences) == 0:
        return {"ece": 0.0, "bins": []}

    correct = np.asarray(y_true, dtype=bool)
    confs = np.asarray(confidences, dtype=float)
    total_samples = len(correct)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confs >= bin_lower) & (confs <= bin_upper)
        else:
            in_bin = (confs >= bin_lower) & (confs < bin_upper)

        bin_count = int(in_bin.sum())
        if bin_count > 0:
            bin_acc = float(correct[in_bin].mean())
            bin_conf = float(confs[in_bin].mean())
            abs_gap = abs(bin_acc - bin_conf)
            weight = bin_count / total_samples
            ece += weight * abs_gap

            bin_details.append({
                "bin_idx": i,
                "lower": round(float(bin_lower), 2),
                "upper": round(float(bin_upper), 2),
                "n": bin_count,
                "accuracy": round(bin_acc, 4),
                "mean_confidence": round(bin_conf, 4),
                "calibration_gap": round(abs_gap, 4),
            })

    return {
        "ece": round(float(ece), 4),
        "n_bins": n_bins,
        "bins": bin_details,
    }


def calculate_safety_and_policy_metrics(
    gold_actions: Sequence[str],
    pred_actions: Sequence[str],
    gold_intents: Optional[Sequence[str]] = None,
    pred_intents: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Calculates safety trust metrics and false automation risk rates."""
    total = len(gold_actions)
    if total == 0:
        return {
            "total_samples": 0,
            "false_auto_handle_count": 0,
            "false_auto_handle_rate": 0.0,
            "unsafe_among_auto": 0.0,
            "auto_handle_rate": 0.0,
            "escalation_rate": 0.0,
        }

    gold_act_arr = np.asarray(gold_actions)
    pred_act_arr = np.asarray(pred_actions)

    auto_mask = (pred_act_arr == "AUTO_HANDLE")
    should_escalate_mask = (gold_act_arr == "ESCALATE")

    # Critical failure: Agent auto-handles when gold truth requires human escalation
    false_auto_mask = auto_mask & should_escalate_mask
    false_auto_count = int(false_auto_mask.sum())

    total_should_escalate = int(should_escalate_mask.sum())
    total_auto_handled = int(auto_mask.sum())

    # False Auto-Handle Rate: false_autos / total_cases_requiring_escalation
    false_auto_handle_rate = (
        round(false_auto_count / total_should_escalate, 4) if total_should_escalate > 0 else 0.0
    )

    # Unsafe Rate Among Auto: false_autos / total_auto_handled_traffic
    unsafe_among_auto = (
        round(false_auto_count / total_auto_handled, 4) if total_auto_handled > 0 else 0.0
    )

    auto_handle_rate = round(total_auto_handled / total, 4)
    escalation_rate = round(float((~auto_mask).mean()), 4)

    # Intent accuracy within auto-handled cohort
    intent_acc_when_auto = None
    joint_correct_when_auto = None
    if gold_intents is not None and pred_intents is not None and total_auto_handled > 0:
        correct_intent_arr = (np.asarray(gold_intents) == np.asarray(pred_intents))
        intent_acc_when_auto = round(float(correct_intent_arr[auto_mask].mean()), 4)
        # Jointly correct: correct intent AND correctly auto-handled (not supposed to escalate)
        joint_correct_mask = correct_intent_arr & (~should_escalate_mask)
        joint_correct_when_auto = round(float(joint_correct_mask[auto_mask].mean()), 4)

    return {
        "total_samples": total,
        "false_auto_handle_count": false_auto_count,
        "false_auto_handle_rate": false_auto_handle_rate,
        "unsafe_among_auto": unsafe_among_auto,
        "unsafe_rate_among_auto": unsafe_among_auto,
        "auto_handle_rate": auto_handle_rate,
        "escalation_rate": escalation_rate,
        "intent_accuracy_when_auto": intent_acc_when_auto,
        "joint_correct_when_auto": joint_correct_when_auto,
    }


def evaluate_agent_predictions(
    gold_df: pd.DataFrame,
    predictions: List[Dict[str, Any]],
    intent_col: str = "intent_gold",
    action_col: str = "action_gold",
) -> Dict[str, Any]:
    """Generates complete holistic benchmark evaluation report across classification, calibration, and safety."""
    if len(gold_df) != len(predictions):
        raise ValueError(f"Mismatch: gold DataFrame has {len(gold_df)} rows, predictions has {len(predictions)} entries.")

    gold_intents = gold_df[intent_col].tolist()
    gold_actions = gold_df[action_col].tolist()

    pred_intents = [p.get("intent", "") for p in predictions]
    pred_actions = [p.get("action", "") for p in predictions]
    confidences = [float(p.get("intent_confidence", 0.0)) for p in predictions]

    # 1. Intent Classification Performance
    intent_metrics = classification_metrics(gold_intents, pred_intents, labels=INTENTS)

    # 2. Escalation Action Performance
    escalation_metrics = classification_metrics(gold_actions, pred_actions, labels=["AUTO_HANDLE", "ESCALATE"])

    # 3. Safety & Policy Trust Metrics
    safety_metrics = calculate_safety_and_policy_metrics(
        gold_actions=gold_actions,
        pred_actions=pred_actions,
        gold_intents=gold_intents,
        pred_intents=pred_intents,
    )

    # 4. Expected Calibration Error (ECE)
    correct_intent_bools = [g == p for g, p in zip(gold_intents, pred_intents)]
    calibration_metrics = compute_expected_calibration_error(correct_intent_bools, confidences)

    # 5. Retrieval Grounding Coverage
    retrieved_counts = [len(p.get("retrieved_examples", [])) for p in predictions]
    retrieval_coverage = round(float(np.mean([c > 0 for c in retrieved_counts])), 4)
    sim_scores = [
        float(p["retrieved_examples"][0].get("similarity_score", p["retrieved_examples"][0].get("similarity", 0.0)))
        for p in predictions
        if p.get("retrieved_examples")
    ]
    mean_top_similarity = round(float(np.mean(sim_scores)), 4) if sim_scores else 0.0

    return {
        "n_samples": len(gold_df),
        "intent_classification": intent_metrics,
        "escalation_classification": escalation_metrics,
        "safety_metrics": safety_metrics,
        "calibration": calibration_metrics,
        "retrieval": {
            "retrieval_coverage": retrieval_coverage,
            "mean_top_similarity": mean_top_similarity,
        },
    }
