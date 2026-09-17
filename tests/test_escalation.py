"""Tests for deterministic safety gatekeeper and multi-tier escalation policy engine."""

import pytest

from src.escalation.safety_rules import (
    SafetyDecision,
    SAFETY_PATTERNS,
    check_safety_risk,
    evaluate_safety_gate,
)
from src.escalation.policy_engine import (
    PolicyDecision,
    decide,
)
from src.intents.taxonomy import Intent


# ====================================================================
# Tier 1: Safety Gatekeeper Tests
# ====================================================================

@pytest.mark.parametrize(
    "utterance,expected_category",
    [
        ("My driver got into a major accident and crashed into another car", "Physical Safety & Collision"),
        ("We need an ambulance immediately, passenger is bleeding and injured", "Physical Safety & Collision"),
        ("The driver pulled out a gun and threatened everyone", "Police & Legal Emergency"),
        ("I called the police on my driver because of his aggressive behavior", "Police & Legal Emergency"),
        ("I am going to sue Uber and hire a lawyer for this incident", "Police & Legal Emergency"),
        ("Driver assaulted me and was sexually harassing me during the ride", "Threat & Harassment"),
        ("Driver was screaming and yelling death threats at me", "Threat & Harassment"),
        ("Driver was visibly drunk, intoxicated, and smelled of alcohol", "Intoxication & Impairment"),
        ("Driver was driving on the wrong side of the road recklessly", "Reckless Driving"),
        ("Driver ran a red light while speeding dangerously", "Reckless Driving"),
        ("Your company is a piece of shit and you are all fucking useless", "Hostility & Abuse"),
    ],
)
def test_safety_gate_triggers_on_high_risk_utterances(utterance, expected_category):
    """Verify that all safety-critical test utterances force immediate ESCALATE with explicit category."""
    decision = evaluate_safety_gate(utterance)
    assert decision.is_escalated is True
    assert decision.action == "ESCALATE"
    assert decision.risk_category == expected_category
    assert decision.reason is not None and len(decision.reason) > 0
    assert decision.matched_keyword is not None


@pytest.mark.parametrize(
    "benign_utterance",
    [
        "I left my sunglasses in the back seat yesterday",
        "How do I update my email address in the app?",
        "My promo code for 20% off was not applied",
        "Driver was friendly and got me to the airport on time",
        "What is the estimated time of arrival for my pickup?",
        "Where can I view my recent trip receipts?",
    ],
)
def test_safety_gate_passes_benign_utterances(benign_utterance):
    """Verify that low-risk customer inquiries pass through safety gate without escalation."""
    decision = evaluate_safety_gate(benign_utterance)
    assert decision.is_escalated is False
    assert decision.action == "PASS"
    assert decision.risk_category is None
    assert decision.reason is None


def test_safety_gate_checks_conversation_context():
    """Verify that safety keywords present in antecedent conversation context trigger escalation."""
    message = "Any update on this issue?"
    context = "customer: My driver crashed into a light pole and we were injured! | support: We are looking into this."

    decision = evaluate_safety_gate(message=message, context=context)
    assert decision.is_escalated is True
    assert decision.action == "ESCALATE"
    assert decision.risk_category == "Physical Safety & Collision"


def test_safety_gate_edge_cases():
    """Verify handling of empty or None inputs."""
    decision_empty = evaluate_safety_gate("")
    assert decision_empty.is_escalated is False
    assert decision_empty.action == "PASS"

    decision_spaces = evaluate_safety_gate("   ", "   ")
    assert decision_spaces.is_escalated is False
    assert decision_spaces.action == "PASS"

    assert check_safety_risk("") is None
    assert check_safety_risk(None) is None  # type: ignore


# ====================================================================
# Multi-Tier Policy Engine Decision Matrix Tests
# ====================================================================

def test_policy_decide_safety_emergency_overrides():
    """Verify safety emergency overrides high confidence and valid retrieval."""
    valid_evidence = [{"similarity_score": 0.85, "brand_reply": "Here is how to contact driver"}]
    decision = decide(
        message="Driver crashed the car and was threatening me",
        intent=Intent.LOST_AND_FOUND.value,
        confidence=0.95,
        evidence=valid_evidence,
    )
    assert decision.action == "ESCALATE"
    assert decision.escalation_tier == "SAFETY_EMERGENCY"
    assert decision.is_auto_handled is False


@pytest.mark.parametrize(
    "high_risk_intent",
    [
        Intent.SAFETY_AND_CONDUCT.value,
        Intent.FARE_DISPUTE_OVERCHARGE.value,
        Intent.CANCELLATION_FEE.value,
        Intent.DRIVER_PARTNER_INQUIRY.value,
    ],
)
def test_policy_decide_high_risk_intent_escalation(high_risk_intent):
    """Verify high-risk domain intents always escalate regardless of high confidence and evidence."""
    valid_evidence = [{"similarity_score": 0.90, "brand_reply": "Historical reply precedent"}]
    decision = decide(
        message="I have a dispute regarding my fare",
        intent=high_risk_intent,
        confidence=0.98,
        evidence=valid_evidence,
    )
    assert decision.action == "ESCALATE"
    assert decision.escalation_tier == "HIGH_RISK_INTENT"
    assert decision.is_auto_handled is False


def test_policy_decide_general_feedback_escalation():
    """Verify general feedback or unclassified queries are escalated."""
    valid_evidence = [{"similarity_score": 0.70, "brand_reply": "Thanks for reaching out."}]
    decision = decide(
        message="Just saying hi to Uber support",
        intent=Intent.OTHER_GENERAL_FEEDBACK.value,
        confidence=0.90,
        evidence=valid_evidence,
    )
    assert decision.action == "ESCALATE"
    assert decision.escalation_tier == "GENERAL_FEEDBACK"
    assert decision.is_auto_handled is False


def test_policy_decide_confidence_gating():
    """Verify low classification confidence (< 0.65) triggers escalation."""
    valid_evidence = [{"similarity_score": 0.75, "brand_reply": "Check in-app lost item flow"}]
    decision = decide(
        message="I might have left something in the car maybe",
        intent=Intent.LOST_AND_FOUND.value,
        confidence=0.55,  # Below 0.65 threshold
        evidence=valid_evidence,
    )
    assert decision.action == "ESCALATE"
    assert decision.escalation_tier == "CONFIDENCE_GATE"
    assert "below safe automation threshold" in decision.reason


def test_policy_decide_retrieval_evidence_gating():
    """Verify missing or low similarity (< 0.30) evidence triggers escalation."""
    # Empty evidence list
    decision_no_ev = decide(
        message="I left my phone in the car",
        intent=Intent.LOST_AND_FOUND.value,
        confidence=0.90,
        evidence=[],
    )
    assert decision_no_ev.action == "ESCALATE"
    assert decision_no_ev.escalation_tier == "RETRIEVAL_GATE"

    # Low similarity score evidence
    low_sim_evidence = [{"similarity_score": 0.22, "brand_reply": "General response"}]
    decision_low_sim = decide(
        message="I left my phone in the car",
        intent=Intent.LOST_AND_FOUND.value,
        confidence=0.90,
        evidence=low_sim_evidence,
    )
    assert decision_low_sim.action == "ESCALATE"
    assert decision_low_sim.escalation_tier == "RETRIEVAL_GATE"


def test_policy_decide_auto_handle_success():
    """Verify low-risk, high-confidence, well-grounded inquiry triggers AUTO_HANDLE."""
    good_evidence = [
        {"similarity_score": 0.82, "brand_reply": "Visit [URL] to connect with driver for lost item."}
    ]
    decision = decide(
        message="I forgot my glasses and jacket in the vehicle",
        intent=Intent.LOST_AND_FOUND.value,
        confidence=0.92,
        evidence=good_evidence,
    )
    assert decision.action == "AUTO_HANDLE"
    assert decision.escalation_tier == "NONE"
    assert decision.is_auto_handled is True
    assert "low-risk self-service" in decision.reason
