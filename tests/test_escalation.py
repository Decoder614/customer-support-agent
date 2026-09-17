"""Tests for deterministic safety gatekeeper and escalation keyword interception."""

import pytest

from src.escalation.safety_rules import (
    SafetyDecision,
    SAFETY_PATTERNS,
    check_safety_risk,
    evaluate_safety_gate,
)


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
