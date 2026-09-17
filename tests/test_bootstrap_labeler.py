"""Tests for keyword bootstrap labeler, summaries, and action assignments."""

import pandas as pd
import pytest

from src.data.bootstrap_labeler import (
    bootstrap_labels,
    generate_label_summary,
    determine_default_action,
)
from src.intents.taxonomy import Intent, INTENTS


def test_bootstrap_labels_basic():
    """Verify that bootstrap labeling correctly assigns silver intent, ambiguity, and action."""
    sample_df = pd.DataFrame([
        {"customer_message": "Driver was speeding and scary", "text": "driver was speeding and scary"},
        {"customer_message": "I was charged $15 for cancellation fee", "text": "i was charged $15 for cancellation fee"},
        {"customer_message": "I left my wallet in the cab", "text": "i left my wallet in the cab"},
        {"customer_message": "App crashes during login", "text": "app crashes during login"},
        {"customer_message": "Just saying hello", "text": "just saying hello"},
    ])

    labeled = bootstrap_labels(sample_df)

    assert "intent" in labeled.columns
    assert "ambiguity" in labeled.columns
    assert "action" in labeled.columns
    assert "label_source" in labeled.columns

    assert labeled.iloc[0]["intent"] == Intent.SAFETY_AND_CONDUCT.value
    assert labeled.iloc[0]["action"] == "ESCALATE"

    assert labeled.iloc[1]["intent"] == Intent.CANCELLATION_FEE.value
    assert labeled.iloc[1]["action"] == "ESCALATE"

    assert labeled.iloc[2]["intent"] == Intent.LOST_AND_FOUND.value
    assert labeled.iloc[2]["action"] == "AUTO_HANDLE"

    assert labeled.iloc[3]["intent"] == Intent.APP_AND_ACCOUNT_ACCESS.value
    assert labeled.iloc[3]["action"] == "AUTO_HANDLE"

    assert labeled.iloc[4]["intent"] == Intent.OTHER_GENERAL_FEEDBACK.value
    assert labeled.iloc[4]["action"] == "ESCALATE"  # General feedback/unclassified escalated/reviewed by default


def test_determine_default_action():
    """Verify action escalation policy logic."""
    # High risk intents -> ESCALATE
    assert determine_default_action("safety_and_conduct", is_ambiguous=False) == "ESCALATE"
    assert determine_default_action("cancellation_fee", is_ambiguous=False) == "ESCALATE"
    assert determine_default_action("fare_dispute_overcharge", is_ambiguous=False) == "ESCALATE"
    assert determine_default_action("driver_partner_inquiry", is_ambiguous=False) == "ESCALATE"

    # Ambiguous -> ESCALATE
    assert determine_default_action("lost_and_found", is_ambiguous=True) == "ESCALATE"

    # Other general feedback -> ESCALATE
    assert determine_default_action("other_general_feedback", is_ambiguous=False) == "ESCALATE"

    # Low risk, unambiguous -> AUTO_HANDLE
    assert determine_default_action("lost_and_found", is_ambiguous=False) == "AUTO_HANDLE"
    assert determine_default_action("pickup_routing_issue", is_ambiguous=False) == "AUTO_HANDLE"
    assert determine_default_action("app_and_account_access", is_ambiguous=False) == "AUTO_HANDLE"
    assert determine_default_action("promotions_and_ubereats", is_ambiguous=False) == "AUTO_HANDLE"


def test_generate_label_summary():
    """Verify distribution summary calculation."""
    sample_df = pd.DataFrame([
        {"intent": "lost_and_found", "ambiguity": False, "action": "AUTO_HANDLE"},
        {"intent": "lost_and_found", "ambiguity": True, "action": "ESCALATE"},
        {"intent": "safety_and_conduct", "ambiguity": False, "action": "ESCALATE"},
    ])

    summary = generate_label_summary(sample_df)
    assert summary["total_rows"] == 3
    assert summary["intent_distribution"]["lost_and_found"] == 2
    assert summary["intent_distribution"]["safety_and_conduct"] == 1
    # Check that all 9 intents exist in dictionary
    for intent in INTENTS:
        assert intent in summary["intent_distribution"]
    assert summary["ambiguity_count"] == 1
    assert summary["action_distribution"]["ESCALATE"] == 2
    assert summary["action_distribution"]["AUTO_HANDLE"] == 1


def test_bootstrap_empty_df():
    """Verify error on empty DataFrame."""
    with pytest.raises(ValueError):
        bootstrap_labels(pd.DataFrame())
