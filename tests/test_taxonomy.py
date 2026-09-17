"""Tests for intent taxonomy definitions, enums, patterns, and priority heuristics."""

from src.intents.taxonomy import (
    Intent,
    INTENTS,
    HIGH_RISK_INTENTS,
    PRIORITY,
    PATTERNS,
    TAXONOMY_METADATA,
    propose,
    is_high_risk,
)


def test_taxonomy_intent_count():
    """Verify exactly 9 canonical intents are defined."""
    assert len(Intent) == 9
    assert len(INTENTS) == 9
    assert len(set(INTENTS)) == 9


def test_taxonomy_metadata_completeness():
    """Verify all 9 intents have complete metadata entries."""
    for intent in INTENTS:
        assert intent in TAXONOMY_METADATA
        meta = TAXONOMY_METADATA[intent]
        assert "name" in meta and len(meta["name"]) > 0
        assert "description" in meta and len(meta["description"]) > 0
        assert "routing_action" in meta and len(meta["routing_action"]) > 0
        assert isinstance(meta["is_high_risk"], bool)
        assert "example_utterance" in meta and len(meta["example_utterance"]) > 0


def test_high_risk_intents():
    """Verify high-risk intent classifications."""
    assert len(HIGH_RISK_INTENTS) == 4
    assert is_high_risk("safety_and_conduct") is True
    assert is_high_risk("fare_dispute_overcharge") is True
    assert is_high_risk("cancellation_fee") is True
    assert is_high_risk("driver_partner_inquiry") is True
    assert is_high_risk("lost_and_found") is False
    assert is_high_risk("other_general_feedback") is False


def test_propose_heuristics():
    """Verify keyword proposal function for representative inputs."""
    # Safety
    intent, ambiguous = propose("Driver was drunk and threatening passengers")
    assert intent == "safety_and_conduct"

    # Cancellation fee
    intent, ambiguous = propose("I was charged a cancellation fee even though driver did not show up")
    assert intent == "cancellation_fee"

    # Fare dispute
    intent, ambiguous = propose("I was overcharged twice for my ride yesterday")
    assert intent == "fare_dispute_overcharge"

    # Lost and found
    intent, ambiguous = propose("I left my wallet and keys in the back seat")
    assert intent == "lost_and_found"

    # Driver partner
    intent, ambiguous = propose("When will my driver payout earnings be deposited?")
    assert intent == "driver_partner_inquiry"

    # App access
    intent, ambiguous = propose("My account is locked out and I cannot get the OTP code")
    assert intent == "app_and_account_access"

    # Promotions & Uber Eats
    intent, ambiguous = propose("My promo code coupon did not apply to my food delivery order")
    assert intent == "promotions_and_ubereats"

    # Other / General feedback
    intent, ambiguous = propose("Just wanted to say thanks for the great service today!")
    assert intent == "other_general_feedback"


def test_propose_precedence_overrides():
    """Verify that safety takes precedence over fare disputes."""
    # Mentions both overcharge and driver harassment/threat
    mixed_msg = "The driver harassed me and threatened me, and also overcharged my fare"
    intent, ambiguous = propose(mixed_msg)
    assert intent == "safety_and_conduct"
    assert ambiguous is True  # Ambiguous because multiple patterns matched


def test_propose_edge_cases():
    """Verify handling of empty or non-string inputs."""
    intent, ambiguous = propose("")
    assert intent == "other_general_feedback"
    assert ambiguous is True

    intent, ambiguous = propose(None)  # type: ignore
    assert intent == "other_general_feedback"
    assert ambiguous is True
