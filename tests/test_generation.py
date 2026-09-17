"""Tests for grounded response drafter, escalation notices, and safety guardrails."""

import pytest

from src.generation.response_drafter import (
    ResponseDrafter,
    draft_response,
    is_unsafe_draft,
    get_grounded_resolution_step,
    ESCALATION_SAFE_NOTICE,
    SAFE_FALLBACK,
)
from src.intents.taxonomy import Intent


def test_draft_lost_and_found_resolution():
    """Verify grounded response generation for lost and found inquiries."""
    drafter = ResponseDrafter()
    evidence = [
        {"brand_reply": "Hi! You can contact the driver for your lost wallet via the app.", "similarity_score": 0.85}
    ]

    result = drafter.draft(
        intent=Intent.LOST_AND_FOUND.value,
        action="AUTO_HANDLE",
        evidence=evidence,
        message="I forgot my wallet in the car",
    )

    assert result.mode == "deterministic_grounded"
    assert result.requires_human_review is False
    assert result.policy_compliant is True
    assert "Find lost item" in result.reply_text
    assert "Your Trips" in result.reply_text


def test_draft_app_troubleshooting_resolution():
    """Verify grounded response generation for app troubleshooting."""
    drafter = ResponseDrafter()
    evidence = [
        {"brand_reply": "Try to force close and restart the app.", "similarity_score": 0.78}
    ]

    result = drafter.draft(
        intent=Intent.APP_AND_ACCOUNT_ACCESS.value,
        action="AUTO_HANDLE",
        evidence=evidence,
        message="App keeps crashing on payment screen",
    )

    assert result.mode == "deterministic_grounded"
    assert result.requires_human_review is False
    assert result.policy_compliant is True
    assert "restart" in result.reply_text.lower() or "latest version" in result.reply_text.lower()


def test_draft_escalation_notice():
    """Verify response drafter outputs polite escalation notice when action is ESCALATE."""
    drafter = ResponseDrafter()
    evidence = [{"brand_reply": "We will review your fare refund.", "similarity_score": 0.90}]

    result = drafter.draft(
        intent=Intent.FARE_DISPUTE_OVERCHARGE.value,
        action="ESCALATE",
        evidence=evidence,
        message="I was charged $50 for a $10 ride",
    )

    assert result.mode == "deterministic_escalation"
    assert result.requires_human_review is True
    assert result.policy_compliant is True
    assert result.reply_text == ESCALATION_SAFE_NOTICE


def test_is_unsafe_draft_guardrails():
    """Verify detection of hallucinated refunds, card details, or unauthorized commitments."""
    # Unsafe: Promises refund
    assert is_unsafe_draft("Your refund has been approved and we have credited $25.") is True
    assert is_unsafe_draft("We have refunded your fare in full.") is True
    assert is_unsafe_draft("We've issued a refund to your card.") is True

    # Unsafe: Asks for credentials
    assert is_unsafe_draft("Please send us your password and card number.") is True
    assert is_unsafe_draft("Please reply with your CVV and pin code.") is True

    # Unsafe: Raw HTTP links
    assert is_unsafe_draft("Click here: https://phishing-uber.com/login") is True

    # Safe: Standard guidance
    assert is_unsafe_draft("You can contact your driver directly through the Uber app.") is False
    assert is_unsafe_draft(ESCALATION_SAFE_NOTICE) is False
    assert is_unsafe_draft(SAFE_FALLBACK) is False


def test_functional_draft_response_wrapper():
    """Verify dictionary output of draft_response helper function."""
    payload = draft_response(
        intent=Intent.LOST_AND_FOUND.value,
        action="AUTO_HANDLE",
        evidence=[{"brand_reply": "Contact driver for lost item."}],
    )
    assert isinstance(payload, dict)
    assert "reply_text" in payload
    assert "mode" in payload
    assert payload["policy_compliant"] is True
