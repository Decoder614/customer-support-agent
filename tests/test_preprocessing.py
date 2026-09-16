"""Unit tests for PII redaction and text normalization utilities."""
import pytest
from src.data.preprocessing import redact_pii, normalize_text, redact, normalize


def test_redact_pii_url():
    sample = "Please check https://help.uber.com/riders or http://t.co/abc1234 for updates."
    expected = "Please check [URL] or [URL] for updates."
    assert redact_pii(sample) == expected


def test_redact_pii_email():
    sample = "You can email me at customer.support+ride@example.com or test@uber.co.uk."
    expected = "You can email me at [EMAIL] or [EMAIL]."
    assert redact_pii(sample) == expected


def test_redact_pii_handle():
    sample = "Hey @Uber_Support, @driver_123 cancelled my trip unexpectedly!"
    expected = "Hey [USER], [USER] cancelled my trip unexpectedly!"
    assert redact_pii(sample) == expected


def test_redact_pii_phone():
    sample = "Call me back at +1-800-555-0199 or (555) 234-5678 immediately."
    result = redact_pii(sample)
    assert "[PHONE]" in result
    assert "+1-800-555-0199" not in result
    assert "(555) 234-5678" not in result


def test_redact_pii_html_entities():
    sample = "I was charged $45 &amp; driver was rude &lt;issue&gt;"
    expected = "I was charged $45 & driver was rude <issue>"
    assert redact_pii(sample) == expected


def test_redact_pii_edge_cases():
    assert redact_pii(None) == ""
    assert redact_pii("") == ""
    assert redact_pii("   \n\t  ") == ""
    assert redact_pii(12345) == "[PHONE]" or redact_pii(12345) == "12345"


def test_normalize_text():
    sample = "Hey @Uber_Support! Check https://uber.com for receipt #12345 &amp; email support@uber.com"
    normalized = normalize_text(sample)
    assert "@uber_support" not in normalized
    assert "https://uber.com" not in normalized
    assert "support@uber.com" not in normalized
    assert "check" in normalized
    assert "receipt" in normalized


def test_aliases():
    assert redact("hello @user") == redact_pii("hello @user")
    assert normalize("hello @user") == normalize_text("hello @user")
