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


def test_leakage_assertion_detection():
    import pandas as pd
    from src.data.split import assert_no_leakage

    df_train = pd.DataFrame({
        'example_id': ['1', '2', '3'],
        'conversation_id': ['c1', 'c2', 'c3'],
        'text': ['item lost in car', 'fare dispute overcharge', 'driver rude']
    })
    df_eval = pd.DataFrame({
        'example_id': ['4', '5'],
        'conversation_id': ['c4', 'c5'],
        'text': ['clean distinct text A', 'clean distinct text B']
    })
    # Should pass without error
    assert_no_leakage(df_train, df_eval)

    # Overlap in conversation_id should raise ValueError
    df_leaky = pd.DataFrame({
        'example_id': ['6', '7'],
        'conversation_id': ['c1', 'c6'],
        'text': ['another text 1', 'another text 2']
    })
    with pytest.raises(ValueError, match="Data leakage detected"):
        assert_no_leakage(df_train, df_leaky)


def test_create_leak_free_splits_synthetic():
    import pandas as pd
    from src.data.split import create_leak_free_splits, assert_no_leakage

    # Construct conversation turns where turns with same conv_id or duplicate text must stay together
    df = pd.DataFrame({
        'example_id': [str(i) for i in range(1, 11)],
        'conversation_id': ['c1', 'c1', 'c2', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'],
        'text': [
            'lost my wallet in uber car',
            'still trying to find my wallet in uber car',
            'driver overcharged me on route',
            'route was changed and overcharged',
            'app crashed during payment',
            'promo code not working at checkout',
            'promo code discount failed to apply at checkout',
            'driver was very polite and helpful',
            'where is my lost bag',
            'cancel fee charged incorrectly'
        ]
    })

    train, dev, eval_df = create_leak_free_splits(df, train_ratio=0.5, dev_ratio=0.25, eval_ratio=0.25, seed=42)

    # Check non-empty
    assert len(train) > 0
    assert len(dev) > 0
    assert len(eval_df) > 0
    assert len(train) + len(dev) + len(eval_df) == len(df)

    # Assert 0 leakage across all pairs
    assert_no_leakage(train, dev)
    assert_no_leakage(train, eval_df)
    assert_no_leakage(dev, eval_df)

    # Ensure all turns of conversation c1 are in the same split
    c1_splits = {
        'train': 'c1' in train['conversation_id'].values,
        'dev': 'c1' in dev['conversation_id'].values,
        'eval': 'c1' in eval_df['conversation_id'].values
    }
    assert sum(c1_splits.values()) == 1

