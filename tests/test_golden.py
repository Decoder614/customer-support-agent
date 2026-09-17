"""Tests for Golden Evaluation Set benchmark integrity, row count, and zero leakage."""

from pathlib import Path
import pandas as pd
import pytest

from src.intents.taxonomy import INTENTS


def test_golden_template_schema_and_size():
    """Verify golden template exists, has exactly 200 rows, and matches expected schema."""
    template_path = Path("data/golden/golden_eval_template.csv")
    assert template_path.exists(), f"Missing {template_path}"

    df = pd.read_csv(template_path)
    assert len(df) == 200

    expected_cols = [
        "example_id",
        "customer_message",
        "conversation_context",
        "historical_brand_reply",
        "intent_gold",
        "action_gold",
        "escalation_reason_gold",
        "notes",
        "ambiguity",
        "annotator",
        "human_verified",
        "second_annotator",
        "adjudication_notes",
    ]
    assert list(df.columns) == expected_cols
    assert df["example_id"].notna().all()
    assert df["customer_message"].notna().all()


def test_golden_verified_integrity():
    """Verify verified golden set has 200 rows, valid intent classes, and no missing labels."""
    verified_path = Path("data/golden/golden_eval_verified.csv")
    assert verified_path.exists(), f"Missing {verified_path}"

    df = pd.read_csv(verified_path)
    assert len(df) == 200

    # No missing values in critical columns
    assert df["example_id"].notna().all()
    assert df["customer_message"].notna().all()
    assert df["intent_gold"].notna().all()
    assert df["action_gold"].notna().all()
    assert df["escalation_reason_gold"].notna().all()

    # Valid intent categories
    assert df["intent_gold"].isin(INTENTS).all()

    # All 9 intents covered
    covered_intents = set(df["intent_gold"].unique())
    assert len(covered_intents) == 9

    # Valid actions
    assert df["action_gold"].isin(["AUTO_HANDLE", "ESCALATE"]).all()


def test_golden_zero_leakage_with_train():
    """Verify strictly 0 example ID overlap between train.csv and golden_eval_verified.csv."""
    train_path = Path("data/processed/train.csv")
    verified_path = Path("data/golden/golden_eval_verified.csv")

    if train_path.exists() and verified_path.exists():
        train_df = pd.read_csv(train_path)
        gold_df = pd.read_csv(verified_path)

        train_ids = set(train_df["example_id"].astype(str))
        gold_ids = set(gold_df["example_id"].astype(str))

        overlap = train_ids & gold_ids
        assert len(overlap) == 0, f"Detected {len(overlap)} leaking IDs between train and golden benchmark!"
