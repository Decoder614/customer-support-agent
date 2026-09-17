"""Tests for historical resolution retrieval engine, indexing, ranking, and intent gating."""

import pandas as pd
import pytest

from src.retrieval.retrieval_index import (
    RetrievalIndex,
    sanitize_evidence,
)
from src.intents.taxonomy import Intent


@pytest.fixture
def sample_training_df():
    """Provides representative training historical interaction pairs."""
    data = [
        {
            "example_id": "101",
            "customer_message": "I forgot my wallet and phone in the Uber car yesterday",
            "text": "i forgot my wallet and phone in the uber car yesterday",
            "historical_brand_reply": "Hi John! We can help. Visit [URL] to contact your driver. /JD",
            "conversation_context": "",
            "intent": Intent.LOST_AND_FOUND.value,
        },
        {
            "example_id": "102",
            "customer_message": "Driver was speeding recklessly on highway",
            "text": "driver was speeding recklessly on highway",
            "historical_brand_reply": "Hey Sarah! We take safety seriously. A specialist will call you. ^TS",
            "conversation_context": "",
            "intent": Intent.SAFETY_AND_CONDUCT.value,
        },
        {
            "example_id": "103",
            "customer_message": "Why was I charged a cancellation fee of $5 when driver cancelled?",
            "text": "why was i charged a cancellation fee of $5 when driver cancelled",
            "historical_brand_reply": "Hello! Send us a DM at [URL] so our fare team can review the cancellation charge.",
            "conversation_context": "",
            "intent": Intent.CANCELLATION_FEE.value,
        },
        {
            "example_id": "104",
            "customer_message": "App crashes whenever I open the payment settings page on iPhone",
            "text": "app crashes whenever i open the payment settings page on iphone",
            "historical_brand_reply": "Hi there! Try reinstalling the Uber app or updating iOS. /KL",
            "conversation_context": "",
            "intent": Intent.APP_AND_ACCOUNT_ACCESS.value,
        },
    ]
    return pd.DataFrame(data)


def test_sanitize_evidence():
    """Verify evidence sanitization cleans names, URLs, PII, and agent signatures."""
    raw = "Hi Robert! We're sorry. Reach out to support@uber.com or call 555-123-4567 [URL] /XYZ"
    sanitized = sanitize_evidence(raw)
    assert "[EMAIL]" in sanitized or "support" in sanitized
    assert "Hi!" in sanitized
    assert "/XYZ" not in sanitized


def test_retrieval_ranking_and_top_k(sample_training_df):
    """Verify top-k retrieval ranking returns most similar candidate first."""
    index = RetrievalIndex(sample_training_df)
    assert len(index) == 4

    query = "I left my wallet in the car"
    results = index.search(query, top_k=2, min_similarity=0.10)

    assert len(results) >= 1
    top_result = results[0]
    assert top_result["example_id"] == "101"
    assert top_result["intent"] == Intent.LOST_AND_FOUND.value
    assert top_result["similarity_score"] > 0.30
    assert "Hi!" in top_result["brand_reply"]
    assert "/JD" not in top_result["brand_reply"]


def test_retrieval_intent_gating(sample_training_df):
    """Verify intent filtering restricts candidates to predicted intent class."""
    index = RetrievalIndex(sample_training_df)

    query = "Why was I charged a fee?"
    # When filtered for cancellation_fee
    results_cf = index.search(query, top_k=2, min_similarity=0.05, intent=Intent.CANCELLATION_FEE.value)
    assert len(results_cf) >= 1
    assert all(r["intent"] == Intent.CANCELLATION_FEE.value for r in results_cf)

    # When filtered for unrelated intent (e.g. driver_partner_inquiry)
    results_unmatched = index.search(query, top_k=2, min_similarity=0.05, intent=Intent.DRIVER_PARTNER_INQUIRY.value)
    assert len(results_unmatched) == 0


def test_retrieval_threshold_fallback(sample_training_df):
    """Verify high threshold or irrelevant queries return empty list fallback."""
    index = RetrievalIndex(sample_training_df)

    # Completely irrelevant query with standard 0.30 threshold
    results = index.search("Quantum physics and galaxy astrophysics", min_similarity=0.30)
    assert results == []


def test_retrieval_empty_query_and_empty_index(sample_training_df):
    """Verify handling of empty queries or empty index instances."""
    index = RetrievalIndex(sample_training_df)
    assert index.search("") == []
    assert index.search("   ") == []
    assert index.search("hello", top_k=0) == []

    empty_index = RetrievalIndex(pd.DataFrame())
    assert empty_index.search("hello") == []
