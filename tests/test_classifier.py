"""Tests for TrivialMajorityClassifier and SimpleTfidfClassifier baselines."""

import numpy as np
import pytest

from src.intents.baseline_classifiers import (
    TrivialMajorityClassifier,
    SimpleTfidfClassifier,
)
from src.intents.taxonomy import Intent


@pytest.fixture
def sample_training_data():
    """Provides representative training samples across multiple intent categories."""
    X = [
        "Driver was reckless and speeding dangerously",
        "Driver threatened me and was intoxicated",
        "I was charged a $5 cancellation fee",
        "Charged when driver cancelled on me",
        "Overcharged on my fare, double charged $30",
        "Wrong fare charged for short trip",
        "I left my wallet and phone in the back seat",
        "Forgot my jacket and backpack in the car",
        "Driver went the wrong way and dropped me off at wrong pin",
        "App keeps crashing when I try to log in",
        "Promo code discount was not applied to order",
        "When will my driver partner payout be processed?",
        "Thank you for the quick ride today",
        "Good morning support team",
    ]
    y = [
        Intent.SAFETY_AND_CONDUCT.value,
        Intent.SAFETY_AND_CONDUCT.value,
        Intent.CANCELLATION_FEE.value,
        Intent.CANCELLATION_FEE.value,
        Intent.FARE_DISPUTE_OVERCHARGE.value,
        Intent.FARE_DISPUTE_OVERCHARGE.value,
        Intent.LOST_AND_FOUND.value,
        Intent.LOST_AND_FOUND.value,
        Intent.PICKUP_ROUTING_ISSUE.value,
        Intent.APP_AND_ACCOUNT_ACCESS.value,
        Intent.PROMOTIONS_AND_UBEREATS.value,
        Intent.DRIVER_PARTNER_INQUIRY.value,
        Intent.OTHER_GENERAL_FEEDBACK.value,
        Intent.OTHER_GENERAL_FEEDBACK.value,
    ]
    return X, y


# ====================================================================
# TrivialMajorityClassifier Tests
# ====================================================================

def test_trivial_majority_fit_predict(sample_training_data):
    """Verify TrivialMajorityClassifier predicts the most frequent training class."""
    X, y = sample_training_data
    # Add extra general feedback to make it the undisputed majority class
    y_with_majority = y + [Intent.OTHER_GENERAL_FEEDBACK.value] * 5
    X_with_majority = X + ["Feedback sample"] * 5

    clf = TrivialMajorityClassifier()
    clf.fit(X_with_majority, y_with_majority)

    assert clf.majority_label == Intent.OTHER_GENERAL_FEEDBACK.value
    expected_conf = (2 + 5) / (len(y) + 5)
    assert pytest.approx(clf.majority_confidence_, rel=1e-3) == expected_conf

    # Predict single string
    pred_single = clf.predict("I lost my keys")
    assert pred_single == [Intent.OTHER_GENERAL_FEEDBACK.value]

    # Predict list of strings
    preds = clf.predict(["I lost my keys", "Driver was dangerous"])
    assert preds == [Intent.OTHER_GENERAL_FEEDBACK.value, Intent.OTHER_GENERAL_FEEDBACK.value]

    # Predict probabilities
    probs = clf.predict_proba(["Query A", "Query B"])
    assert probs.shape == (2, len(clf.classes_))
    # Each row sums to 1.0
    assert np.allclose(probs.sum(axis=1), 1.0)

    # Predict with confidence
    preds, confs = clf.predict_with_confidence(["Query 1", "Query 2"])
    assert len(preds) == 2
    assert len(confs) == 2
    assert all(c == pytest.approx(clf.majority_confidence_) for c in confs)


def test_trivial_majority_errors():
    """Verify error conditions for un-fitted or empty data."""
    clf = TrivialMajorityClassifier()
    with pytest.raises(ValueError, match="is not fitted"):
        clf.predict(["test"])

    with pytest.raises(ValueError, match="Cannot fit"):
        clf.fit([], [])


# ====================================================================
# SimpleTfidfClassifier Tests
# ====================================================================

def test_simple_tfidf_fit_predict(sample_training_data):
    """Verify SimpleTfidfClassifier fits and outputs valid predictions and probability distributions."""
    X, y = sample_training_data
    clf = SimpleTfidfClassifier(c_value=2.0, random_state=42)
    clf.fit(X, y)

    assert len(clf.classes_) > 0

    test_queries = [
        "I forgot my wallet in the car",
        "Driver was speeding and driving reckless",
        "Why was I charged a cancellation fee?",
    ]

    preds = clf.predict(test_queries)
    assert len(preds) == 3
    assert isinstance(preds[0], str)

    # Check probabilities
    probs = clf.predict_proba(test_queries)
    assert probs.shape == (3, len(clf.classes_))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Check predict_with_confidence
    preds, confs = clf.predict_with_confidence(test_queries)
    assert len(preds) == 3
    assert len(confs) == 3
    for conf in confs:
        assert 0.0 <= conf <= 1.0


def test_simple_tfidf_edge_cases(sample_training_data):
    """Verify handling of empty strings, single string inputs, and un-fitted exceptions."""
    X, y = sample_training_data
    clf = SimpleTfidfClassifier()

    with pytest.raises(ValueError, match="is not fitted"):
        clf.predict("test")

    with pytest.raises(ValueError, match="Cannot fit"):
        clf.fit([], [])

    clf.fit(X, y)

    # Empty string query
    pred_empty = clf.predict("")
    assert len(pred_empty) == 1

    probs_empty = clf.predict_proba("")
    assert probs_empty.shape == (1, len(clf.classes_))
    assert np.allclose(probs_empty.sum(axis=1), 1.0)
