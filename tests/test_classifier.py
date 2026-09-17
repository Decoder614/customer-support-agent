"""Tests for TrivialMajorityClassifier, SimpleTfidfClassifier, and IntentClassifier."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.intents.baseline_classifiers import (
    TrivialMajorityClassifier,
    SimpleTfidfClassifier,
)
from src.intents.classifier import IntentClassifier
from src.intents.taxonomy import Intent, INTENTS


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
    y_with_majority = y + [Intent.OTHER_GENERAL_FEEDBACK.value] * 5
    X_with_majority = X + ["Feedback sample"] * 5

    clf = TrivialMajorityClassifier()
    clf.fit(X_with_majority, y_with_majority)

    assert clf.majority_label == Intent.OTHER_GENERAL_FEEDBACK.value
    expected_conf = (2 + 5) / (len(y) + 5)
    assert pytest.approx(clf.majority_confidence_, rel=1e-3) == expected_conf

    pred_single = clf.predict("I lost my keys")
    assert pred_single == [Intent.OTHER_GENERAL_FEEDBACK.value]

    preds = clf.predict(["I lost my keys", "Driver was dangerous"])
    assert preds == [Intent.OTHER_GENERAL_FEEDBACK.value, Intent.OTHER_GENERAL_FEEDBACK.value]

    probs = clf.predict_proba(["Query A", "Query B"])
    assert probs.shape == (2, len(clf.classes_))
    assert np.allclose(probs.sum(axis=1), 1.0)

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

    probs = clf.predict_proba(test_queries)
    assert probs.shape == (3, len(clf.classes_))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

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

    pred_empty = clf.predict("")
    assert len(pred_empty) == 1

    probs_empty = clf.predict_proba("")
    assert probs_empty.shape == (1, len(clf.classes_))
    assert np.allclose(probs_empty.sum(axis=1), 1.0)


# ====================================================================
# IntentClassifier (Main Candidate) Tests
# ====================================================================

def test_intent_classifier_fit_predict(sample_training_data):
    """Verify IntentClassifier fits multi-ngram feature union and outputs confident predictions."""
    X, y = sample_training_data
    clf = IntentClassifier(c_value=4.0, seed=42)
    clf.fit(X, y)

    assert len(clf.classes_) > 0

    test_queries = [
        "I left my wallet in the back of the car",
        "Driver was driving recklessly and almost crashed",
        "Why was I charged a cancellation fee?",
    ]

    preds, confs = clf.predict_with_confidence(test_queries)
    assert len(preds) == 3
    assert len(confs) == 3
    for conf in confs:
        assert 0.0 <= conf <= 1.0

    probs = clf.predict_proba(test_queries)
    assert probs.shape == (3, len(clf.classes_))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_intent_classifier_zero_feature_fallback(sample_training_data):
    """Verify zero-feature fallback gracefully defaults to general feedback with 0.0 confidence."""
    X, y = sample_training_data
    clf = IntentClassifier(seed=42)
    clf.fit(X, y)

    # Empty string or non-informative punctuation
    preds, confs = clf.predict_with_confidence(["", "??? !!!"])
    assert len(preds) == 2
    assert preds[0] == Intent.OTHER_GENERAL_FEEDBACK.value
    assert confs[0] == 0.0


def test_intent_classifier_save_load(tmp_path, sample_training_data):
    """Verify serialization and deserialization with joblib."""
    X, y = sample_training_data
    clf = IntentClassifier(seed=42)
    clf.fit(X, y)

    model_path = tmp_path / "model.joblib"
    clf.save(model_path)
    assert model_path.exists()

    loaded = IntentClassifier.load(model_path)
    test_msg = "Driver was speeding"
    assert clf.predict(test_msg) == loaded.predict(test_msg)


def test_golden_set_benchmark_accuracy():
    """Verify trained Main Candidate classifier achieves >= 80% accuracy on verified golden set."""
    train_path = Path("data/processed/train.csv")
    gold_path = Path("data/golden/golden_eval_verified.csv")

    if train_path.exists() and gold_path.exists():
        train_df = pd.read_csv(train_path)
        gold_df = pd.read_csv(gold_path)

        clf = IntentClassifier(c_value=4.0, seed=42)
        clf.fit(train_df["customer_message"].tolist(), train_df["intent"].tolist())

        preds = clf.predict(gold_df["customer_message"].tolist())
        accuracy = (np.array(preds) == np.array(gold_df["intent_gold"])).mean()
        assert accuracy >= 0.80, f"Golden set accuracy was {accuracy:.2%}, expected >= 80%"
