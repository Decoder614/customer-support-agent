"""Intent classification package for Uber Support Agent."""

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
from src.intents.baseline_classifiers import (
    TrivialMajorityClassifier,
    SimpleTfidfClassifier,
)
from src.intents.classifier import IntentClassifier

__all__ = [
    "Intent",
    "INTENTS",
    "HIGH_RISK_INTENTS",
    "PRIORITY",
    "PATTERNS",
    "TAXONOMY_METADATA",
    "propose",
    "is_high_risk",
    "TrivialMajorityClassifier",
    "SimpleTfidfClassifier",
    "IntentClassifier",
]
