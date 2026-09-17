"""Main Candidate Intent Classifier using subword character/word TF-IDF and heuristic boosting.

Combines word (1, 2) and character within-boundary char_wb (3, 5) n-grams with
balanced class weights, rule-based keyword boosting, and zero-feature fallback handling.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data.preprocessing import redact_pii, normalize_text
from src.intents.taxonomy import Intent, INTENTS, propose


class IntentClassifier:
    """High-accuracy multi-class Intent Classifier for customer support interactions."""

    def __init__(
        self,
        c_value: float = 4.0,
        max_iter: int = 1000,
        word_ngram_range: Tuple[int, int] = (1, 2),
        char_ngram_range: Tuple[int, int] = (3, 5),
        max_word_features: int = 25000,
        max_char_features: int = 35000,
        class_weight: Optional[Union[str, Dict[str, float]]] = "balanced",
        use_rule_boost: bool = True,
        seed: int = 42,
    ) -> None:
        self.c_value = c_value
        self.max_iter = max_iter
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.max_word_features = max_word_features
        self.max_char_features = max_char_features
        self.class_weight = class_weight
        self.use_rule_boost = use_rule_boost
        self.seed = seed

        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.word_ngram_range,
            sublinear_tf=True,
            max_features=self.max_word_features,
            min_df=1,
            lowercase=True,
            strip_accents="unicode",
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.char_ngram_range,
            sublinear_tf=True,
            max_features=self.max_char_features,
            min_df=1,
            lowercase=True,
            strip_accents="unicode",
        )
        self.features = FeatureUnion([
            ("word", self.word_vectorizer),
            ("char", self.char_vectorizer),
        ])
        self.classifier = LogisticRegression(
            C=self.c_value,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.seed,
        )
        self.model = Pipeline([
            ("features", self.features),
            ("classifier", self.classifier),
        ])
        self.classes_: np.ndarray = np.array([])
        self.fallback_intent: str = Intent.OTHER_GENERAL_FEEDBACK.value

    def _preprocess(self, texts: Union[List[str], np.ndarray, str]) -> List[str]:
        """Ensures text is sanitized, normalized, and PII-redacted for feature extraction."""
        if isinstance(texts, str):
            texts = [texts]
        cleaned = []
        for t in texts:
            if t is None:
                cleaned.append("")
            else:
                s = str(t)
                cleaned.append(normalize_text(redact_pii(s)) if "@" in s or "http" in s else s.lower().strip())
        return cleaned

    def fit(
        self,
        messages: Union[List[str], np.ndarray],
        labels: Union[List[str], np.ndarray],
    ) -> "IntentClassifier":
        """Fits TF-IDF feature unions and Logistic Regression classifier.

        Args:
            messages: Raw or normalized text messages.
            labels: Ground-truth or silver intent labels.
        """
        if len(messages) == 0 or len(labels) == 0:
            raise ValueError("Cannot fit IntentClassifier on empty training data.")

        clean_messages = self._preprocess(messages)
        clean_labels = list(labels)

        if len(set(clean_labels)) < 2:
            raise ValueError("Training needs at least two distinct intent classes.")

        self.model.fit(clean_messages, clean_labels)
        self.classes_ = np.array(self.model.named_steps["classifier"].classes_)
        return self

    def predict_proba(self, messages: Union[List[str], np.ndarray, str]) -> np.ndarray:
        """Computes class probability distribution for input messages.

        Returns array of shape (n_samples, n_classes).
        """
        if len(self.classes_) == 0:
            raise ValueError("IntentClassifier is not fitted yet.")

        clean_messages = self._preprocess(messages)
        if not clean_messages:
            return np.empty((0, len(self.classes_)))

        probs = self.model.predict_proba(clean_messages)

        if self.use_rule_boost:
            for idx, msg in enumerate(clean_messages):
                rule_intent, is_ambiguous = propose(msg)
                if not is_ambiguous and rule_intent != Intent.OTHER_GENERAL_FEEDBACK.value:
                    if rule_intent in self.classes_:
                        cls_idx = int(np.where(self.classes_ == rule_intent)[0][0])
                        # Boost probability distribution towards rule intent
                        boosted_row = np.zeros(len(self.classes_))
                        boosted_row[cls_idx] = 0.95
                        remaining = 0.05 / max(1, len(self.classes_) - 1)
                        for c_i in range(len(self.classes_)):
                            if c_i != cls_idx:
                                boosted_row[c_i] = remaining
                        probs[idx] = boosted_row

        return probs

    def predict(self, messages: Union[List[str], np.ndarray, str]) -> List[str]:
        """Predicts the most probable intent label for each message."""
        preds, _ = self.predict_with_confidence(messages)
        return preds

    def predict_with_confidence(
        self,
        messages: Union[List[str], np.ndarray, str],
    ) -> Tuple[List[str], List[float]]:
        """Predicts intent labels and calibrated confidence scores with zero-feature fallback handling.

        Args:
            messages: Input string or list/array of text messages.

        Returns:
            Tuple of (predicted_intents, confidence_scores).
        """
        if len(self.classes_) == 0:
            raise ValueError("IntentClassifier is not fitted yet.")

        clean_messages = self._preprocess(messages)
        if not clean_messages:
            return [], []

        probabilities = self.predict_proba(clean_messages)
        max_indices = np.argmax(probabilities, axis=1)
        labels = self.classes_[max_indices].tolist()
        confidences = [float(probabilities[i, max_indices[i]]) for i in range(len(max_indices))]

        # Zero-feature fallback check: if an input transformed to an all-zero sparse vector
        try:
            matrix = self.model.named_steps["features"].transform(clean_messages)
            nnz_counts = np.asarray(matrix.getnnz(axis=1)).ravel()
            for idx, count in enumerate(nnz_counts):
                if count == 0 and not (self.use_rule_boost and propose(clean_messages[idx])[0] != Intent.OTHER_GENERAL_FEEDBACK.value):
                    labels[idx] = self.fallback_intent
                    confidences[idx] = 0.0
        except Exception:
            pass

        return labels, confidences

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes the fitted classifier model using joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "IntentClassifier":
        """Loads a serialized IntentClassifier instance."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        return joblib.load(path)
