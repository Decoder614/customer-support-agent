"""Baseline Intent Classifiers for Uber Support Agent.

Includes:
1. TrivialMajorityClassifier: Zero-rule majority class predictor baseline.
2. SimpleTfidfClassifier: Word-level unigram TF-IDF + Logistic Regression baseline.
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TrivialMajorityClassifier:
    """Trivial Baseline classifier that always predicts the most frequent training class."""

    def __init__(self) -> None:
        self.majority_label: Optional[str] = None
        self.classes_: np.ndarray = np.array([])
        self.class_priors_: Dict[str, float] = {}
        self.majority_confidence_: float = 0.0

    def fit(self, X: Union[List[str], np.ndarray], y: Union[List[str], np.ndarray]) -> "TrivialMajorityClassifier":
        """Fits majority class and empirical prior distributions.

        Args:
            X: List or array of text samples (unused).
            y: List or array of class labels.
        """
        if len(y) == 0:
            raise ValueError("Cannot fit TrivialMajorityClassifier on empty labels.")

        counts = Counter(y)
        total = len(y)
        self.majority_label, majority_count = counts.most_common(1)[0]
        self.majority_confidence_ = majority_count / total

        # Store sorted unique classes and priors
        self.classes_ = np.array(sorted(counts.keys()))
        self.class_priors_ = {cls: counts[cls] / total for cls in self.classes_}
        return self

    def predict(self, X: Union[List[str], np.ndarray, str]) -> List[str]:
        """Predicts the majority class for all inputs."""
        if self.majority_label is None:
            raise ValueError("Classifier is not fitted yet.")

        if isinstance(X, str):
            X = [X]

        return [self.majority_label] * len(X)

    def predict_proba(self, X: Union[List[str], np.ndarray, str]) -> np.ndarray:
        """Returns empirical class probability distribution for each input sample."""
        if self.majority_label is None:
            raise ValueError("Classifier is not fitted yet.")

        if isinstance(X, str):
            X = [X]

        n_samples = len(X)
        probs_row = np.array([self.class_priors_[cls] for cls in self.classes_])
        return np.tile(probs_row, (n_samples, 1))

    def predict_with_confidence(self, X: Union[List[str], np.ndarray, str]) -> Tuple[List[str], List[float]]:
        """Returns tuple of (predictions, confidences)."""
        preds = self.predict(X)
        confs = [self.majority_confidence_] * len(preds)
        return preds, confs


class SimpleTfidfClassifier:
    """Simple Baseline classifier using Word-level Unigram TF-IDF + Logistic Regression."""

    def __init__(
        self,
        c_value: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.c_value = c_value
        self.max_iter = max_iter
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 1),
            min_df=1,
            lowercase=True,
            strip_accents="unicode",
        )
        self.model = LogisticRegression(
            C=self.c_value,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.pipeline: Optional[Pipeline] = None
        self.classes_: np.ndarray = np.array([])
        self.fallback_class_: Optional[str] = None

    def fit(self, X: Union[List[str], np.ndarray], y: Union[List[str], np.ndarray]) -> "SimpleTfidfClassifier":
        """Fits word TF-IDF vectorizer and Logistic Regression model.

        Args:
            X: List or array of text samples.
            y: List or array of class labels.
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Cannot fit SimpleTfidfClassifier on empty data.")

        # Clean text inputs
        clean_X = [str(x) if x is not None else "" for x in X]
        labels = list(y)

        counts = Counter(labels)
        self.fallback_class_ = counts.most_common(1)[0][0]

        self.pipeline = Pipeline([
            ("tfidf", self.vectorizer),
            ("clf", self.model),
        ])
        self.pipeline.fit(clean_X, labels)
        self.classes_ = np.array(self.pipeline.named_steps["clf"].classes_)
        return self

    def predict(self, X: Union[List[str], np.ndarray, str]) -> List[str]:
        """Predicts class labels for text inputs."""
        if self.pipeline is None:
            raise ValueError("Classifier is not fitted yet.")

        if isinstance(X, str):
            X = [X]

        clean_X = [str(x) if x is not None else "" for x in X]
        return list(self.pipeline.predict(clean_X))

    def predict_proba(self, X: Union[List[str], np.ndarray, str]) -> np.ndarray:
        """Computes class probability distribution for text inputs."""
        if self.pipeline is None:
            raise ValueError("Classifier is not fitted yet.")

        if isinstance(X, str):
            X = [X]

        clean_X = [str(x) if x is not None else "" for x in X]
        return self.pipeline.predict_proba(clean_X)

    def predict_with_confidence(self, X: Union[List[str], np.ndarray, str]) -> Tuple[List[str], List[float]]:
        """Returns tuple of (predictions, confidences).

        Confidence is defined as max predicted class probability.
        """
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        preds = [self.classes_[idx] for idx in pred_indices]
        confs = [float(probs[i, idx]) for i, idx in enumerate(pred_indices)]
        return preds, confs
