"""Historical Resolution Retrieval Engine for Uber Support Agent.

Indexes historical training interaction pairs strictly from train.csv to retrieve
grounded resolution precedents using sparse TF-IDF and cosine similarity search.
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.preprocessing import redact_pii, normalize_text


def sanitize_evidence(text: str) -> str:
    """Sanitizes retrieved text by redacting PII, normalizing greetings, and stripping signatures."""
    if not text or not isinstance(text, str):
        return ""
    # Redact sensitive tokens
    text = redact_pii(text)
    # Genericize direct agent greetings (e.g. "Hi John!" -> "Hi!")
    text = re.sub(r"\b(Hi|Hey|Hello)\s+(?!there\b|everyone\b|all\b)[A-Z][a-z]+[!,.]", r"\1!", text)
    # Strip agent sign-off codes (e.g. " /AB", " ^JD")
    text = re.sub(r"\s+[/^][A-Z]{1,4}\b", "", text)
    return text.strip()


class RetrievalResult(TypedDict):
    """Schema for retrieved resolution precedent."""
    example_id: str
    customer_message: str
    brand_reply: str
    intent: str
    conversation_context: str
    similarity_score: float


class RetrievalIndex:
    """TF-IDF Sparse Index for historical customer support resolutions."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 40000,
        text_column: str = "text",
        fallback_column: str = "customer_message",
    ) -> None:
        """Initializes and builds the retrieval index strictly over training data."""
        if train_df is None or len(train_df) == 0:
            self.examples = pd.DataFrame()
            self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, sublinear_tf=True)
            self.matrix = None
            return

        # Ensure index only holds unique training utterances
        self.examples = train_df.drop_duplicates(subset=[text_column] if text_column in train_df.columns else [fallback_column]).reset_index(drop=True).copy()

        # Determine index text series
        if text_column in self.examples.columns:
            index_texts = self.examples[text_column].fillna("").astype(str)
        elif fallback_column in self.examples.columns:
            index_texts = self.examples[fallback_column].fillna("").astype(str).map(lambda s: normalize_text(redact_pii(s)))
        else:
            raise ValueError(f"Neither '{text_column}' nor '{fallback_column}' found in training DataFrame.")

        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            sublinear_tf=True,
            max_features=max_features,
            lowercase=True,
            strip_accents="unicode",
        )
        self.matrix = self.vectorizer.fit_transform(index_texts)

    @classmethod
    def from_file(cls, train_path: Union[str, Path] = "data/processed/train.csv") -> "RetrievalIndex":
        """Factory method to load and build index strictly from train.csv."""
        path = Path(train_path)
        if not path.exists():
            raise FileNotFoundError(f"Training dataset not found at {path}")
        df = pd.read_csv(path)
        return cls(df)

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.30,
        intent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Searches historical resolutions for the most similar precedents.

        Args:
            query: Customer inquiry text.
            top_k: Maximum number of precedents to return.
            min_similarity: Minimum cosine similarity threshold (e.g. 0.30).
            intent: Optional intent category to restrict candidates to.

        Returns:
            List of retrieved precedent dictionaries sorted by similarity descending.
        """
        if self.matrix is None or len(self.examples) == 0 or not query or not query.strip() or top_k <= 0:
            return []

        # Preprocess and vectorize query
        clean_query = normalize_text(redact_pii(query))
        if not clean_query:
            return []

        query_vec = self.vectorizer.transform([clean_query])
        # Dense cosine similarity scores (TF-IDF vectors are unit-normalized by default in sklearn)
        scores = (self.matrix @ query_vec.T).toarray().ravel()

        results: List[Dict[str, Any]] = []
        # Sort candidate indices by descending similarity
        ranked_indices = scores.argsort()[::-1]

        for idx in ranked_indices:
            score = float(scores[idx])
            # Strict threshold cutoff: stop if below minimum similarity
            if score <= 0.0 or score < min_similarity:
                break

            row = self.examples.iloc[idx]

            # Intent gating: restrict to matching intent if requested
            if intent is not None and "intent" in row and row["intent"] != intent:
                continue

            example_id = str(row.get("example_id", ""))
            cust_msg = sanitize_evidence(str(row.get("customer_message", "")))
            brand_reply = sanitize_evidence(str(row.get("historical_brand_reply", "")))
            conv_ctx = sanitize_evidence(str(row.get("conversation_context", "")))
            row_intent = str(row.get("intent", ""))

            results.append({
                "example_id": example_id,
                "customer_message": cust_msg,
                "brand_reply": brand_reply,
                "intent": row_intent,
                "conversation_context": conv_ctx,
                "similarity_score": round(score, 4),
            })

            if len(results) >= top_k:
                break

        return results

    def __len__(self) -> int:
        return len(self.examples)
