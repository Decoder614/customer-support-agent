"""PII redaction and text normalization utilities for Twitter support interactions."""
import html
import re
from typing import Optional


def redact_pii(text: Optional[str]) -> str:
    """
    Redact sensitive personal identifiable information (PII) from customer support text.
    
    Replaces:
    - URLs with [URL] (or [LINK])
    - Email addresses with [EMAIL]
    - Phone numbers / digit sequences with [PHONE]
    - Twitter @ handles with [USER]
    - Unescapes HTML entities and normalizes whitespace.
    """
    if text is None:
        return ""
    
    # 1. Unescape HTML entities (e.g., &amp; -> &)
    cleaned = html.unescape(str(text))
    
    # 2. Redact URLs
    cleaned = re.sub(r'https?://\S+|www\.\S+', '[URL]', cleaned, flags=re.IGNORECASE)
    
    # 3. Redact Email addresses
    cleaned = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL]', cleaned)
    
    # 4. Redact Twitter @ handles
    cleaned = re.sub(r'@[\w_]+', '[USER]', cleaned)
    
    # 5. Redact Phone numbers and long account/card digit sequences (7+ digits with common separators)
    cleaned = re.sub(r'(?<!\w)(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}(?!\w)', '[PHONE]', cleaned)
    cleaned = re.sub(r'(?<!\w)\+?\d[\d ()-]{6,}\d(?!\w)', '[PHONE]', cleaned)
    
    # 6. Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def normalize_text(text: Optional[str]) -> str:
    """
    Normalize text for downstream classification and retrieval.
    
    Applies PII redaction, converts to lowercase, strips leftover placeholder brackets/tokens,
    and removes redundant whitespace.
    """
    if text is None:
        return ""
    
    redacted = redact_pii(text).lower()
    
    # Remove redaction token brackets to create clean tokenized text for TF-IDF / embeddings
    normalized = re.sub(r'\[(url|link|email|user|phone|number)\]', ' ', redacted)
    
    # Collapse multiple whitespace characters into single space
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


# Aliases for compatibility
redact = redact_pii
normalize = normalize_text
