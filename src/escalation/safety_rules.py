"""Deterministic Safety Keyword Gatekeeper for Uber Support Agent.

Intercepts safety-critical incidents (physical accidents, police/emergency response,
harassment, assault, intoxication, weapons, legal threats) and forces immediate human
escalation with explicit safety reasons prior to model inference.
"""

from dataclasses import dataclass
import re
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class SafetyDecision:
    """Represents the output decision of the deterministic safety gatekeeper."""
    is_escalated: bool
    action: str
    risk_category: Optional[str] = None
    reason: Optional[str] = None
    matched_keyword: Optional[str] = None


# Prioritized list of safety trigger regular expressions with explicit operational escalation rationales
SAFETY_PATTERNS: List[Tuple[str, str, str]] = [
    # 1. Physical Safety, Collisions & Medical Emergencies
    (
        r"\b(accident|crash|crashed|collision|ambulance|injur\w*|hospital|medical emergency|bleeding|hit and run|pedestrian hit|run over)\b",
        "Physical Safety & Collision",
        "Critical physical safety, vehicular collision, or medical emergency requires immediate Safety Response Team escalation.",
    ),
    # 2. Weapons, Police & Law Enforcement
    (
        r"\b(gun|weapon|knife|armed|police|911|cop|cops|arrest\w*|lawsuit|sue|legal action|lawyer|attorney)\b",
        "Police & Legal Emergency",
        "Report of weapons, police intervention, or formal legal emergency requires immediate human escalation.",
    ),
    # 3. Assault, Threat, Harassment & Inappropriate Conduct
    (
        r"\b(assault\w*|threat\w*|harass\w*|attack\w*|stalk\w*|kidnap\w*|touching|inappropriate|sexual\w*|creepy|yell\w*|scream\w*)\b",
        "Threat & Harassment",
        "Severe driver misconduct, harassment, assault, or personal threat requires immediate human safety investigation.",
    ),
    # 4. Intoxication & Impairment
    (
        r"\b(drunk|intoxicated|passed out|unconscious|drugs|alcohol|vomit\w*)\b",
        "Intoxication & Impairment",
        "Reported driver or passenger intoxication/impairment requires immediate human safety review.",
    ),
    # 5. Reckless Driving & Extreme Danger
    (
        r"\b(reckless\w*|wrong way|wrong side of the road|ran red light|speeding dangerously|scary driver|dangerous driver)\b",
        "Reckless Driving",
        "Reported dangerous or reckless vehicle operation requires human safety ops review.",
    ),
    # 6. Abusive or Hostile Communication
    (
        r"\b(fuck\w*|shit\w*|bastard\w*|bitch\w*|asshole\w*)\b",
        "Hostility & Abuse",
        "Hostile or abusive content requires human moderation and specialized de-escalation.",
    ),
]


def check_safety_risk(text: str) -> Optional[Tuple[str, str, str]]:
    """Checks text against deterministic safety risk regex patterns.

    Args:
        text: Customer message and/or conversation context string.

    Returns:
        Tuple of (risk_category, escalation_reason, matched_keyword) if triggered, else None.
    """
    if not text or not isinstance(text, str):
        return None

    for pattern, category, reason in SAFETY_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return category, reason, match.group(0)

    return None


def evaluate_safety_gate(message: str, context: str = "") -> SafetyDecision:
    """Evaluates customer message and antecedent context against safety risk gatekeeper.

    Args:
        message: Current customer utterance.
        context: Prior conversation turns (if multi-turn thread).

    Returns:
        SafetyDecision with action 'ESCALATE' and detailed reason if safety risk found,
        otherwise action 'PASS'.
    """
    combined_text = f"{message} {context}".strip()
    result = check_safety_risk(combined_text)

    if result is not None:
        category, reason, matched_kw = result
        return SafetyDecision(
            is_escalated=True,
            action="ESCALATE",
            risk_category=category,
            reason=reason,
            matched_keyword=matched_kw,
        )

    return SafetyDecision(
        is_escalated=False,
        action="PASS",
        risk_category=None,
        reason=None,
        matched_keyword=None,
    )
