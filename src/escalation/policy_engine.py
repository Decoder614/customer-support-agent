"""Multi-Tier Escalation Policy Engine for Uber Support Agent.

Evaluates safety gatekeeper, intent category risk policies, model classification confidence,
and retrieval grounding evidence to make deterministic, reasoned AUTO_HANDLE vs. ESCALATE decisions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from src.escalation.safety_rules import evaluate_safety_gate
from src.intents.taxonomy import Intent, HIGH_RISK_INTENTS, is_high_risk


@dataclass(frozen=True)
class PolicyDecision:
    """Represents the final policy decision for a customer support interaction."""
    action: str  # "AUTO_HANDLE" or "ESCALATE"
    reason: str  # Human-readable rationale
    escalation_tier: str  # Categorical reason code
    is_auto_handled: bool


def load_escalation_config(config_path: Union[str, Path] = "config/default.yaml") -> Dict[str, Any]:
    """Loads escalation thresholds and policy settings from configuration YAML."""
    path = Path(config_path)
    if not path.exists():
        return {
            "intent_confidence_threshold": 0.65,
            "retrieval_grounding_threshold": 0.30,
            "high_risk_intents": HIGH_RISK_INTENTS,
        }

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    esc_cfg = cfg.get("escalation", {})
    return {
        "intent_confidence_threshold": float(esc_cfg.get("intent_confidence_threshold", 0.65)),
        "retrieval_grounding_threshold": float(esc_cfg.get("retrieval_grounding_threshold", 0.30)),
        "high_risk_intents": esc_cfg.get("high_risk_intents", HIGH_RISK_INTENTS),
    }


def decide(
    message: str,
    context: str = "",
    intent: Optional[str] = None,
    confidence: float = 1.0,
    evidence: Optional[List[Dict[str, Any]]] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> PolicyDecision:
    """Executes multi-tier escalation evaluation with stated operational reasons.

    Evaluation Order:
    1. Deterministic Safety Gate: Immediate escalation on emergency, collision, threat, harassment, or impairment.
    2. Intent Domain Policy: High-risk financial/partner intents require specialist human verification.
    3. General Feedback / Unclear: Broad opinions or non-actionable chatter escalated for review.
    4. Model Confidence Gating: Inquiries with intent classification confidence < threshold (default: 0.65) escalated.
    5. Retrieval Evidence Gating: Inquiries lacking sufficiently similar historical resolutions (< threshold, default: 0.30) escalated.
    6. Auto-Handle: Low-risk, high-confidence, well-grounded inquiries receive automated resolution steps.

    Args:
        message: Incoming customer message text.
        context: Prior conversation turns (if multi-turn thread).
        intent: Predicted intent category name.
        confidence: Classification probability score (0.0 to 1.0).
        evidence: List of retrieved historical precedents from RetrievalIndex.
        settings: Optional dictionary overriding default threshold settings.

    Returns:
        PolicyDecision with action ('AUTO_HANDLE' or 'ESCALATE'), explicit reason, and tier.
    """
    if settings is None:
        settings = load_escalation_config()

    confidence_threshold = float(settings.get("intent_confidence_threshold", 0.65))
    retrieval_threshold = float(settings.get("retrieval_grounding_threshold", 0.30))
    high_risk_list = settings.get("high_risk_intents", HIGH_RISK_INTENTS)

    # Tier 1: Deterministic Safety Gatekeeper
    safety_result = evaluate_safety_gate(message=message, context=context)
    if safety_result.is_escalated:
        return PolicyDecision(
            action="ESCALATE",
            reason=safety_result.reason or "Critical physical safety, driver conduct, or legal investigation requires a human specialist.",
            escalation_tier="SAFETY_EMERGENCY",
            is_auto_handled=False,
        )

    # Tier 2: Intent Domain Risk Policy
    if intent in high_risk_list:
        if intent == Intent.SAFETY_AND_CONDUCT.value:
            reason = "Critical physical safety, driver conduct, or legal investigation requires a human specialist."
        elif intent == Intent.FARE_DISPUTE_OVERCHARGE.value:
            reason = "Fare adjustment, refund, or financial charge dispute requires account verification."
        elif intent == Intent.CANCELLATION_FEE.value:
            reason = "Cancellation fee review requires trip history verification by support."
        elif intent == Intent.DRIVER_PARTNER_INQUIRY.value:
            reason = "Driver partner inquiry requires driver operations review."
        else:
            reason = "This intent domain requires account-specific verification by support."

        return PolicyDecision(
            action="ESCALATE",
            reason=reason,
            escalation_tier="HIGH_RISK_INTENT",
            is_auto_handled=False,
        )

    # Tier 3: General Feedback / Non-actionable Unclear Intent
    if intent == Intent.OTHER_GENERAL_FEEDBACK.value or not intent:
        return PolicyDecision(
            action="ESCALATE",
            reason="Customer intent is general feedback or outside structured auto-handling.",
            escalation_tier="GENERAL_FEEDBACK",
            is_auto_handled=False,
        )

    # Tier 4: Classification Confidence Gating
    if confidence < confidence_threshold:
        return PolicyDecision(
            action="ESCALATE",
            reason=f"Intent classification confidence ({confidence:.2f}) is below safe automation threshold ({confidence_threshold:.2f}).",
            escalation_tier="CONFIDENCE_GATE",
            is_auto_handled=False,
        )

    # Tier 5: Retrieval Evidence Grounding Gating
    if not evidence:
        return PolicyDecision(
            action="ESCALATE",
            reason="No sufficiently grounded historical resolution evidence is available.",
            escalation_tier="RETRIEVAL_GATE",
            is_auto_handled=False,
        )

    max_sim = max(float(e.get("similarity_score", e.get("similarity", 0.0))) for e in evidence)
    if max_sim < retrieval_threshold:
        return PolicyDecision(
            action="ESCALATE",
            reason=f"Historical resolution similarity ({max_sim:.2f}) is below grounding threshold ({retrieval_threshold:.2f}).",
            escalation_tier="RETRIEVAL_GATE",
            is_auto_handled=False,
        )

    # Tier 6: Safe Automated Resolution
    return PolicyDecision(
        action="AUTO_HANDLE",
        reason="A low-risk self-service response is supported by verified historical resolution precedent.",
        escalation_tier="NONE",
        is_auto_handled=True,
    )
