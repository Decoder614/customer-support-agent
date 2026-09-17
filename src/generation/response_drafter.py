"""Template-Grounded Response Drafter for Uber Support Agent.

Generates conservative, policy-compliant customer support replies based on
retrieved historical resolutions, domain intent categories, and escalation policies
without hallucinating financial commitments, trip status, or private account facts.
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional


ESCALATION_SAFE_NOTICE = (
    "Thanks for reaching out. A specialized customer support agent will review your "
    "trip details and account history. Please check the 'Help' section in your Uber app "
    "or monitor your registered email for updates."
)

SAFE_FALLBACK = (
    "Thanks for reaching out to Uber Support. We are here to help. "
    "Please visit the 'Help' section in your Uber app or reach out via official support channels."
)


def is_unsafe_draft(reply: str) -> bool:
    """Detects policy violations, hallucinated refunds, or unauthorized credential requests."""
    if not reply or not isinstance(reply, str):
        return True

    # Patterns for hallucinated financial approvals, refund commitments, or sensitive credential leaks
    unsafe_patterns = [
        r"\b(refund(ed| approved)|we have (refunded|credited|issued a refund))\b",
        r"\b(we.?ve (issued|processed|adjusted your fare)|your (fare|charge) has been refunded)\b",
        r"\b(password|pin code|card number|cvv|security code)\b",
        r"https?://",  # Unchecked external URLs
    ]

    for pat in unsafe_patterns:
        if re.search(pat, reply, re.IGNORECASE):
            return True

    return False


def get_grounded_resolution_step(intent: str, evidence: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    """Derives grounded self-service troubleshooting instructions based on intent and evidence.

    Args:
        intent: Classified intent category.
        evidence: Retrieved historical resolution precedents.

    Returns:
        Grounded guidance string, or None if no specific self-service template applies.
    """
    evidence_text = " ".join(e.get("brand_reply", "").lower() for e in (evidence or []))

    if intent == "lost_and_found" or re.search(r"\b(lost|left|item|wallet|phone|keys?|bag)\b", evidence_text):
        return (
            "If you left an item behind in a vehicle, you can contact your driver directly through "
            "'Your Trips' > select your trip > 'Find lost item' in the Uber app to arrange retrieval."
        )

    if intent == "app_and_account_access":
        if re.search(r"\b(restart|reboot|force close)\b", evidence_text):
            return "Could you try force-closing and restarting the Uber app to see if that resolves the issue?"
        if re.search(r"\b(reinstall|update|latest version)\b", evidence_text):
            return "Please verify that you have the latest version of the Uber app installed from your app store, or try reinstalling the app."
        return (
            "For account access issues, please check your network connection, ensure your app is updated "
            "to the latest version, or request a new verification code in the login screen."
        )

    if intent == "promotions_and_ubereats":
        return (
            "For promotion and coupon inquiries, please check the 'Promotions' tab under 'Payment' in your Uber app "
            "to confirm valid dates and qualifying ride or order conditions."
        )

    if intent == "pickup_routing_issue":
        return (
            "If your driver is having trouble locating your pickup spot, you can use the in-app call or message "
            "feature to share your exact terminal, door number, or landmark pin."
        )

    return None


@dataclass(frozen=True)
class DraftResponse:
    """Represents the drafted response metadata."""
    reply_text: str
    mode: str
    requires_human_review: bool
    policy_compliant: bool


class ResponseDrafter:
    """Policy-compliant response generation engine for Uber Support interactions."""

    def __init__(self, escalation_notice: str = ESCALATION_SAFE_NOTICE) -> None:
        self.escalation_notice = escalation_notice

    def draft(
        self,
        intent: str,
        action: str,
        evidence: Optional[List[Dict[str, Any]]] = None,
        message: str = "",
        context: str = "",
    ) -> DraftResponse:
        """Generates a grounded, policy-compliant response draft.

        Args:
            intent: Classified domain intent.
            action: Policy action ('AUTO_HANDLE' or 'ESCALATE').
            evidence: Historical precedents from RetrievalIndex.
            message: Customer message.
            context: Conversation context.

        Returns:
            DraftResponse with safe reply text and policy status.
        """
        # If policy mandates human escalation
        if action == "ESCALATE":
            return DraftResponse(
                reply_text=self.escalation_notice,
                mode="deterministic_escalation",
                requires_human_review=True,
                policy_compliant=True,
            )

        # For AUTO_HANDLE, extract grounded template steps
        grounded_step = get_grounded_resolution_step(intent=intent, evidence=evidence)

        if grounded_step:
            reply = f"Hi! Thanks for reaching out. {grounded_step}"
            if not is_unsafe_draft(reply):
                return DraftResponse(
                    reply_text=reply,
                    mode="deterministic_grounded",
                    requires_human_review=False,
                    policy_compliant=True,
                )

        # Fallback if grounded step fails safety checks
        return DraftResponse(
            reply_text=SAFE_FALLBACK,
            mode="safe_fallback",
            requires_human_review=True,
            policy_compliant=True,
        )


def draft_response(
    intent: str,
    action: str,
    evidence: Optional[List[Dict[str, Any]]] = None,
    message: str = "",
    context: str = "",
) -> Dict[str, Any]:
    """Functional convenience wrapper returning a dictionary response payload."""
    drafter = ResponseDrafter()
    result = drafter.draft(
        intent=intent,
        action=action,
        evidence=evidence,
        message=message,
        context=context,
    )
    return {
        "reply_text": result.reply_text,
        "mode": result.mode,
        "requires_human_review": result.requires_human_review,
        "policy_compliant": result.policy_compliant,
    }
