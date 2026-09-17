"""Domain-specific intent taxonomy for Uber Support customer interactions.

Defines the 9 canonical intent categories, priority resolution order,
keyword trigger patterns, risk classifications, and metadata.
"""

from enum import Enum
import re
from typing import Dict, List, Optional, Tuple, TypedDict


class Intent(str, Enum):
    """Canonical 9-intent taxonomy for Uber Support."""
    SAFETY_AND_CONDUCT = "safety_and_conduct"
    CANCELLATION_FEE = "cancellation_fee"
    FARE_DISPUTE_OVERCHARGE = "fare_dispute_overcharge"
    LOST_AND_FOUND = "lost_and_found"
    PICKUP_ROUTING_ISSUE = "pickup_routing_issue"
    APP_AND_ACCOUNT_ACCESS = "app_and_account_access"
    PROMOTIONS_AND_UBEREATS = "promotions_and_ubereats"
    DRIVER_PARTNER_INQUIRY = "driver_partner_inquiry"
    OTHER_GENERAL_FEEDBACK = "other_general_feedback"


class IntentMetadata(TypedDict):
    """Metadata schema for intent taxonomy entries."""
    name: str
    description: str
    routing_action: str
    is_high_risk: bool
    example_utterance: str


# Regex pattern triggers for empirical keyword bootstrapping and heuristic routing
PATTERNS: Dict[str, str] = {
    Intent.SAFETY_AND_CONDUCT.value: (
        r"\b(safety|accident|crash|police|harass\w*|assault\w*|threat\w*|"
        r"drunk|intoxicated|weapon|scary|dangerous\w*|creepy|inappropriate|"
        r"attack\w*|yell\w*|scream\w*|stolen car|hit and run|reckless|"
        r"unprofessional|rude driver)\b"
    ),
    Intent.CANCELLATION_FEE.value: (
        r"\b(cancel\w* fee|cancellation fee|charged for cancel\w*|"
        r"charged when cancel\w*|driver cancel\w*|driver didn.?t show|"
        r"no show fee|cancel\w* charge)\b"
    ),
    Intent.FARE_DISPUTE_OVERCHARGE.value: (
        r"\b(overcharg\w*|extra charg\w*|wrong amount|fare|toll\w*|"
        r"refund\w*|charged twice|double charg\w*|upfront (fare|price)|"
        r"price surge|excessive charg\w*|receipt|cost too much|meter|"
        r"trip charge)\b"
    ),
    Intent.LOST_AND_FOUND.value: (
        r"\b(lost|left (my|a|the)|forgot (my|a|the)|wallet|phone|keys?|"
        r"bag|purse|backpack|jacket|glasses|sunglasses|item in (the|car|cab|vehicle)|"
        r"dropped my|retrieve my)\b"
    ),
    Intent.PICKUP_ROUTING_ISSUE.value: (
        r"\b(pickup|pick up|route|wrong (direction|way|address|location|pin)|"
        r"driver went|stranded|waiting for driver|refused to pick|no driver|"
        r"eta|gps|drop off location)\b"
    ),
    Intent.APP_AND_ACCOUNT_ACCESS.value: (
        r"\b(account|log ?in|sign ?in|password|otp|verification code|"
        r"locked out|update phone|change number|hacked|"
        r"app (crash\w*|freeze\w*|glitch\w*|bug\w*|update)|"
        r"payment method|credit card update)\b"
    ),
    Intent.PROMOTIONS_AND_UBEREATS.value: (
        r"\b(promo\w*|discount\w*|coupon\w*|uber ?cash|credits?|"
        r"ubereats|uber ?eats|food|delivery|order|restaurant)\b"
    ),
    Intent.DRIVER_PARTNER_INQUIRY.value: (
        r"\b(driver partner|driving for uber|payout\w*|earnings?|"
        r"background check|vehicle inspection|upload document|"
        r"driver account|become a driver|driver app)\b"
    ),
}

INTENTS: List[str] = [intent.value for intent in Intent]

# High-risk intents requiring mandatory human escalation by policy
HIGH_RISK_INTENTS: List[str] = [
    Intent.SAFETY_AND_CONDUCT.value,
    Intent.FARE_DISPUTE_OVERCHARGE.value,
    Intent.CANCELLATION_FEE.value,
    Intent.DRIVER_PARTNER_INQUIRY.value,
]

# Resolution priority: Safety & financial disputes take precedence over operational categories
PRIORITY: List[str] = [
    Intent.SAFETY_AND_CONDUCT.value,
    Intent.CANCELLATION_FEE.value,
    Intent.FARE_DISPUTE_OVERCHARGE.value,
    Intent.LOST_AND_FOUND.value,
    Intent.DRIVER_PARTNER_INQUIRY.value,
    Intent.APP_AND_ACCOUNT_ACCESS.value,
    Intent.PICKUP_ROUTING_ISSUE.value,
    Intent.PROMOTIONS_AND_UBEREATS.value,
]

# Detailed metadata per intent category
TAXONOMY_METADATA: Dict[str, IntentMetadata] = {
    Intent.SAFETY_AND_CONDUCT.value: {
        "name": "Safety and Driver Conduct",
        "description": "Physical safety, reckless driving, verbal abuse, harassment, accidents, emergency incidents.",
        "routing_action": "ESCALATE (Safety Ops Team)",
        "is_high_risk": True,
        "example_utterance": "Driver was speeding and almost crashed into a guardrail.",
    },
    Intent.CANCELLATION_FEE.value: {
        "name": "Cancellation Fee Dispute",
        "description": "Disputes over cancellation charges, driver no-shows, driver-forced cancellations.",
        "routing_action": "ESCALATE (Financial / Trip Review)",
        "is_high_risk": True,
        "example_utterance": "Driver cancelled on me after 15 mins and I was charged $5.",
    },
    Intent.FARE_DISPUTE_OVERCHARGE.value: {
        "name": "Fare Dispute and Overcharge",
        "description": "Unexpected tolls, wrong route charges, upfront pricing discrepancies, double billing.",
        "routing_action": "ESCALATE (Fare Adjustment / Refund)",
        "is_high_risk": True,
        "example_utterance": "I was charged $45 for a ride that quoted $18 upfront.",
    },
    Intent.LOST_AND_FOUND.value: {
        "name": "Lost and Found",
        "description": "Inquiries about items left behind in a vehicle (phone, keys, wallet, jacket).",
        "routing_action": "AUTO_HANDLE (Self-serve Driver Contact Guide)",
        "is_high_risk": False,
        "example_utterance": "I left my glasses in the back seat of my Uber yesterday.",
    },
    Intent.PICKUP_ROUTING_ISSUE.value: {
        "name": "Pickup and Routing Issue",
        "description": "Driver arriving at incorrect pin, navigation errors, refusal to pick up, ETA delays.",
        "routing_action": "CONDITIONAL_AUTO_HANDLE (Live ETA & Re-dispatch / Escalate)",
        "is_high_risk": False,
        "example_utterance": "Driver went to the wrong terminal at JFK and left me stranded.",
    },
    Intent.APP_AND_ACCOUNT_ACCESS.value: {
        "name": "App and Account Access",
        "description": "Login issues, 2FA/OTP failures, app crashes, updating phone number or email.",
        "routing_action": "CONDITIONAL_AUTO_HANDLE (Troubleshooting Steps / Auth Review)",
        "is_high_risk": False,
        "example_utterance": "App keeps crashing on the payment screen on iOS 17.",
    },
    Intent.PROMOTIONS_AND_UBEREATS.value: {
        "name": "Promotions and Uber Eats",
        "description": "Promo codes not applying, Uber Cash issues, UberEATS delivery delays or missing items.",
        "routing_action": "CONDITIONAL_AUTO_HANDLE (Promo Guidelines / Order Resolution)",
        "is_high_risk": False,
        "example_utterance": "My 20% promo code didn't apply to my last trip.",
    },
    Intent.DRIVER_PARTNER_INQUIRY.value: {
        "name": "Driver Partner Inquiry",
        "description": "Driver-specific issues: payouts, document uploads, vehicle inspection, background checks.",
        "routing_action": "ESCALATE (Driver Partner Operations)",
        "is_high_risk": True,
        "example_utterance": "When will my weekly direct deposit payout process?",
    },
    Intent.OTHER_GENERAL_FEEDBACK.value: {
        "name": "Other / General Feedback",
        "description": "General compliments, broad service feedback, non-actionable tweets, greetings.",
        "routing_action": "CONDITIONAL_AUTO_HANDLE / ESCALATE (Standard Acknowledgment)",
        "is_high_risk": False,
        "example_utterance": "Thanks to my driver John for the smooth ride today!",
    },
}


def propose(message: str) -> Tuple[str, bool]:
    """Matches keyword rules against message text.

    Args:
        message: Raw or normalized user inquiry text.

    Returns:
        Tuple of (predicted_intent, is_ambiguous).
        is_ambiguous is True if 0 or >=2 intent pattern triggers match.
    """
    if not message or not isinstance(message, str):
        return Intent.OTHER_GENERAL_FEEDBACK.value, True

    matches = [
        intent
        for intent in PRIORITY
        if re.search(PATTERNS[intent], message, re.IGNORECASE)
    ]

    if not matches:
        return Intent.OTHER_GENERAL_FEEDBACK.value, True

    return matches[0], len(matches) != 1


def is_high_risk(intent: str) -> bool:
    """Returns True if the intent is marked high-risk requiring human escalation."""
    return intent in HIGH_RISK_INTENTS
