"""End-to-End SupportAgent Pipeline Runner for Uber Support interactions.

Connects Preprocessing -> Classification -> Retrieval -> Escalation Policy -> Response Drafting
into a unified, fully reproducible decision and response pipeline.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import pandas as pd
import yaml

from src.data.preprocessing import redact_pii, normalize_text
from src.intents.taxonomy import Intent, HIGH_RISK_INTENTS
from src.intents.baseline_classifiers import TrivialMajorityClassifier, SimpleTfidfClassifier
from src.intents.classifier import IntentClassifier
from src.retrieval.retrieval_index import RetrievalIndex
from src.escalation.policy_engine import PolicyDecision, decide, load_escalation_config
from src.generation.response_drafter import ResponseDrafter, DraftResponse


class SupportAgent:
    """Unified AI Support Agent combining classification, retrieval, escalation, and response drafting."""

    def __init__(
        self,
        classifier: Any,
        retrieval_index: RetrievalIndex,
        settings: Optional[Dict[str, Any]] = None,
        system_tier: str = "main",
        drafter: Optional[ResponseDrafter] = None,
    ) -> None:
        self.classifier = classifier
        self.retrieval_index = retrieval_index
        self.settings = settings or load_escalation_config()
        self.system_tier = system_tier
        self.drafter = drafter or ResponseDrafter()

    def process(self, message: str, context: str = "") -> Dict[str, Any]:
        """Processes a customer inquiry through the full end-to-end pipeline.

        Args:
            message: Incoming customer inquiry message.
            context: Antecedent multi-turn context thread.

        Returns:
            Dictionary containing predicted intent, confidence score, policy decision,
            escalation rationale, retrieved resolution precedents, and drafted reply text.
        """
        if not isinstance(message, str):
            raise TypeError("message must be a string")
        if not isinstance(context, str):
            raise TypeError("context must be a string")

        if len(message) > 15000 or len(context) > 30000:
            raise ValueError("Input string exceeds maximum allowable character limit.")

        clean_message = normalize_text(redact_pii(message))

        # 1. Intent Classification & Confidence Scoring
        if hasattr(self.classifier, "predict_with_confidence"):
            labels, confs = self.classifier.predict_with_confidence([message])
            predicted_intent = labels[0]
            confidence = float(confs[0])
        elif hasattr(self.classifier, "predict"):
            predicted_intent = self.classifier.predict([message])[0]
            confidence = 1.0
        else:
            predicted_intent = Intent.OTHER_GENERAL_FEEDBACK.value
            confidence = 0.0

        # Trivial baseline fast path
        if self.system_tier == "trivial":
            return {
                "intent": predicted_intent,
                "intent_confidence": confidence,
                "action": "ESCALATE",
                "escalation_reason": "The trivial baseline escalates every customer inquiry.",
                "escalation_tier": "TRIVIAL_BASELINE",
                "reply": "Thanks for reaching out. A support specialist will review your request.",
                "generation_mode": "generic_baseline",
                "retrieved_examples": [],
                "requires_human_review": True,
            }

        # 2. Historical Precedent Retrieval
        top_k = int(self.settings.get("top_k", 3))
        min_sim = float(self.settings.get("retrieval_grounding_threshold", 0.30))
        intent_filter = predicted_intent if self.system_tier == "main" else None

        evidence = self.retrieval_index.search(
            query=message,
            top_k=top_k,
            min_similarity=min_sim,
            intent=intent_filter,
        )

        # 3. Multi-Factor Escalation Policy Evaluation
        policy_decision = decide(
            message=message,
            context=context,
            intent=predicted_intent,
            confidence=confidence,
            evidence=evidence,
            settings=self.settings,
        )

        # 4. Response Drafting
        if self.system_tier == "simple" and evidence:
            # Simple baseline uses nearest historical brand reply verbatim held for review
            draft_reply = evidence[0]["brand_reply"]
            draft_mode = "verbatim_retrieval_baseline"
            requires_review = True
        else:
            draft_result = self.drafter.draft(
                intent=predicted_intent,
                action=policy_decision.action,
                evidence=evidence,
                message=message,
                context=context,
            )
            draft_reply = draft_result.reply_text
            draft_mode = draft_result.mode
            requires_review = draft_result.requires_human_review

        # Final action override if draft requires review
        final_action = policy_decision.action
        final_reason = policy_decision.reason
        if requires_review and final_action == "AUTO_HANDLE":
            final_action = "ESCALATE"
            final_reason = "Generated resolution draft requires human verification before sending."

        return {
            "intent": predicted_intent,
            "intent_confidence": round(confidence, 4),
            "action": final_action,
            "escalation_reason": final_reason,
            "escalation_tier": policy_decision.escalation_tier,
            "reply": draft_reply,
            "generation_mode": draft_mode,
            "retrieved_examples": evidence,
            "requires_human_review": requires_review,
        }


def load_agent(
    system_tier: str = "main",
    models_dir: Union[str, Path] = "artifacts/models",
    train_path: Union[str, Path] = "data/processed/train.csv",
    config_path: Union[str, Path] = "config/default.yaml",
) -> SupportAgent:
    """Factory loader that restores or initializes classifier models and the retrieval index."""
    models_dir = Path(models_dir)
    model_file = models_dir / f"{system_tier}.joblib"

    train_path = Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found at {train_path}. Run data preparation first.")

    train_df = pd.read_csv(train_path, dtype=str, keep_default_na=False)

    # Load serialized classifier if present, else fit a new instance
    if model_file.exists():
        classifier = joblib.load(model_file)
    else:
        logging.info(f"Model file {model_file} not found; initializing and training new {system_tier} classifier...")
        if system_tier == "trivial":
            classifier = TrivialMajorityClassifier().fit(train_df["customer_message"], train_df["intent"])
        elif system_tier == "simple":
            classifier = SimpleTfidfClassifier().fit(train_df["customer_message"], train_df["intent"])
        else:
            classifier = IntentClassifier().fit(train_df["customer_message"], train_df["intent"])

    retrieval_index = RetrievalIndex(train_df)
    settings = load_escalation_config(config_path)

    return SupportAgent(
        classifier=classifier,
        retrieval_index=retrieval_index,
        settings=settings,
        system_tier=system_tier,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", required=True, help="Customer support message text")
    parser.add_argument("--context", default="", help="Antecedent conversation context thread")
    parser.add_argument("--system", choices=["main", "simple", "trivial"], default="main", help="System model tier")
    parser.add_argument("--config", default="config/default.yaml", help="Configuration YAML path")
    args = parser.parse_args()

    agent = load_agent(system_tier=args.system, config_path=args.config)
    result = agent.process(message=args.message, context=args.context)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
