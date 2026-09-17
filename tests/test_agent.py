"""Tests for end-to-end SupportAgent pipeline runner."""

import json
from pathlib import Path
import pytest

from src.agent.support_agent import SupportAgent, load_agent
from src.intents.taxonomy import Intent


def test_agent_lost_and_found_auto_handle():
    """Verify end-to-end pipeline auto-handles lost and found inquiry with grounded response."""
    agent = load_agent(system_tier="main")
    res = agent.process("I left my wallet in the car")

    assert res["intent"] == Intent.LOST_AND_FOUND.value
    assert res["intent_confidence"] > 0.65
    assert res["action"] == "AUTO_HANDLE"
    assert res["requires_human_review"] is False
    assert "Find lost item" in res["reply"]
    assert len(res["retrieved_examples"]) >= 1


def test_agent_safety_escalation():
    """Verify end-to-end pipeline escalates safety incidents immediately."""
    agent = load_agent(system_tier="main")
    res = agent.process("The driver was drunk, pulled out a weapon, and crashed into a pole")

    assert res["intent"] == Intent.SAFETY_AND_CONDUCT.value
    assert res["action"] == "ESCALATE"
    assert res["escalation_tier"] == "SAFETY_EMERGENCY"
    assert res["requires_human_review"] is True


def test_agent_fare_dispute_escalation():
    """Verify end-to-end pipeline escalates financial disputes."""
    agent = load_agent(system_tier="main")
    res = agent.process("I was charged $45 for a ride that quoted $15 upfront. Please refund.")

    assert res["intent"] == Intent.FARE_DISPUTE_OVERCHARGE.value
    assert res["action"] == "ESCALATE"
    assert res["escalation_tier"] == "HIGH_RISK_INTENT"
    assert res["requires_human_review"] is True


def test_agent_trivial_baseline_mode():
    """Verify trivial baseline system escalates all requests with baseline message."""
    agent = load_agent(system_tier="trivial")
    res = agent.process("I lost my keys")

    assert res["action"] == "ESCALATE"
    assert res["generation_mode"] == "generic_baseline"
    assert res["retrieved_examples"] == []


def test_agent_simple_baseline_mode():
    """Verify simple baseline uses verbatim retrieval held for review."""
    agent = load_agent(system_tier="simple")
    res = agent.process("I left my wallet in the car")

    assert res["action"] == "ESCALATE" or res["requires_human_review"] is True
    assert res["generation_mode"] == "verbatim_retrieval_baseline"
    assert len(res["retrieved_examples"]) > 0


def test_agent_input_validation():
    """Verify error handling on invalid types or excessive length."""
    agent = load_agent(system_tier="main")

    with pytest.raises(TypeError):
        agent.process(None)  # type: ignore

    with pytest.raises(ValueError):
        agent.process("x" * 20000)
