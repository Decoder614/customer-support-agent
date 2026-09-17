"""Unit and integration tests for Flask web application and API endpoints."""

import pytest
from unittest.mock import patch
from app import app, get_agent, _agent


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """Test that the index dashboard loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Uber AI Customer Support Agent" in response.data
    assert b"Hiver SDE Intern" in response.data


def test_health_route(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_endpoint_valid_auto_handle(client):
    """Test API prediction with an auto-handle candidate (Lost Item)."""
    payload = {
        "message": "I forgot my black backpack and jacket in the back seat of the car.",
        "context": "",
        "system": "main"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()

    assert "intent" in data
    assert data["intent"] == "lost_and_found"
    assert "intent_confidence" in data
    assert isinstance(data["intent_confidence"], float)
    assert data["action"] in ["AUTO_HANDLE", "ESCALATE"]
    assert "reply" in data
    assert len(data["reply"]) > 0
    assert "retrieved_examples" in data
    assert isinstance(data["retrieved_examples"], list)
    assert response.headers.get("Cache-Control") == "no-store"


def test_predict_endpoint_valid_escalation(client):
    """Test API prediction with a safety-critical message forcing escalation."""
    payload = {
        "message": "Driver crashed into another vehicle on the freeway and I am injured.",
        "context": "",
        "system": "main"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()

    assert data["action"] == "ESCALATE"
    assert "escalation_reason" in data
    assert "accident" in data["escalation_reason"].lower() or "safety" in data["escalation_reason"].lower() or "emergency" in data["escalation_reason"].lower() or len(data["escalation_reason"]) > 0


def test_respond_alias_endpoint(client):
    """Test that /api/respond alias works identically to /api/predict."""
    payload = {
        "message": "The Uber app keeps crashing and giving glitch errors when I try to log in.",
        "context": ""
    }
    response = client.post("/api/respond", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["intent"] == "app_and_account_access"


@pytest.mark.parametrize("invalid_payload,expected_status", [
    ({}, 400),
    ({"message": ""}, 400),
    ({"message": "   "}, 400),
    ({"message": 12345}, 400),
    ({"message": None}, 400),
    ({"message": "Valid", "context": 123}, 400),
    ({"message": "a" * 10001}, 400),
])
def test_predict_validation_errors(client, invalid_payload, expected_status):
    """Test invalid payloads return 400 Bad Request."""
    response = client.post("/api/predict", json=invalid_payload)
    assert response.status_code == expected_status
    data = response.get_json()
    assert "error" in data


def test_predict_non_json_content_type(client):
    """Test non-JSON content type returns 415 Unsupported Media Type."""
    response = client.post("/api/predict", data="message=hello", content_type="application/x-www-form-urlencoded")
    assert response.status_code == 415
    data = response.get_json()
    assert "error" in data


def test_predict_trivial_system_tier(client):
    """Test prediction using the trivial baseline tier."""
    payload = {
        "message": "Where is my driver?",
        "system": "trivial"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["action"] == "ESCALATE"


def test_internal_error_handling(client):
    """Test that unexpected server exceptions return 500 without leaking private internals."""
    with patch("app.get_agent") as mock_agent_getter:
        mock_agent_getter.side_effect = RuntimeError("Private database trace that should not leak")
        response = client.post("/api/predict", json={"message": "Help me with my ride"})
        assert response.status_code == 500
        data = response.get_json()
        assert "error" in data
        assert "Private database trace" not in data["error"]
