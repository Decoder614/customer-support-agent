"""Flask Web Application for Uber AI Support Agent Live Demo.

Provides an interactive user interface and REST API endpoints for real-time
intent classification, precedent retrieval, escalation decisioning, and response drafting.
"""

from threading import Lock
from typing import Optional
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from src.agent.support_agent import SupportAgent, load_agent

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024  # 128 KB max request body

_agent: Optional[SupportAgent] = None
_agent_lock = Lock()


def get_agent(system_tier: str = "main") -> SupportAgent:
    """Lazily loads and caches the SupportAgent instance thread-safely."""
    global _agent
    with _agent_lock:
        if _agent is None:
            _agent = load_agent(system_tier=system_tier)
    return _agent


@app.get("/")
def index():
    """Renders the interactive web demo dashboard."""
    return render_template("index.html")


@app.get("/health")
def health():
    """Health check endpoint confirming web server and model availability."""
    return jsonify({
        "status": "ok",
        "model_loaded": _agent is not None,
    })


@app.post("/api/predict")
@app.post("/api/respond")
def predict():
    """API endpoint to process an inbound customer message through the AI pipeline."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json."}), 415

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a valid JSON object."}), 400

    message = payload.get("message")
    context = payload.get("context", "")
    system_tier = payload.get("system", "main")

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "'message' field must be a non-empty string."}), 400

    if not isinstance(context, str):
        return jsonify({"error": "'context' field must be a string if provided."}), 400

    if len(message) > 10000 or len(context) > 20000:
        return jsonify({"error": "Input length exceeds character limits (max 10000 for message, 20000 for context)."}), 400

    agent = get_agent(system_tier=system_tier)
    result = agent.process(message=message.strip(), context=context.strip())

    response = jsonify(result)
    response.headers["Cache-Control"] = "no-store"
    return response, 200


@app.errorhandler(HTTPException)
def handle_http_exception(error):
    return jsonify({"error": error.description}), error.code


@app.errorhandler(Exception)
def handle_unexpected_exception(error):
    if not app.config.get("TESTING"):
        app.logger.error(f"Support demo request encountered unexpected error: {type(error).__name__}: {error}")
    else:
        app.logger.debug(f"Handling expected test exception: {type(error).__name__}: {error}")
    return jsonify({"error": "The support agent service encountered an unexpected error. Please try again."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
