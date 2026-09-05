from fastapi.testclient import TestClient

from aether_scientist.api.endpoints import app
from aether_scientist.core.inference import InferenceEngine


def test_inference_engine_stream(monkeypatch):
    InferenceEngine.reset()
    monkeypatch.setattr(
        InferenceEngine,
        "generate",
        lambda **kwargs: {"text": "Entropy increases over time continuously."},
    )

    tokens = list(InferenceEngine.stream("What is entropy?"))
    assert len(tokens) >= 3
    full = "".join(tokens)
    assert "Entropy" in full


def test_api_analyze_stream_events(monkeypatch):
    monkeypatch.setattr(
        InferenceEngine,
        "stream",
        lambda *args, **kwargs: iter(["Entropy ", "is ", "disorder."]),
    )

    client = TestClient(app)
    headers = {"X-API-Key": "secret-aether-key"}
    response = client.post(
        "/analyze/stream",
        json={"query": "Explain entropy"},
        headers=headers,
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    body = response.text
    assert "event: token" in body
    assert "event: sources" in body
    assert "event: done" in body
    assert "Entropy" in body
