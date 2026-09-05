from fastapi.testclient import TestClient

from aether_scientist.api.endpoints import app
from aether_scientist.core.inference import InferenceEngine


def test_get_profiles_endpoint():
    client = TestClient(app)
    response = client.get("/profiles")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    profs = data["data"]["profiles"]
    assert len(profs) == 4
    names = [p["name"] for p in profs]
    assert "offline" in names
    assert "fast" in names
    assert "balanced" in names
    assert "quality" in names


def test_analyze_valid_profile(monkeypatch):
    monkeypatch.setenv("AETHER_API_KEY", "test-key")
    monkeypatch.setattr(
        InferenceEngine,
        "generate",
        lambda **kwargs: {"text": "Thermodynamics is the study of heat and work."},
    )
    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}
    response = client.post(
        "/analyze",
        json={"query": "Explain thermodynamics", "profile": "fast"},
        headers=headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["profile"] == "fast"
    assert "device" in data["data"]


def test_analyze_invalid_profile(monkeypatch):
    monkeypatch.setenv("AETHER_API_KEY", "test-key")
    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}
    response = client.post(
        "/analyze",
        json={"query": "Explain thermodynamics", "profile": "non_existent_preset"},
        headers=headers,
    )
    assert response.status_code == 422
    assert "non_existent_preset" in response.text
    assert "offline" in response.text


def test_analyze_stream_profile_metadata(monkeypatch):
    InferenceEngine.reset()
    monkeypatch.setattr(
        InferenceEngine,
        "stream",
        lambda *args, **kwargs: iter(["Entropy ", "increases."]),
    )
    monkeypatch.setenv("AETHER_API_KEY", "test-key")
    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}
    response = client.post(
        "/analyze/stream",
        json={"query": "Explain entropy", "profile": "balanced"},
        headers=headers,
    )
    assert response.status_code == 200
    assert "balanced" in response.text
    assert "completed" in response.text


def test_analyze_stream_invalid_profile(monkeypatch):
    monkeypatch.setenv("AETHER_API_KEY", "test-key")
    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}
    response = client.post(
        "/analyze/stream",
        json={"query": "Explain entropy", "profile": "bad_model"},
        headers=headers,
    )
    assert response.status_code == 422
