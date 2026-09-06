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


def test_research_empty_index_returns_409(monkeypatch):
    monkeypatch.setenv("AETHER_API_KEY", "test-key")
    monkeypatch.setattr(
        "aether_scientist.retrieval.store.VectorStore.__len__",
        lambda self: 0,
    )
    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}
    response = client.post(
        "/research",
        json={"question": "What is dark matter?"},
        headers=headers,
    )
    assert response.status_code == 409
    assert "No documents indexed" in response.text


def test_research_invalid_profile_returns_422(monkeypatch):
    monkeypatch.setenv("AETHER_API_KEY", "test-key")
    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}
    response = client.post(
        "/research",
        json={"question": "What is dark matter?", "profile": "nonexistent"},
        headers=headers,
    )
    assert response.status_code == 422


def test_research_success(monkeypatch):
    from aether_scientist.agent.pipeline import ResearchAgent, ResearchResult
    from aether_scientist.agent.state import ResearchState

    state = ResearchState(question="What is gravity?")
    state.record("plan", "Decompose")
    res = ResearchResult(
        question="What is gravity?",
        report_md="# Research Report: What is gravity?",
        state=state,
        sources=[],
        confidence=0.85,
        coverage={"n_sources": 0, "n_docs": 0, "mean_score": 0.0},
    )

    monkeypatch.setenv("AETHER_API_KEY", "test-key")
    monkeypatch.setattr(ResearchAgent, "run", lambda *args, **kwargs: res)
    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}

    # Mock store to not be empty
    monkeypatch.setattr(
        "aether_scientist.retrieval.store.VectorStore.__len__",
        lambda self: 5,
    )

    response = client.post(
        "/research",
        json={"question": "What is gravity?", "profile": "fast"},
        headers=headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "report_md" in data["data"]
    assert data["data"]["confidence"] == 0.85


def test_research_stream_success(monkeypatch):
    from aether_scientist.agent.pipeline import ResearchAgent

    events = [
        {"event": "step", "kind": "plan", "description": "Decomposing"},
        {"event": "result", "data": {"report_md": "# Report"}},
    ]
    monkeypatch.setenv("AETHER_API_KEY", "test-key")
    monkeypatch.setattr(ResearchAgent, "stream", lambda *args, **kwargs: iter(events))
    monkeypatch.setattr(
        "aether_scientist.retrieval.store.VectorStore.__len__",
        lambda self: 5,
    )

    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}
    response = client.post(
        "/research/stream",
        json={"question": "What is quantum tunneling?"},
        headers=headers,
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "Decomposing" in response.text
    assert "# Report" in response.text


def test_analyze_empty_index_returns_409(monkeypatch):
    monkeypatch.setenv("AETHER_API_KEY", "test-key")
    monkeypatch.setattr(
        "aether_scientist.retrieval.store.VectorStore.__len__",
        lambda self: 0,
    )
    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}
    response = client.post(
        "/analyze",
        json={"query": "Explain gravity", "use_rag": True},
        headers=headers,
    )
    assert response.status_code == 409
    assert "No documents indexed" in response.text


def test_analyze_mocked_index_error_returns_clean_500(monkeypatch):
    monkeypatch.setenv("AETHER_API_KEY", "test-key")
    from aether_scientist.core.model import AetherScientist

    def fake_analyze(*args, **kwargs):
        raise IndexError("Internal index error")

    monkeypatch.setattr(AetherScientist, "analyze", fake_analyze)
    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}
    response = client.post(
        "/analyze",
        json={"query": "Explain gravity"},
        headers=headers,
    )
    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "internal error"
    assert "traceback" not in response.text.lower()
    assert "IndexError" not in response.text


def test_analyze_stream_sources_event_with_rag(monkeypatch):
    monkeypatch.setenv("AETHER_API_KEY", "test-key")
    monkeypatch.setattr(
        InferenceEngine,
        "stream",
        lambda *args, **kwargs: iter(["Analysis ", "done."]),
    )
    from aether_scientist.retrieval.engine import GroundedAnswer, RAGEngine, Source

    mock_src = Source(
        doc_id="d1", title="Paper 1", location="chunk_0", snippet="text", score=0.9
    )
    mock_ans = GroundedAnswer(answer="Grounded answer", sources=[mock_src], confidence=0.9)
    monkeypatch.setattr(RAGEngine, "grounded_generate", lambda *args, **kwargs: mock_ans)
    monkeypatch.setattr(
        "aether_scientist.retrieval.store.VectorStore.__len__",
        lambda self: 5,
    )

    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}
    response = client.post(
        "/analyze/stream",
        json={"query": "Explain entropy", "use_rag": True},
        headers=headers,
    )
    assert response.status_code == 200
    assert "event: sources" in response.text
    assert "Paper 1" in response.text


def test_roadmap_endpoints_return_501(monkeypatch):
    monkeypatch.setenv("AETHER_API_KEY", "test-key")
    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}

    r1 = client.post(
        "/synthesize", json={"domain": "physics", "paper_ids": ["p1"]}, headers=headers
    )
    assert r1.status_code == 501
    assert "not implemented — on roadmap" in r1.json()["detail"]

    r2 = client.post("/hypothesize", json={"observations": ["obs1"]}, headers=headers)
    assert r2.status_code == 501
    assert "not implemented — on roadmap" in r2.json()["detail"]

    r3 = client.post("/experiment", json={"hypothesis": "hyp1"}, headers=headers)
    assert r3.status_code == 501
    assert "not implemented — on roadmap" in r3.json()["detail"]
