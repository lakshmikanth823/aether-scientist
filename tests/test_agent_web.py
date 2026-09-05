from fastapi.testclient import TestClient
from typer.testing import CliRunner

from aether_scientist.agent.pipeline import ResearchAgent
from aether_scientist.api.endpoints import app as fastapi_app
from aether_scientist.cli import app as cli_app
from aether_scientist.core.config import AetherConfig
from aether_scientist.core.inference import InferenceEngine
from aether_scientist.retrieval.chunker import Chunk
from aether_scientist.retrieval.web_search import WebHit

runner = CliRunner()


def test_agent_web_fallback_when_store_empty(monkeypatch, tmp_path):
    cfg = AetherConfig(profile="offline", cache_dir=str(tmp_path / "cache"))
    agent = ResearchAgent(cfg)
    agent.rag_engine.store.clear()
    assert len(agent.rag_engine.store) == 0

    fake_hits = [
        WebHit(
            title="Fusion Energy Advances",
            url="https://nature.com/articles/fusion-net-gain",
            snippet="Net energy gain in magnetic confinement tokamak achieved.",
        )
    ]
    fake_chunks = [
        Chunk(
            chunk_id="chunk_0",
            doc_id="https://nature.com/articles/fusion-net-gain",
            text="Net energy gain in magnetic confinement tokamak achieved with high plasma pressure.",
            index=0,
        )
    ]

    monkeypatch.setattr(
        "aether_scientist.retrieval.web_search.WebSearchEngine.search",
        lambda *args, **kwargs: fake_hits,
    )
    monkeypatch.setattr(
        "aether_scientist.agent.pipeline.fetch_and_chunk",
        lambda *args, **kwargs: fake_chunks,
    )
    monkeypatch.setattr(
        InferenceEngine,
        "generate",
        lambda **kwargs: {"text": "Net energy gain has been achieved [1]."},
    )

    res = agent.run("What are recent fusion energy advances?", use_web=True)
    assert res.report_md
    assert "Executive Summary" in res.report_md
    assert "References" in res.report_md
    assert "[web]" in res.report_md
    assert "https://nature.com/articles/fusion-net-gain" in res.report_md
    assert len(res.sources) >= 1
    assert any("[web]" in s.get("title", "") for s in res.sources)
    assert any("nature.com" in s.get("source", "") for s in res.sources)


def test_agent_fails_when_store_empty_and_no_web(tmp_path):
    import pytest

    cfg = AetherConfig(profile="offline", cache_dir=str(tmp_path / "cache"))
    agent = ResearchAgent(cfg)
    agent.rag_engine.store.clear()
    with pytest.raises(IndexError, match="No documents indexed"):
        agent.run("Any question", use_web=False)


def test_cli_research_with_web_flag(monkeypatch, tmp_path):
    fake_hits = [WebHit(title="Superconductors", url="https://arxiv.org/abs/sc", snippet="High-Tc")]
    fake_chunks = [
        Chunk(chunk_id="c0", doc_id="https://arxiv.org/abs/sc", text="High-Tc superconductor.", index=0)
    ]

    monkeypatch.setattr(
        "aether_scientist.retrieval.web_search.WebSearchEngine.search",
        lambda *args, **kwargs: fake_hits,
    )
    monkeypatch.setattr(
        "aether_scientist.agent.pipeline.fetch_and_chunk",
        lambda *args, **kwargs: fake_chunks,
    )
    monkeypatch.setattr(
        InferenceEngine,
        "generate",
        lambda **kwargs: {"text": "High temperature superconductors discovered [1]."},
    )

    out_file = tmp_path / "report.md"
    result = runner.invoke(
        cli_app,
        ["research", "High-Tc superconductors", "--web", "--out", str(out_file), "--json"],
    )
    assert result.exit_code == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "https://arxiv.org/abs/sc" in content


def test_api_research_with_web_field(monkeypatch):
    fake_hits = [WebHit(title="Graphene", url="https://materials.org/graphene", snippet="2D carbon")]
    fake_chunks = [
        Chunk(chunk_id="c0", doc_id="https://materials.org/graphene", text="Graphene 2D lattice.", index=0)
    ]

    monkeypatch.setattr(
        "aether_scientist.retrieval.web_search.WebSearchEngine.search",
        lambda *args, **kwargs: fake_hits,
    )
    monkeypatch.setattr(
        "aether_scientist.agent.pipeline.fetch_and_chunk",
        lambda *args, **kwargs: fake_chunks,
    )
    monkeypatch.setattr(
        InferenceEngine,
        "generate",
        lambda **kwargs: {"text": "Graphene forms a 2D honeycomb lattice [1]."},
    )

    client = TestClient(fastapi_app)
    response = client.post(
        "/research",
        json={"question": "What is graphene?", "use_web": True},
        headers={"X-API-Key": "dev-aether-key"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "report_md" in data["data"]
    assert "materials.org" in data["data"]["report_md"]
