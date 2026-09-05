import pytest

from aether_scientist.agent.pipeline import ResearchAgent, ResearchResult
from aether_scientist.core.config import AetherConfig, RagConfig
from aether_scientist.core.inference import InferenceEngine


def test_agent_empty_index_raises_index_error(tmp_path):
    config = AetherConfig(rag=RagConfig(cache_dir=str(tmp_path / "empty_cache")))
    agent = ResearchAgent(config)
    with pytest.raises(IndexError, match="No documents indexed"):
        agent.run("What is entropy?")


def test_agent_pipeline_end_to_end_with_ingestion(tmp_path, monkeypatch):
    # 1. Create temporary scientific document
    doc_path = tmp_path / "superconductivity.txt"
    doc_path.write_text(
        "Superconductivity is a physical phenomenon observed in certain materials.\n\n"
        "At critical temperatures, electrical resistance drops precisely to zero.",
        encoding="utf-8",
    )

    # 2. Configure agent with isolated cache dir and small chunk size
    cache_dir = str(tmp_path / "cache")
    config = AetherConfig(
        profile="fast",
        rag=RagConfig(cache_dir=cache_dir, chunk_size=60, overlap=10),
    )
    agent = ResearchAgent(config)

    # 3. Ingest document
    stats = agent.rag_engine.index([str(doc_path)])
    assert stats.docs == 1
    assert stats.chunks >= 2

    # 4. Mock InferenceEngine to generate cited statements
    monkeypatch.setattr(
        InferenceEngine,
        "generate",
        lambda **kwargs: {"text": "Superconductivity exhibits zero electrical resistance [1]."},
    )

    # 5. Run research pipeline
    result = agent.run("What is superconductivity?", k=2)
    assert isinstance(result, ResearchResult)
    assert result.confidence > 0.0
    assert len(result.sources) >= 1
    assert len(result.state.findings) >= 2

    # Verify report sections & citation resolution
    report = result.report_md
    assert "# Research Report: What is superconductivity?" in report
    assert "## Executive Summary" in report
    assert "## Key Findings" in report
    assert "## References" in report
    assert "[1]" in report and "superconductivity.txt" in report

    # Verify to_dict structure
    d = result.to_dict()
    assert d["question"] == "What is superconductivity?"
    assert len(d["steps"]) >= 4
    assert len(d["findings"]) >= 2


def test_agent_pipeline_stream_events(tmp_path, monkeypatch):
    doc_path = tmp_path / "physics.txt"
    doc_path.write_text(
        "Quantum mechanics describes matter and radiation at an atomic scale.",
        encoding="utf-8",
    )

    config = AetherConfig(rag=RagConfig(cache_dir=str(tmp_path / "cache")))
    agent = ResearchAgent(config)
    agent.rag_engine.index([str(doc_path)])

    monkeypatch.setattr(
        InferenceEngine,
        "generate",
        lambda **kwargs: {"text": "Quantum mechanics governs micro states [1]."},
    )

    events = list(agent.stream("What is quantum mechanics?", k=2))
    assert len(events) >= 5

    event_kinds = [e.get("kind") for e in events if e.get("event") == "step"]
    assert "plan" in event_kinds
    assert "retrieve" in event_kinds
    assert "synthesize" in event_kinds
    assert "report" in event_kinds

    # Final event carries result object
    last = events[-1]
    assert last["event"] == "result"
    assert "report_md" in last["data"]
