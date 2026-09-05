"""Tests for agent quality: topic extraction, citation backfill, and sentence trimming."""

import numpy as np

from aether_scientist.agent.pipeline import ResearchAgent
from aether_scientist.agent.planner import extract_topic
from aether_scientist.core.format import clean_output
from aether_scientist.retrieval.chunker import Chunk
from aether_scientist.retrieval.store import Hit


def test_sub_query_grammar_topic_extraction():
    """Verify topic extraction strips question words, auxiliaries, second clauses, lone verbs."""
    q = "How does the attention mechanism work and what are its limitations?"
    assert extract_topic(q) == "the attention mechanism"

    # Additional queries
    assert (
        extract_topic("What is quantum entanglement and why is it useful?")
        == "quantum entanglement"
    )
    assert extract_topic("How does CRISPR works?") == "CRISPR"


def test_truncated_output_trimmed_to_complete_sentence():
    """Verify mid-sentence truncation and dangling list markers are trimmed."""
    t1 = "Attention mechanism improves sequence transduction. 4. **Computational"
    assert clean_output(t1) == "Attention mechanism improves sequence transduction."

    t2 = "Models achieve state of the art in three areas. Research areas: 1."
    assert clean_output(t2) == "Models achieve state of the art in three areas."

    t3 = "A perfectly complete sentence."
    assert clean_output(t3) == "A perfectly complete sentence."


def test_deterministic_citation_backfill(monkeypatch):
    """Verify pipeline backfills (sources: [i], ...) when model finding omits [n] markers."""
    agent = ResearchAgent()
    c = Chunk(chunk_id="c0", doc_id="d0", text="Self-attention replaces recurrence.", index=0)
    agent.rag_engine.store.add(
        [c], np.zeros((1, 8), dtype=np.float32), sources=["paper.pdf"], titles=["Paper"]
    )

    hit = Hit(chunk=c, doc_id="d0", source="paper.pdf", score=0.9, title="Paper")
    monkeypatch.setattr(agent.rag_engine, "retrieve", lambda q, k=None: [hit])

    # Mock generator returning finding with NO citation brackets
    monkeypatch.setattr(
        "aether_scientist.core.inference.InferenceEngine.generate",
        lambda **kwargs: {"text": "Self-attention replaces recurrence.", "model": "test"},
    )

    events = list(agent.stream("How does attention work?", k=1))
    result_ev = next(e for e in events if e.get("event") == "result")
    result = result_ev["result_object"]

    # Finding should have backfilled (sources: [1])
    assert any("(sources: [1])" in f["text"] for f in result.state.findings)
    assert 1 in result.state.findings[0]["citations"]
