"""Tests for confidence gating and URL fabrication guard."""

from aether_scientist.core.format import clean_output
from aether_scientist.retrieval.chunker import Chunk
from aether_scientist.retrieval.engine import RAGEngine
from aether_scientist.retrieval.store import Hit


def test_low_score_retrieval_never_calls_inference_engine(monkeypatch):
    """When mean score < 0.35, InferenceEngine.generate must not be called."""
    rag = RAGEngine()
    mock_hits = [
        Hit(
            chunk=Chunk(
                chunk_id="c1",
                doc_id="d1",
                text="Attention is all you need...",
                index=0,
            ),
            doc_id="d1",
            source="paper.pdf",
            score=0.22,
            title="Attention Paper",
        )
    ]
    monkeypatch.setattr(rag, "retrieve", lambda q, k=None: mock_hits)

    llm_called = False

    def fake_generate(**kwargs):
        nonlocal llm_called
        llm_called = True
        return {"text": "Fabricated answer http://fake.url", "model": "test"}

    monkeypatch.setattr(
        "aether_scientist.core.inference.InferenceEngine.generate", fake_generate
    )

    answer = rag.grounded_generate(
        "What does this paper conclude about quantum error correction?"
    )
    assert not llm_called, "InferenceEngine must NOT be called when confidence < 0.35"
    assert "Insufficient evidence" in answer.answer
    assert "confidence 0.22" in answer.answer


def test_fabricated_urls_sanitized():
    """URLs not in source snippets should be stripped or trigger fallback."""
    # Test stripping uncited URLs
    text = "Find details at http://fabricated.org/fake and cited at http://real.edu/data."
    snippets = ["Real facts from http://real.edu/data."]
    cleaned = clean_output(text, source_snippets=snippets)
    assert "http://real.edu/data" in cleaned
    assert "http://fabricated.org/fake" not in cleaned

    # Test fallback when > 50% of text is stripped
    fake_only = "http://hallucination1.com/link http://hallucination2.com/link"
    cleaned_fallback = clean_output(fake_only, source_snippets=["Clean context without links"])
    assert "Insufficient evidence" in cleaned_fallback
