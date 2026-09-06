"""Regression tests verifying remediation of all 10 external QA audit findings."""

import sys
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from aether_scientist.api.endpoints import app
from aether_scientist.benchmarks.eval import compute_citation_validity, run_eval
from aether_scientist.core.inference import InferenceEngine
from aether_scientist.multimodal.captioner import CaptionEngine
from aether_scientist.retrieval.embeddings import EmbeddingEngine
from aether_scientist.retrieval.engine import GroundedAnswer, RAGEngine, Source
from aether_scientist.retrieval.ingest import ingest_file


def test_qa_finding_1_zero_overlap_oov_retrieval_empty(tmp_path):
    """Finding 1: Out-of-vocabulary query vector is all-zero and returns 0 sources."""
    doc = tmp_path / "physics.txt"
    doc.write_text("Quantum electrodynamics explains photon interaction.", encoding="utf-8")
    rag = RAGEngine(embedder=EmbeddingEngine(backend="tfidf", cache_dir=str(tmp_path / "c")))
    rag.index([str(doc)])

    # Query with completely out-of-vocabulary words
    hits = rag.retrieve("xyzzy qwerty nonesuch", k=3)
    assert len(hits) == 0

    ans = rag.grounded_generate("xyzzy qwerty nonesuch")
    assert ans.sources == []
    assert "Insufficient evidence" in ans.answer


def test_qa_finding_2_prompt_truncation_no_index_error():
    """Finding 2: Long prompt truncates to context window without IndexError."""
    long_prompt = "word " * 5000
    res = InferenceEngine.generate(prompt=long_prompt, max_new_tokens=32)
    assert "text" in res
    assert len(res["text"]) > 0


def test_qa_finding_3_api_empty_index_409(monkeypatch):
    """Finding 3: /analyze with use_rag=True on empty index returns HTTP 409."""
    monkeypatch.setenv("AETHER_API_KEY", "test-key")
    monkeypatch.setattr("aether_scientist.retrieval.store.VectorStore.__len__", lambda self: 0)
    client = TestClient(app)
    response = client.post(
        "/analyze",
        json={"query": "What is quantum tunneling?", "use_rag": True},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 409
    assert "No documents indexed" in response.text


def test_qa_finding_4_api_clean_500_no_traceback(monkeypatch):
    """Finding 4: Unhandled exceptions return 500 clean message without traceback."""
    monkeypatch.setenv("AETHER_API_KEY", "test-key")
    from aether_scientist.core.model import AetherScientist

    def blow_up(*args, **kwargs):
        raise KeyError("Sensitive internal database column missing")

    monkeypatch.setattr(AetherScientist, "analyze", blow_up)
    client = TestClient(app)
    response = client.post(
        "/analyze",
        json={"query": "Explain magnetism"},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "internal error"
    assert "Sensitive" not in response.text
    assert "Traceback" not in response.text


def test_qa_finding_5_and_6_citation_validity_honest_and_quiz_null(monkeypatch):
    """Findings 5 & 6: Citation validity computes brackets or returns None in quiz."""
    garbled_output = ["The answer is given in quantum physics.", "Another answer."]
    assert compute_citation_validity(garbled_output, n_contexts=4) == 0.0

    valid_output = ["According to [2], gravity is attractive.", "Per [1], light bends."]
    assert compute_citation_validity(valid_output, n_contexts=4) == 1.0

    monkeypatch.setattr(
        "aether_scientist.core.inference.InferenceEngine.generate",
        lambda **kwargs: {"text": "Farad"},
    )
    quiz_report = run_eval(n=2)
    assert quiz_report.citation_validity_rate is None


def test_qa_finding_7_streaming_emits_real_sources(monkeypatch):
    """Finding 7: /analyze/stream with use_rag=True emits populated sources event."""
    monkeypatch.setenv("AETHER_API_KEY", "test-key")
    monkeypatch.setattr(
        InferenceEngine, "stream", lambda *args, **kwargs: iter(["token1", "token2"])
    )
    mock_src = Source(
        doc_id="doc1", title="Quantum Paper", location="chunk_0", snippet="text", score=0.95
    )
    mock_ans = GroundedAnswer(answer="Grounded answer", sources=[mock_src], confidence=0.95)
    monkeypatch.setattr(RAGEngine, "grounded_generate", lambda *args, **kwargs: mock_ans)
    monkeypatch.setattr("aether_scientist.retrieval.store.VectorStore.__len__", lambda self: 10)

    client = TestClient(app)
    response = client.post(
        "/analyze/stream",
        json={"query": "Explain superconductors", "use_rag": True},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    assert "event: sources" in response.text
    assert "Quantum Paper" in response.text


def test_qa_finding_8_chunk_section_tracking_markdown(tmp_path):
    """Finding 8: Markdown ingestion records heading in Chunk.section."""
    md = tmp_path / "survey.md"
    md.write_text(
        "# Introduction\n\nBackground overview of neural networks.\n\n"
        "## Architecture\n\nDetailed specifications of layer connectivity.\n\n"
        "### Experiments\n\nEmpirical benchmarks on ImageNet.",
        encoding="utf-8",
    )
    docs = ingest_file(md)
    chunks = docs[0].chunks
    assert len(chunks) >= 3
    assert chunks[0].section == "# Introduction"
    assert chunks[1].section == "## Architecture"
    assert chunks[2].section == "### Experiments"


def test_qa_finding_9_vision_distinguished_error_messages(tmp_path):
    """Finding 9: Missing PIL vs failed pipeline load produce distinct informative errors."""
    CaptionEngine.reset()
    engine = CaptionEngine()
    test_img = tmp_path / "diagram.png"
    test_img.write_bytes(b"image")

    # Missing PIL
    with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
        res_pil = engine.caption(test_img)
        assert "vision dependencies not installed (pip install aether-scientist[vision])" in res_pil

    # Failed pipeline load
    CaptionEngine.reset()
    engine = CaptionEngine()
    mock_pil = MagicMock()
    with (
        patch("PIL.Image.open", return_value=mock_pil),
        patch.object(engine, "_get_pipeline", return_value=False),
    ):
        res_pipe = engine.caption(test_img)
        assert "vision model failed to load (network/HF access required)" in res_pipe


def test_qa_finding_10_roadmap_endpoints_501(monkeypatch):
    """Finding 10: /synthesize, /hypothesize, /experiment return 501 on roadmap."""
    monkeypatch.setenv("AETHER_API_KEY", "test-key")
    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}

    for route, payload in [
        ("/synthesize", {"domain": "physics", "paper_ids": ["p1"]}),
        ("/hypothesize", {"observations": ["obs"]}),
        ("/experiment", {"hypothesis": "hyp"}),
    ]:
        res = client.post(route, json=payload, headers=headers)
        assert res.status_code == 501
        assert "not implemented — on roadmap" in res.json()["detail"]
