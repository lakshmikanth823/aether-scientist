import builtins

import numpy as np
import pytest

from aether_scientist.retrieval.chunker import chunk
from aether_scientist.retrieval.embeddings import EmbeddingEngine
from aether_scientist.retrieval.engine import RAGEngine
from aether_scientist.retrieval.ingest import ingest_file
from aether_scientist.retrieval.store import VectorStore


def test_chunker_invariants():
    text = (
        "Paragraph one discusses quantum mechanics and wavefunction collapse.\n\n"
        "Paragraph two covers general relativity and curved spacetime.\n\n"
        "Paragraph three examines thermodynamics and statistical entropy."
    )
    chunks = chunk(text, size=80, overlap=20, doc_id="doc_1")
    assert len(chunks) >= 3
    assert all(c.doc_id == "doc_1" for c in chunks)
    assert all(len(c.text) <= 120 for c in chunks)
    assert chunks[0].index == 0


def test_tfidf_embedding_engine(tmp_path):
    engine = EmbeddingEngine(backend="tfidf", cache_dir=str(tmp_path / "cache"))
    texts = [
        "quantum entanglement and photons",
        "organic synthesis and chemical catalysts",
        "cellular respiration and mitochondria",
    ]
    vectors = engine.embed(texts)
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape[0] == 3
    # Check L2 normalization (norm == 1.0)
    for row in vectors:
        assert abs(np.linalg.norm(row) - 1.0) < 1e-4

    # Related query should have highest similarity with first doc
    query_vec = engine.embed(["quantum photon physics"])
    sims = np.dot(vectors, query_vec.T).flatten()
    assert sims[0] > sims[1]
    assert sims[0] > sims[2]


def test_vector_store_add_search_save_load(tmp_path):
    store = VectorStore()
    text = "Photosynthesis occurs in chloroplasts. Mitochondria generate ATP energy."
    chunks = chunk(text, size=40, overlap=10, doc_id="bio_1")
    engine = EmbeddingEngine(backend="tfidf", cache_dir=str(tmp_path / "cache"))
    vectors = engine.embed([c.text for c in chunks])

    sources = ["paper.txt" for _ in chunks]
    titles = ["Bio Paper" for _ in chunks]
    added = store.add(chunks, vectors, sources=sources, titles=titles)
    assert added == len(chunks)
    assert len(store) == len(chunks)

    # Search
    q_vec = engine.embed(["ATP mitochondria energy"])
    hits = store.search(q_vec[0], k=2)
    assert len(hits) <= 2
    assert hits[0].doc_id == "bio_1"
    assert hits[0].score >= hits[1].score

    # Save & Load round-trip
    save_path = tmp_path / "store" / "test_store"
    store.save(save_path)

    loaded_store = VectorStore()
    loaded_store.load(save_path)
    assert len(loaded_store) == len(store)

    loaded_hits = loaded_store.search(q_vec[0], k=2)
    assert len(loaded_hits) == len(hits)
    assert abs(loaded_hits[0].score - hits[0].score) < 1e-5


def test_rag_citation_grounding_and_guard(tmp_path, monkeypatch):
    # Setup test document
    doc_path = tmp_path / "physics_notes.txt"
    doc_path.write_text(
        "Conservation of energy states that energy cannot be created or destroyed. "
        "Entropy always increases in isolated systems according to the second law.",
        encoding="utf-8",
    )

    rag = RAGEngine(embedder=EmbeddingEngine(backend="tfidf", cache_dir=str(tmp_path / "c")))
    stats = rag.index([str(doc_path)])
    assert stats.docs == 1
    assert stats.chunks >= 1

    # Mock generator returning text with both valid [1] and hallucinated [99] citations
    monkeypatch.setattr(
        "aether_scientist.core.inference.InferenceEngine.generate",
        lambda **kwargs: {
            "text": "Energy is conserved [1], but spontaneous creation occurs [99].",
            "model": "distilgpt2",
        },
    )

    answer = rag.grounded_generate("What is energy conservation?", domain="physics")
    assert len(answer.sources) >= 1
    assert answer.confidence > 0.0
    # Invalid citation [99] must be stripped by hallucination guard
    assert "[1]" in answer.answer
    assert "[99]" not in answer.answer
    assert answer.citations_valid is False


def test_ingest_file_text(tmp_path):
    f = tmp_path / "sample.md"
    f.write_text(
        "# Dark Matter Research\nObservations indicate gravitational lensing.",
        encoding="utf-8",
    )
    docs = ingest_file(f)
    assert len(docs) == 1
    assert docs[0].title == "Dark Matter Research"
    assert "gravitational lensing" in docs[0].text


def test_ingest_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ingest_file("nonexistent_path_12345.pdf")


def test_ingest_pdf_missing_fitz(tmp_path, monkeypatch):
    pdf_file = tmp_path / "fake.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 test")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fitz":
            raise ImportError("No fitz")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="PyMuPDF is required for PDF ingestion"):
        ingest_file(pdf_file)


def test_vector_store_empty_search():
    store = VectorStore()
    hits = store.search(np.array([1.0, 0.0]), k=3)
    assert hits == []
    assert len(store) == 0


def test_rag_engine_empty_retrieve_and_grounded():
    rag = RAGEngine()
    hits = rag.retrieve("any query")
    assert hits == []

    grounded = rag.grounded_generate("query with no indexed docs")
    assert "Insufficient evidence" in grounded.answer
    assert grounded.sources == []
    assert grounded.confidence == 0.0


def test_rag_retrieve_deduplicates_chunks(tmp_path):
    rag = RAGEngine(embedder=EmbeddingEngine(backend="tfidf", cache_dir=str(tmp_path / "c")))
    c1 = chunk("Quantum mechanics describes atomic particles and wavefunctions.", doc_id="doc_a")
    vecs = rag.embedder.embed([c.text for c in c1])
    # Add identical chunks to simulate duplicate indexing
    rag.store.add(c1, vecs, sources=["p.txt"], titles=["Doc A"])
    rag.store.add(c1, vecs, sources=["p.txt"], titles=["Doc A"])
    assert len(rag.store) == 2 * len(c1)

    hits = rag.retrieve("quantum atomic particles", k=4)
    keys = [(h.doc_id, getattr(h.chunk, "chunk_id", h.chunk.index)) for h in hits]
    assert len(keys) == len(set(keys))
    assert len(hits) == len(c1)


def test_ingest_sanitizes_special_tokens(tmp_path):
    """Special token strings (<EOS>, <pad>, <|...|>) must be stripped from text."""
    f = tmp_path / "tokens.txt"
    f.write_text(
        "First line title\nSome text <EOS> more <pad> text <|endoftext|> end.",
        encoding="utf-8",
    )
    docs = ingest_file(f)
    assert "<EOS>" not in docs[0].text
    assert "<pad>" not in docs[0].text
    assert "<|endoftext|>" not in docs[0].text
    assert "Some text" in docs[0].text


def test_ingest_skips_boilerplate_title(tmp_path):
    """Title extraction should skip boilerplate lines (license, arxiv, copyright)."""
    f = tmp_path / "boiler.txt"
    f.write_text(
        "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023\n"
        "Attention Is All You Need\nSome body text.\n",
        encoding="utf-8",
    )
    docs = ingest_file(f)
    assert docs[0].title == "Attention Is All You Need"


def test_chunker_deduplicates_content():
    """Chunks with identical whitespace-collapsed text should be dropped."""
    text = "Hello world.\n\nHello world.\n\nDifferent content here."
    chunks_out = chunk(text, size=512, doc_id="d")
    texts = [c.text for c in chunks_out]
    # "Hello world." should appear at most once
    assert texts.count("Hello world.") <= 1
    assert any("Different content" in t for t in texts)


def test_confidence_gate_skips_llm(monkeypatch):
    """When mean top-k score < 0.35, grounded_generate should NOT call the LLM."""
    from aether_scientist.retrieval.chunker import Chunk
    from aether_scientist.retrieval.store import Hit

    rag = RAGEngine()
    mock_hits = [
        Hit(
            chunk=Chunk(
                chunk_id="c1",
                doc_id="d1",
                text="Attention is a mechanism...",
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
    assert "Insufficient evidence" in answer.answer
    assert answer.confidence == 0.22
    assert not llm_called, "LLM should NOT be called when confidence < 0.35"


def test_rag_index_force_clears_store(tmp_path):
    """RAGEngine.index(force=True) should wipe store before re-indexing."""
    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Some scientific text about physics.", encoding="utf-8")

    rag = RAGEngine(embedder=EmbeddingEngine(backend="tfidf", cache_dir=str(tmp_path / "c")))
    rag.index([str(doc_path)])
    count_after_first = len(rag.store)
    assert count_after_first > 0

    rag.index([str(doc_path)], force=True)
    count_after_rebuild = len(rag.store)
    assert count_after_rebuild == count_after_first  # Same docs, same count (not doubled)


