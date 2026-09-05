from aether_scientist.agent.report import render_report, write_report
from aether_scientist.agent.state import ResearchState


def test_render_report_structure():
    state = ResearchState(question="How does photosynthesis work?", profile="fast")
    state.add_sources(
        [
            {
                "doc_id": "bio1",
                "chunk_id": 0,
                "title": "Plant Biology",
                "source": "plants.txt",
                "score": 0.88,
            },
            {
                "doc_id": "bio2",
                "chunk_id": 1,
                "title": "Chloroplast Dynamics",
                "source": "chloroplast.txt",
                "score": 0.92,
            },
        ]
    )
    state.findings.append(
        {
            "sub_query": "What is photosynthesis?",
            "text": (
                "Photosynthesis converts light into chemical energy [1]. "
                "It occurs in chloroplasts."
            ),
            "citations": [1],
        }
    )
    state.findings.append(
        {
            "sub_query": "How does it work?",
            "text": "Chlorophyll pigments absorb photons [2]. This drives ATP synthesis.",
            "citations": [2],
        }
    )

    report = render_report(
        question=state.question, state=state, confidence=0.90, profile="fast", device="cpu"
    )
    assert "# Research Report: How does photosynthesis work?" in report
    assert "## Executive Summary" in report
    assert "Photosynthesis converts light into chemical energy [1]." in report
    assert "## Key Findings" in report
    assert "### What is photosynthesis?" in report
    assert "## Coverage & Limitations" in report
    assert "## References" in report
    assert "[1] Plant Biology — plants.txt, chunk_0 (score 0.88)" in report
    assert "[2] Chloroplast Dynamics — chloroplast.txt, chunk_1 (score 0.92)" in report


def test_render_report_citation_validation():
    state = ResearchState(question="Explain quantum mechanics")
    state.add_sources(
        [{"doc_id": "d1", "chunk_id": 0, "title": "QM Intro", "source": "qm.pdf", "score": 0.9}]
    )
    # Finding contains valid citation [1] and hallucinated citation [42]
    state.findings.append(
        {
            "sub_query": "Basics",
            "text": "Wavefunctions collapse on measurement [1], leading to reality branching [42].",
            "citations": [1],
        }
    )
    report = render_report("Explain quantum mechanics", state)
    assert "[1]" in report
    assert "[42]" not in report


def test_render_report_insufficient_evidence_marks_unanswered():
    state = ResearchState(question="Explain dark matter")
    state.findings.append(
        {
            "sub_query": "Composition of dark matter",
            "text": "Insufficient evidence in indexed corpus.",
            "citations": [],
        }
    )
    report = render_report("Explain dark matter", state)
    assert "Composition of dark matter" in report
    assert "Unanswered sub-queries: Composition of dark matter" in report


def test_write_report_roundtrip(tmp_path):
    out_file = tmp_path / "reports" / "summary.md"
    content = "# Test Report\nContent line."
    written = write_report(content, out_file)
    assert written.exists()
    assert written.read_text(encoding="utf-8") == content
