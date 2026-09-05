import pytest

from aether_scientist.agent.state import AgentStepLimit, ResearchState


def test_state_initialization():
    state = ResearchState(question="Explain entropy", profile="fast", max_steps=5)
    assert state.question == "Explain entropy"
    assert state.profile == "fast"
    assert state.max_steps == 5
    assert len(state.steps) == 0
    assert len(state.sources) == 0
    assert len(state.findings) == 0


def test_state_record_step_and_limit_enforcement():
    state = ResearchState(question="Explain entropy", max_steps=2)
    step1 = state.record("plan", "Decomposing question")
    assert step1.kind == "plan"
    assert step1.status == "done"
    assert len(state.steps) == 1

    step2 = state.record("retrieve", "Retrieving sources")
    assert step2.kind == "retrieve"
    assert len(state.steps) == 2

    with pytest.raises(AgentStepLimit, match="Agent step limit of 2 exceeded"):
        state.record("synthesize", "Synthesizing answer")


def test_add_sources_deduplication_and_global_numbering():
    state = ResearchState(question="Test question")

    hits_batch_1 = [
        {"doc_id": "doc1", "chunk_id": 0, "title": "Paper 1", "source": "p1.pdf", "score": 0.70},
        {"doc_id": "doc2", "chunk_id": 0, "title": "Paper 2", "source": "p2.pdf", "score": 0.80},
    ]
    indices1 = state.add_sources(hits_batch_1)
    assert indices1 == [1, 2]
    assert len(state.sources) == 2
    assert state.sources[0]["score"] == 0.70

    # Batch 2 contains doc1 chunk 0 again with higher score + new doc3
    hits_batch_2 = [
        {"doc_id": "doc1", "chunk_id": 0, "title": "Paper 1", "source": "p1.pdf", "score": 0.95},
        {"doc_id": "doc3", "chunk_id": 0, "title": "Paper 3", "source": "p3.pdf", "score": 0.60},
    ]
    indices2 = state.add_sources(hits_batch_2)
    assert indices2 == [1, 3]  # doc1 mapped back to 1, doc3 is 3
    assert len(state.sources) == 3
    # Verify higher score updated
    assert state.sources[0]["score"] == 0.95


def test_state_coverage():
    state = ResearchState(question="Test question")
    cov_empty = state.coverage()
    assert cov_empty["n_sources"] == 0
    assert cov_empty["n_docs"] == 0
    assert cov_empty["mean_score"] == 0.0

    state.add_sources(
        [
            {"doc_id": "doc1", "chunk_id": 0, "source": "p1.pdf", "score": 0.8},
            {"doc_id": "doc1", "chunk_id": 1, "source": "p1.pdf", "score": 0.6},
            {"doc_id": "doc2", "chunk_id": 0, "source": "p2.pdf", "score": 1.0},
        ]
    )
    cov = state.coverage()
    assert cov["n_sources"] == 3
    assert cov["n_docs"] == 2
    assert cov["mean_score"] == 0.80
