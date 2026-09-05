from aether_scientist.benchmarks.eval import _load_quiz, run_eval


def test_bundled_quiz_structure():
    questions = _load_quiz(use_sciq=False)
    assert len(questions) == 20
    for q in questions:
        assert "id" in q
        assert "domain" in q
        assert "question" in q
        assert "options" in q
        assert len(q["options"]) == 4
        assert "answer" in q
        assert q["answer"] in q["options"]


def test_benchmark_eval_mocked(monkeypatch):
    # Mock generator to answer "Farad" for first question
    monkeypatch.setattr(
        "aether_scientist.core.inference.InferenceEngine.generate",
        lambda **kwargs: {"text": "The correct answer is Farad."},
    )

    report = run_eval(n=3)
    assert report.total == 3
    assert report.correct >= 1
    assert 0.0 <= report.accuracy <= 1.0
    assert report.avg_latency_ms >= 0.0
    assert report.citation_validity_rate == 1.0
    assert len(report.details) == 3

    d = report.to_dict()
    assert d["total"] == 3
    assert "accuracy" in d
    assert "details" in d
