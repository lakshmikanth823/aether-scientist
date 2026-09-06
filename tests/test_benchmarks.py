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
    assert report.citation_validity_rate is None
    assert len(report.details) == 3

    d = report.to_dict()
    assert d["total"] == 3
    assert "accuracy" in d
    assert "details" in d
    assert "profile" in d
    assert "device" in d
    assert d.get("citation_validity_rate") is None


def test_citation_validity_computation():
    from aether_scientist.benchmarks.eval import compute_citation_validity

    # 1. Garbled text without valid [n] brackets -> 0.0
    garbled = ["Quantum decoherence occurs naturally.", "Energy is conserved."]
    assert compute_citation_validity(garbled, n_contexts=4) == 0.0

    # 2. Answers with [1] where n_contexts=4 -> 1.0
    cited = ["According to [1], photons carry momentum.", "As shown in [3], entropy increases."]
    assert compute_citation_validity(cited, n_contexts=4) == 1.0

    # 3. Out of bounds brackets -> 0.0
    out_of_bounds = ["See [5] for proof.", "Invalid context [0]."]
    assert compute_citation_validity(out_of_bounds, n_contexts=4) == 0.0

    # 4. Partial valid -> 0.5
    mixed = ["Valid [2] here.", "No citation at all."]
    assert compute_citation_validity(mixed, n_contexts=4) == 0.5

    # 5. Empty answers -> 0.0
    assert compute_citation_validity([], n_contexts=4) == 0.0


def test_benchmark_eval_with_rag_answers(monkeypatch):
    monkeypatch.setattr(
        "aether_scientist.core.inference.InferenceEngine.generate",
        lambda **kwargs: {"text": "Farad"},
    )
    rag_answers = ["Answer supported by [1].", "Another answer supported by [2]."]
    report = run_eval(n=2, rag_answers=rag_answers, n_contexts=4)
    assert report.citation_validity_rate == 1.0


def test_compare_profiles_mocked():
    def fake_gen(prompt, profile, **kwargs):
        return {"text": "The answer is Farad."}

    from aether_scientist.benchmarks.eval import compare_profiles

    res = compare_profiles(["offline", "fast"], n=2, generator=fake_gen)
    assert "profiles" in res
    assert "summary" in res
    assert len(res["summary"]) == 2
    assert res["summary"][0]["profile"] == "offline"
    assert res["summary"][1]["profile"] == "fast"
    assert res["profiles"]["offline"]["total"] == 2

