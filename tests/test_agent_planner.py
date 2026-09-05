from aether_scientist.agent.planner import extract_topic, plan, plan_llm


def test_extract_topic_question_words():
    assert extract_topic("What is quantum entanglement?") == "quantum entanglement"
    assert extract_topic("How does CRISPR work?") == "CRISPR work"
    assert extract_topic("Why is entropy increasing?") == "entropy increasing"
    assert extract_topic("Which particle has no charge?") == "particle has no charge"
    assert extract_topic("Can machines think?") == "machines think"


def test_extract_topic_trailing_punctuation_and_whitespace():
    assert extract_topic("  What are black holes???  ") == "black holes"
    assert extract_topic("Is dark matter real?!.") == "dark matter real"


def test_plan_heuristic_templates():
    sub_queries = plan("What is photosynthesis?", k=4)
    assert len(sub_queries) == 4
    assert any("definition and background" in sq for sq in sub_queries)
    assert any("mechanisms and principles" in sq for sq in sub_queries)
    assert any("applications" in sq for sq in sub_queries)
    assert any("limitations and challenges" in sq for sq in sub_queries)
    for sq in sub_queries:
        assert "photosynthesis" in sq


def test_plan_cap_and_min():
    sub_queries = plan("What is gravity?", k=2)
    assert len(sub_queries) == 2


def test_plan_empty_topic_fallback():
    assert plan("???") == ["???"]
    assert plan("") == [""]


def test_plan_llm_success():
    class MockEngine:
        def generate(self, **kwargs):
            return {
                "text": (
                    "1. What is dark energy?\n"
                    "2. How is it measured?\n"
                    "3. What theories explain it?"
                )
            }

    sq = plan_llm("What is dark energy?", engine=MockEngine(), k=3)
    assert len(sq) == 3
    assert sq[0] == "What is dark energy?"
    assert sq[1] == "How is it measured?"


def test_plan_llm_fallback_on_parse_failure():
    class BrokenEngine:
        def generate(self, **kwargs):
            return {"text": "I am not able to answer this question at this time."}

    sq = plan_llm("What is graphene?", engine=BrokenEngine(), k=4)
    assert len(sq) == 4
    assert any("definition and background" in s for s in sq)
