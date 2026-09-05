import pytest

from aether_scientist.domains import available_domains, load_adapter
from aether_scientist.domains.base import DomainResult


def test_available_domains():
    domains = available_domains()
    assert isinstance(domains, list)
    assert "physics" in domains
    assert "chemistry" in domains
    assert "biology" in domains


def test_domain_result():
    result = DomainResult(
        domain="physics",
        analysis="test analysis",
        confidence=0.9,
        reasoning_chain=["step 1"],
        citations=["ref1"],
        metadata={"key": "val"},
    )
    assert result.domain == "physics"
    assert result.analysis == "test analysis"
    assert result.confidence == 0.9
    assert result.reasoning_chain == ["step 1"]
    assert result.citations == ["ref1"]
    assert result.metadata == {"key": "val"}


def test_load_adapter_invalid():
    with pytest.raises(ValueError):
        load_adapter("unknown")


@pytest.mark.parametrize("domain_name", ["physics", "chemistry", "biology"])
def test_adapters(domain_name):
    adapter = load_adapter(domain_name)

    term = adapter.terminology()
    assert isinstance(term, dict)
    assert len(term) >= 20

    rp = adapter.reasoning_patterns()
    assert isinstance(rp, list)
    assert len(rp) > 0

    em = adapter.experimental_methods()
    assert isinstance(em, list)
    assert len(em) > 0

    assert adapter.domain_name == domain_name

    res_process = adapter.process([1, 2, 3], context={"test": 1})
    assert hasattr(res_process, "domain")

    res_synth = adapter.synthesize(["paper1"])
    assert hasattr(res_synth, "domain")

    res_hypo = adapter.hypothesize(["obs1"])
    assert hasattr(res_hypo, "domain")


def test_physics_adapter_specifics():
    adapter = load_adapter("physics")
    assert hasattr(adapter, "validate_units")
    res = adapter.validate_units("5 m/s")
    assert isinstance(res, bool)
