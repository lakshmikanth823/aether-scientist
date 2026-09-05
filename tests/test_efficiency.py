import pytest

from aether_scientist.utils.efficiency import CacheManager, EfficiencyOptimizer
from aether_scientist.utils.validation import OutputValidator

try:
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_cache_manager():
    cache = CacheManager(max_size=3)
    cache.put("key1", "val1")
    assert cache.get("key1") == "val1"
    assert cache.get("missing") is None

    cache.put("key2", "val2")
    cache.put("key3", "val3")
    cache.put("key4", "val4")
    assert cache.get("key1") is None

    stats = cache.stats()
    assert "hits" in stats


def test_efficiency_optimizer_init():
    opt = EfficiencyOptimizer(bits=4, cache_enabled=True, cache_size=100)
    assert opt.bits == 4
    assert hasattr(opt, "cache")


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_efficiency_optimizer_optimize():
    opt = EfficiencyOptimizer(bits=4, cache_enabled=True, cache_size=100)
    model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
    optimized = opt.optimize(model)
    assert optimized is not None


def test_efficiency_optimizer_compression_report():
    opt = EfficiencyOptimizer(bits=4, cache_enabled=True, cache_size=100)
    report = opt._compression_report(1000, 700)
    assert "compression_ratio" in report
    assert abs(report["compression_ratio"] - 0.3) < 1e-6


def test_output_validator_valid():
    validator = OutputValidator()
    result = validator.validate({"text": "Short text"})
    assert result.is_valid is True


def test_output_validator_citations_warning():
    validator = OutputValidator()
    result = validator.validate({"text": "x" * 600})
    assert len(result.warnings) > 0


def test_output_validator_unit_error():
    validator = OutputValidator()
    result = validator.validate({"text": "We used 5 kg and 3 g of reagent"})
    assert len(result.errors) > 0


def test_output_validator_statistical_warning():
    validator = OutputValidator()
    result = validator.validate({"text": "The result was p < 0.05"})
    assert len(result.warnings) > 0
