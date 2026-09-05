import pytest

from aether_scientist.core.config import (
    AetherConfig,
    EfficiencyConfig,
    RagConfig,
    TokenizerConfig,
)
from aether_scientist.core.inference import InferenceEngine
from aether_scientist.core.model import AetherScientist
from aether_scientist.core.tokenizer import ScientificTokenizer


def test_aether_config_defaults():
    config = AetherConfig()
    assert config.model_name == "distilgpt2"
    assert config.model_size == "82M"
    assert config.quantization_bits == 4
    assert config.domains == ["physics", "chemistry", "biology"]
    assert config.reasoning_depth == 8
    assert config.max_papers == 100
    assert config.max_sequence_length == 4096
    assert config.batch_size == 8
    assert config.device == "auto"
    assert config.cache_enabled is True
    assert config.cache_size == 1000
    assert config.api_rate_limit == 60
    assert config.max_new_tokens == 256
    assert config.temperature == 0.7
    assert isinstance(config.tokenizer, TokenizerConfig)
    assert config.tokenizer.vocab_size == 50000
    assert isinstance(config.tokenizer.special_tokens, list)
    assert isinstance(config.efficiency, EfficiencyConfig)
    assert config.efficiency.use_flash_attention is True
    assert isinstance(config.rag, RagConfig)
    assert config.rag.chunk_size == 512
    assert config.rag.overlap == 64
    assert config.rag.backend == "auto"


def test_aether_config_custom():
    config = AetherConfig(model_size="7B", quantization_bits=8, domains=["physics"])
    assert config.model_size == "7B"
    assert config.quantization_bits == 8
    assert config.domains == ["physics"]


def test_scientific_tokenizer_special_tokens():
    tok = ScientificTokenizer()
    assert tok.vocab["[PAD]"] == 0
    assert tok.vocab["[UNK]"] == 1
    assert tok.vocab["[CLS]"] == 2
    assert tok.vocab["[SEP]"] == 3
    assert tok.vocab["[EQ]"] == 4


def test_scientific_tokenizer_encode_decode():
    tok = ScientificTokenizer()
    tokens = tok.encode("quantum mechanics")
    assert len(tokens) == 2
    assert tok.vocab["quantum"] >= 5

    text = tok.decode(tokens)
    assert text == "quantum mechanics"


def test_scientific_tokenizer_batch():
    tok = ScientificTokenizer()
    batch = tok.encode_batch(["text one", "text two"])
    assert isinstance(batch, list)
    assert len(batch) == 2
    assert isinstance(batch[0], list)


def test_aether_scientist_properties_and_context():
    config = AetherConfig(domains=["physics"])
    scientist = AetherScientist(config)
    assert isinstance(scientist.active_domains, list)
    assert scientist.device_status == config.device

    with AetherScientist(config) as s:
        assert s._is_active is True
    assert s._is_active is False


def test_aether_scientist_methods(monkeypatch):
    monkeypatch.setattr(
        "aether_scientist.core.model.InferenceEngine.generate",
        lambda **kwargs: {
            "text": "Simulated quantum analysis result.",
            "model": kwargs.get("model_name", "distilgpt2"),
            "tokens_generated": 4,
            "elapsed_seconds": 0.005,
        },
    )
    config = AetherConfig(domains=["physics"])
    scientist = AetherScientist(config)

    result = scientist.analyze("quantum entanglement research")
    assert "domain" in result
    assert "query_tokens" in result
    assert "analysis_result" in result
    assert result["analysis_result"]["generated_text"] == "Simulated quantum analysis result."

    result = scientist.synthesize_papers(["paper1", "paper2"], domain="physics")
    assert "summary" in result

    result = scientist.generate_hypothesis(["obs1"], domain="physics")
    assert "hypothesis" in result

    result = scientist.design_experiment("test hypothesis", domain="physics")
    assert "experiment_design" in result

    with pytest.raises(ValueError):
        scientist.generate_hypothesis(["obs1"], domain="unknown_domain")

    with pytest.raises(ValueError):
        scientist.design_experiment("test", domain="unknown_domain")


def test_inference_engine_lazy_load():
    InferenceEngine.reset()
    assert not InferenceEngine.is_loaded()


def test_inference_engine_generate_mock(monkeypatch):
    InferenceEngine.reset()

    class FakePipeline:
        def __call__(self, prompt, **kwargs):
            return [{"generated_text": " Entropy is a measure of disorder."}]

    monkeypatch.setattr(
        InferenceEngine, "get", classmethod(lambda cls, model_name: FakePipeline())
    )

    res = InferenceEngine.generate("What is entropy?", model_name="test-model", max_new_tokens=10)
    assert res["text"] == "Entropy is a measure of disorder."
    assert res["model"] == "test-model"
    assert res["tokens_generated"] == 6
    assert "elapsed_seconds" in res


def test_inference_engine_missing_transformers(monkeypatch):
    InferenceEngine.reset()
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("No transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="transformers is required for inference"):
        InferenceEngine.get("some-model")
