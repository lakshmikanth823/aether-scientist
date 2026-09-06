from aether_scientist.core.format import (
    STATIC_MARKERS,
    clean_output,
    format_prompt,
    get_stop_markers,
)
from aether_scientist.core.inference import InferenceEngine, _detect_device
from aether_scientist.core.profiles import get_profile


class FakeTokenizer:
    """Mock tokenizer providing chat_template, apply_chat_template, and dynamic tokens."""

    def __init__(self, has_template: bool = True):
        self.chat_template = (
            "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
            if has_template
            else None
        )
        self.eos_token = "[DYNAMIC_EOS_TOKEN]"
        self.additional_special_tokens = ["[DYNAMIC_STOP_A]", "[DYNAMIC_STOP_B]"]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        lines = [f"<role:{m['role']}>{m['content']}</role>" for m in messages]
        if add_generation_prompt:
            lines.append("<generation_prompt>")
        return "".join(lines)


def test_format_prompt_chat_profile_with_template():
    tok = FakeTokenizer(has_template=True)
    prof = get_profile("fast")  # chat = True
    formatted = format_prompt(
        tok, prof, system="You are a physicist.", user="What is entropy?"
    )
    assert "<role:system>You are a physicist.</role>" in formatted
    assert "<role:user>What is entropy?</role>" in formatted
    assert "<generation_prompt>" in formatted


def test_format_prompt_offline_profile_raw():
    tok = FakeTokenizer(has_template=True)
    prof = get_profile("offline")  # chat = False
    formatted = format_prompt(
        tok, prof, system="You are a chemist.", user="What is a covalent bond?"
    )
    assert "<role:" not in formatted
    assert formatted == "You are a chemist.\n\nQuestion: What is a covalent bond?\n\nAnswer:"


def test_format_prompt_chat_profile_without_template():
    tok = FakeTokenizer(has_template=False)
    prof = get_profile("fast")
    formatted = format_prompt(
        tok, prof, system="You are a biologist.", user="What is DNA?"
    )
    assert "<role:" not in formatted
    assert formatted == "You are a biologist.\n\nQuestion: What is DNA?\n\nAnswer:"


def test_clean_output_cuts_at_static_markers():
    dirty = "Light travels at 300,000 km/s.\n\n### Assistant: Let me add more irrelevant info."
    cleaned = clean_output(dirty, STATIC_MARKERS)
    assert cleaned == "Light travels at 300,000 km/s."


def test_clean_output_cuts_at_dynamic_eos_marker():
    tok = FakeTokenizer()
    markers = get_stop_markers(tok)
    assert "[DYNAMIC_EOS_TOKEN]" in markers
    assert "[DYNAMIC_STOP_A]" in markers

    dirty = "E=mc^2 explains mass-energy equivalence.[DYNAMIC_EOS_TOKEN]Extra junk tokens."
    cleaned = clean_output(dirty, markers)
    assert cleaned == "E=mc^2 explains mass-energy equivalence."


def test_clean_output_no_markers_just_strips():
    text = "   Quantum tunneling is a probabilistic phenomenon.   "
    cleaned = clean_output(text, STATIC_MARKERS)
    assert cleaned == "Quantum tunneling is a probabilistic phenomenon."


def test_device_detection_fallback(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("Torch not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert _detect_device() == "cpu"


def test_inference_engine_generate_with_profile(monkeypatch):
    InferenceEngine.reset()

    class FakePipeline:
        tokenizer = FakeTokenizer()

        def __call__(self, prompt, **kwargs):
            return [{"generated_text": f"Output for {prompt[:20]}\n\n### Extra commentary"}]

    monkeypatch.setattr(
        InferenceEngine, "get", classmethod(lambda cls, model_name: FakePipeline())
    )

    res = InferenceEngine.generate(query="Explain black holes", profile="fast")
    assert res["profile"] == "fast"
    assert res["model"] == "HuggingFaceTB/SmolLM2-360M-Instruct"
    assert "### Extra commentary" not in res["text"]
    assert "Output for" in res["text"]


def test_context_window_truncation_5000_words(monkeypatch):
    """5000-word prompt -> truncated without error."""
    InferenceEngine.reset()

    class Config:
        max_position_embeddings = 512

    class Model:
        config = Config()

    class PipelineWithTokenizer:
        model = Model()

        class Tokenizer:
            @staticmethod
            def encode(text):
                return list(range(len(text.split())))

            @staticmethod
            def decode(tokens, **kwargs):
                return " ".join(f"w{i}" for i in tokens)

        tokenizer = Tokenizer()

        def __call__(self, prompt, **kwargs):
            words = prompt.split()
            assert len(words) <= 512
            return [{"generated_text": "Safe response after truncation."}]

    monkeypatch.setattr(
        InferenceEngine, "get", classmethod(lambda cls, model_name: PipelineWithTokenizer())
    )

    long_prompt = "word " * 5000
    res = InferenceEngine.generate(prompt=long_prompt, profile="fast")
    assert res is not None
    assert "Safe response" in res["text"]


def test_index_error_never_escapes_generate(monkeypatch):
    """IndexError must be trapped and return clean truncated notice."""
    InferenceEngine.reset()

    class BuggyPipeline:
        tokenizer = None

        def __call__(self, prompt, **kwargs):
            raise IndexError("position_ids index out of range")

    monkeypatch.setattr(
        InferenceEngine, "get", classmethod(lambda cls, model_name: BuggyPipeline())
    )

    res = InferenceEngine.generate(prompt="hello", profile="fast")
    assert "[Context length exceeded" in res["text"]
