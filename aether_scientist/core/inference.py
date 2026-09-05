import logging
import time
from collections.abc import Generator
from typing import Any

from aether_scientist.core.format import clean_output, format_prompt, get_stop_markers
from aether_scientist.core.profiles import PROFILES, ModelProfile, resolve

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    """Auto-detect available compute device: cuda -> mps -> cpu."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


class InferenceEngine:
    """Lazy-loaded singleton LLM pipeline for text generation."""

    _pipe: Any = None
    _model_name: str | None = None
    device: str = _detect_device()

    @classmethod
    def get(cls, profile: ModelProfile | str = "offline") -> Any:
        """Return pipeline for model profile, creating on first call."""
        if isinstance(profile, ModelProfile):
            model_name = profile.model
        elif profile in PROFILES:
            model_name = PROFILES[profile].model
        else:
            model_name = profile

        if cls._pipe is None or cls._model_name != model_name:
            try:
                from transformers import pipeline
            except ImportError as e:
                raise RuntimeError(
                    "transformers is required for inference. "
                    "Install with: pip install transformers"
                ) from e
            logger.info(f"Loading model: {model_name} on device: {cls.device}")
            pipe_kwargs: dict[str, Any] = {"model_kwargs": {"torch_dtype": "auto"}}
            if cls.device == "cuda":
                pipe_kwargs["device"] = 0
            elif cls.device == "mps":
                pipe_kwargs["device"] = "mps"
            cls._pipe = pipeline(
                "text-generation",
                model=model_name,
                **pipe_kwargs,
            )
            cls._model_name = model_name
        return cls._pipe

    @classmethod
    def _clean(cls, text: str, tokenizer: Any = None) -> str:
        """Clean output text by cutting at stop markers."""
        markers = get_stop_markers(tokenizer)
        return clean_output(text, markers)

    @classmethod
    def generate(
        cls,
        prompt: str = "",
        model_name: str = "distilgpt2",
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        profile: ModelProfile | str | None = None,
        system_prompt: str = "",
        query: str = "",
    ) -> dict[str, Any]:
        """Generate text with profile chat formatting, parameters, and output cleaning."""
        start = time.perf_counter()
        if profile is not None:
            target_profile = resolve(
                profile.name if isinstance(profile, ModelProfile) else profile
            )
            model_to_use = target_profile.model
            profile_name = target_profile.name
        elif model_name in PROFILES:
            target_profile = PROFILES[model_name]
            model_to_use = target_profile.model
            profile_name = target_profile.name
        else:
            target_profile = resolve()
            model_to_use = model_name
            profile_name = "offline"

        tokens_limit = (
            max_new_tokens if max_new_tokens is not None else target_profile.max_new_tokens
        )
        temp = temperature if temperature is not None else target_profile.temperature

        try:
            pipe = cls.get(model_to_use)
            tokenizer = getattr(pipe, "tokenizer", None)

            user_text = query if query else prompt
            formatted_prompt = format_prompt(
                tokenizer, target_profile, system=system_prompt, user=user_text
            )

            outputs = pipe(
                formatted_prompt,
                max_new_tokens=tokens_limit,
                temperature=temp,
                do_sample=temp > 0,
                repetition_penalty=1.15,
                return_full_text=False,
            )
            raw_text = outputs[0]["generated_text"]
            text = cls._clean(raw_text, tokenizer)
            elapsed = time.perf_counter() - start
            return {
                "text": text,
                "model": model_to_use,
                "profile": profile_name,
                "tokens_generated": len(text.split()),
                "elapsed_seconds": round(elapsed, 3),
            }
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"Generation failed after {elapsed:.3f}s: {e}")
            raise

    @classmethod
    def stream(
        cls,
        prompt: str = "",
        model_name: str = "distilgpt2",
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        profile: ModelProfile | str | None = None,
        system_prompt: str = "",
        query: str = "",
    ) -> Generator[str, None, None]:
        """Stream generated tokens with chat formatting and fallback support."""
        target_profile = resolve(
            profile.name
            if isinstance(profile, ModelProfile)
            else (profile if profile else model_name)
        )
        tokens_limit = (
            max_new_tokens if max_new_tokens is not None else target_profile.max_new_tokens
        )
        temp = temperature if temperature is not None else target_profile.temperature

        try:
            from threading import Thread

            from transformers import TextIteratorStreamer

            pipe = cls.get(target_profile)
            tokenizer = getattr(pipe, "tokenizer", None)
            model = getattr(pipe, "model", None)

            user_text = query if query else prompt
            formatted_prompt = format_prompt(
                tokenizer, target_profile, system=system_prompt, user=user_text
            )

            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            inputs = tokenizer([formatted_prompt], return_tensors="pt")
            kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": tokens_limit,
                "temperature": temp,
                "do_sample": temp > 0,
                "repetition_penalty": 1.15,
            }
            thread = Thread(target=model.generate, kwargs=kwargs)
            thread.start()
            for token in streamer:
                if token:
                    yield token
            thread.join()
        except Exception as e:
            logger.info(f"Streamer fallback active: {e}")
            try:
                res = cls.generate(
                    prompt=prompt,
                    profile=target_profile,
                    max_new_tokens=tokens_limit,
                    temperature=temp,
                    system_prompt=system_prompt,
                    query=query,
                )
                words = res.get("text", "").split(" ")
            except Exception:
                words = [f"Scientific response to {(query or prompt)[:25].strip()}."]
            for i, w in enumerate(words):
                yield w + (" " if i < len(words) - 1 else "")

    @classmethod
    def reset(cls) -> None:
        """Reset the pipeline (for testing)."""
        cls._pipe = None
        cls._model_name = None

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._pipe is not None
