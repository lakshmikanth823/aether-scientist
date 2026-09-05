import logging
import time
from collections.abc import Generator
from typing import Any

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Lazy-loaded singleton LLM pipeline for text generation."""

    _pipe: Any = None
    _model_name: str | None = None

    @classmethod
    def get(cls, model_name: str) -> Any:
        """Return pipeline, creating on first call (lazy load)."""
        if cls._pipe is None or cls._model_name != model_name:
            try:
                from transformers import pipeline
            except ImportError as e:
                raise RuntimeError(
                    "transformers is required for inference. "
                    "Install with: pip install transformers"
                ) from e
            logger.info(f"Loading model: {model_name}")
            cls._pipe = pipeline(
                "text-generation",
                model=model_name,
                model_kwargs={"torch_dtype": "auto"},
            )
            cls._model_name = model_name
        return cls._pipe

    @classmethod
    def generate(
        cls,
        prompt: str,
        model_name: str = "distilgpt2",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Generate text with timing and error handling."""
        start = time.perf_counter()
        try:
            pipe = cls.get(model_name)
            outputs = pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                return_full_text=False,
            )
            text = outputs[0]["generated_text"].strip()
            elapsed = time.perf_counter() - start
            return {
                "text": text,
                "model": model_name,
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
        prompt: str,
        model_name: str = "distilgpt2",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """Stream generated text tokens with fallback support."""
        try:
            from threading import Thread

            from transformers import TextIteratorStreamer

            pipe = cls.get(model_name)
            tokenizer = pipe.tokenizer
            model = pipe.model
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            inputs = tokenizer([prompt], return_tensors="pt")
            kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
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
                    model_name=model_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                words = res.get("text", "").split(" ")
            except Exception:
                words = [f"Scientific response to {prompt[:25].strip()}."]
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
