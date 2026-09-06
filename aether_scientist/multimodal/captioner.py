import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CaptionEngine:
    """Singleton engine for generating descriptive captions for figures and diagrams."""

    _instance: "CaptionEngine | None" = None
    _pipeline: Any = None
    _pipeline_error: str | None = None
    _model_name: str = "Salesforce/blip-image-captioning-base"

    def __new__(cls) -> "CaptionEngine":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (useful in testing)."""
        cls._instance = None
        cls._pipeline = None
        cls._pipeline_error = None

    def _get_pipeline(self, raise_on_error: bool = False) -> Any:
        if self._pipeline is None:
            try:
                import PIL.Image  # noqa: F401
                from transformers import pipeline
            except ImportError as e:
                err = "vision dependencies not installed (pip install aether-scientist[vision])"
                self._pipeline_error = err
                self._pipeline = False
                logger.warning(f"Failed to load vision dependencies: {e}")
                if raise_on_error:
                    raise RuntimeError(err) from e
                return False

            try:
                self._pipeline = pipeline("image-to-text", model=self._model_name)
            except Exception as e:
                err = "vision model failed to load (network/HF access required)"
                self._pipeline_error = err
                self._pipeline = False
                logger.warning(f"Failed to load vision captioning pipeline: {e}")
                if raise_on_error:
                    raise RuntimeError(err) from e
                return False

        if not self._pipeline and raise_on_error:
            raise RuntimeError(
                self._pipeline_error
                or "vision model failed to load (network/HF access required)"
            )
        return self._pipeline

    def caption(self, image_path: str | Path, raise_on_error: bool = False) -> str:
        """Generate a caption for the given image path, with graceful degradation."""
        p = Path(image_path)
        if not p.exists():
            if raise_on_error:
                raise FileNotFoundError(f"File not found: {p}")
            return "[Image: file not found]"

        try:
            from PIL import Image
        except ImportError as e:
            err = "vision dependencies not installed (pip install aether-scientist[vision])"
            if raise_on_error:
                raise RuntimeError(err) from e
            return f"[Image: {err}]"

        pipe = self._get_pipeline(raise_on_error=raise_on_error)
        if not pipe:
            err = self._pipeline_error or "vision model failed to load (network/HF access required)"
            if raise_on_error:
                raise RuntimeError(err)
            return f"[Image: {err}]"

        try:
            img = Image.open(p).convert("RGB")
            outputs = pipe(img)
            if outputs and isinstance(outputs, list) and "generated_text" in outputs[0]:
                cap = str(outputs[0]["generated_text"]).strip()
                return cap or "[Image: caption empty]"
            return "[Image: no description generated]"
        except Exception as e:
            logger.warning(f"Error generating caption for {p}: {e}")
            if raise_on_error:
                raise RuntimeError(f"captioning failed - {e}") from e
            return f"[Image: captioning failed - {e}]"
