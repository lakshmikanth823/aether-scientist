import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CaptionEngine:
    """Singleton engine for generating descriptive captions for figures and diagrams."""

    _instance: "CaptionEngine | None" = None
    _pipeline: Any = None
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

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                from PIL import Image  # noqa: F401
                from transformers import pipeline

                self._pipeline = pipeline("image-to-text", model=self._model_name)
            except Exception as e:
                logger.warning(f"Failed to load vision captioning pipeline: {e}")
                self._pipeline = False
        return self._pipeline

    def caption(self, image_path: str | Path) -> str:
        """Generate a caption for the given image path, with graceful degradation."""
        p = Path(image_path)
        if not p.exists():
            return "[Image: file not found]"

        try:
            from PIL import Image
        except ImportError:
            return "[Image: vision dependencies not installed]"

        pipe = self._get_pipeline()
        if not pipe:
            return "[Image: vision dependencies not installed]"

        try:
            img = Image.open(p).convert("RGB")
            outputs = pipe(img)
            if outputs and isinstance(outputs, list) and "generated_text" in outputs[0]:
                cap = str(outputs[0]["generated_text"]).strip()
                return cap or "[Image: caption empty]"
            return "[Image: no description generated]"
        except Exception as e:
            logger.warning(f"Error generating caption for {p}: {e}")
            return f"[Image: captioning failed - {e}]"
