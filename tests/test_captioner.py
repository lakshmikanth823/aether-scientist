import sys
from unittest.mock import MagicMock, patch

from aether_scientist.multimodal.captioner import CaptionEngine


def test_caption_missing_file():
    CaptionEngine.reset()
    engine = CaptionEngine()
    res = engine.caption("non_existent_figure.png")
    assert res == "[Image: file not found]"


def test_caption_missing_pil(tmp_path):
    img_file = tmp_path / "fig.png"
    img_file.write_bytes(b"dummy image data")
    CaptionEngine.reset()
    engine = CaptionEngine()
    with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
        res = engine.caption(img_file)
        assert "vision dependencies not installed" in res


def test_caption_pipeline_failure(tmp_path):
    img_file = tmp_path / "fig.png"
    img_file.write_bytes(b"dummy image data")
    CaptionEngine.reset()
    engine = CaptionEngine()
    mock_pil_img = MagicMock()
    with (
        patch("PIL.Image.open", return_value=mock_pil_img),
        patch.object(engine, "_get_pipeline", return_value=False),
    ):
        res = engine.caption(img_file)
        assert "vision dependencies not installed" in res


def test_caption_success(tmp_path):
    img_file = tmp_path / "fig.png"
    img_file.write_bytes(b"dummy image data")
    CaptionEngine.reset()
    engine = CaptionEngine()

    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"generated_text": "a diagram showing transformer attention flow"}]
    mock_pil = MagicMock()

    with (
        patch("PIL.Image.open", return_value=mock_pil),
        patch.object(engine, "_get_pipeline", return_value=mock_pipe),
    ):
        res = engine.caption(img_file)
        assert res == "a diagram showing transformer attention flow"
