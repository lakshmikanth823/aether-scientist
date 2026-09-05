import sys
from unittest.mock import MagicMock, patch

from aether_scientist.retrieval.images import ImageAsset, extract_images


def test_extract_images_non_pdf(tmp_path):
    txt_file = tmp_path / "paper.txt"
    txt_file.write_text("plain text", encoding="utf-8")
    assert extract_images(txt_file) == []


def test_extract_images_missing_file():
    assert extract_images("non_existent_file.pdf") == []


def test_extract_images_missing_fitz(tmp_path):
    pdf_file = tmp_path / "paper.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 mock")
    with patch.dict(sys.modules, {"fitz": None}):
        assert extract_images(pdf_file) == []


def test_extract_images_mocked_success(tmp_path):
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy content")

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 2

    mock_page_1 = MagicMock()
    mock_page_1.get_images.return_value = [[101, 0, 100, 100, 8, "DeviceRGB"]]

    mock_page_2 = MagicMock()
    mock_page_2.get_images.return_value = []

    mock_doc.__getitem__.side_effect = lambda idx: mock_page_1 if idx == 0 else mock_page_2
    mock_doc.extract_image.return_value = {"ext": "png", "image": b"\x89PNG\r\n\x1a\nfakeimage"}

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc

    with patch.dict(sys.modules, {"fitz": mock_fitz}):
        cache_dir = tmp_path / ".cache"
        assets = extract_images(pdf_file, cache_dir=cache_dir)
        assert len(assets) == 1
        asset = assets[0]
        assert isinstance(asset, ImageAsset)
        assert asset.page_num == 1
        assert asset.image_path.endswith(".png")
        mock_doc.close.assert_called_once()
