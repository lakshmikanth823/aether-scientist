import json
import sys
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from aether_scientist.api.endpoints import create_app
from aether_scientist.cli import app
from aether_scientist.core.config import AetherConfig
from aether_scientist.retrieval.engine import RAGEngine
from aether_scientist.retrieval.images import ImageAsset
from aether_scientist.retrieval.ingest import Document, ingest_file


def test_ingest_file_pdf_with_figures(tmp_path):
    pdf_file = tmp_path / "paper_with_figs.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 mock")

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 1
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Figure 1 illustrates the neural architecture."
    mock_doc.__getitem__.return_value = mock_page
    mock_doc.metadata = {"title": "Multimodal Transformer Paper"}

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc

    mock_assets = [
        ImageAsset(doc_id="testdoc", page_num=1, image_path=str(tmp_path / "fig1.png"))
    ]

    with (
        patch.dict(sys.modules, {"fitz": mock_fitz}),
        patch("aether_scientist.retrieval.ingest.extract_images", return_value=mock_assets),
        patch(
            "aether_scientist.multimodal.captioner.CaptionEngine.caption",
            return_value="architecture diagram",
        ),
    ):
        docs = ingest_file(pdf_file)
        assert len(docs) == 1
        doc = docs[0]
        assert "[Figure 1: architecture diagram]" in doc.text
        assert len(doc.images) == 1
        assert len(doc.captions) == 1


def test_rag_engine_index_figure_captions(tmp_path):
    doc_text = "Detailed methodology.\n\n[Figure 1: electron microscopy of graphene crystals]"
    mock_doc = Document(
        doc_id="doc1",
        source=str(tmp_path / "graphene.pdf"),
        title="Graphene Paper",
        text=doc_text,
        images=[ImageAsset("doc1", 1, "p1.png")],
        captions=["electron microscopy of graphene crystals"],
    )

    rag = RAGEngine()
    with patch("aether_scientist.retrieval.engine.ingest_file", return_value=[mock_doc]):
        stats = rag.index([str(tmp_path / "graphene.pdf")])
        assert stats.images_extracted == 1
        assert stats.captions_generated == 1
        assert stats.docs == 1

        hits = rag.retrieve("electron microscopy graphene")
        assert len(hits) > 0
        assert "electron microscopy" in hits[0].chunk.text


def test_cli_vision_command(tmp_path):
    runner = CliRunner()
    img_file = tmp_path / "test_diagram.png"
    img_file.write_bytes(b"dummy")

    with patch(
        "aether_scientist.multimodal.captioner.CaptionEngine.caption",
        return_value="scatter plot of benchmark results",
    ):
        res = runner.invoke(app, ["vision", str(img_file), "--json"])
        assert res.exit_code == 0
        parsed = json.loads(res.output)
        assert parsed["caption"] == "scatter plot of benchmark results"


def test_api_post_vision_endpoint(tmp_path):
    client = TestClient(create_app(AetherConfig()))
    img_file = tmp_path / "test.png"
    img_file.write_bytes(b"png data")

    with patch(
        "aether_scientist.multimodal.captioner.CaptionEngine.caption",
        return_value="diagram of solar cells",
    ):
        files = {"file": ("test.png", b"fake png bytes", "image/png")}
        headers = {"X-API-Key": "dev-aether-key"}
        response = client.post("/vision", files=files, headers=headers)
        assert response.status_code == 200
        json_data = response.json()
        assert json_data["status"] == "success"
        assert json_data["data"]["caption"] == "diagram of solar cells"

        resp_path = client.post(f"/vision?image_path={img_file}", headers=headers)
        assert resp_path.status_code == 200
        assert resp_path.json()["data"]["caption"] == "diagram of solar cells"
