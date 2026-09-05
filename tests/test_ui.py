"""Offline unit tests for Streamlit interactive web interface."""

import io
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from aether_scientist.agent.pipeline import ResearchResult
from aether_scientist.agent.state import ResearchState
from aether_scientist.retrieval.engine import GroundedAnswer, IndexStats, Source
from ui.app import get_caption_engine, get_config, get_rag_engine, get_research_agent

APP_PATH = str(Path(__file__).parent.parent / "ui" / "app.py")


def test_app_shell_navigation() -> None:
    """Verify app shell loads, renders sidebar, and navigates between views."""
    at = AppTest.from_file(APP_PATH, default_timeout=10)
    at.run()
    assert not at.exception
    assert len(at.sidebar.radio) > 0
    assert len(at.sidebar.selectbox) > 0

    # Verify sidebar profile selector defaults to 'fast'
    prof_select = at.sidebar.selectbox[0]
    assert prof_select.value == "fast"
    assert "fast" in prof_select.options

    # Update profile in sidebar and check session state
    prof_select.set_value("balanced").run()
    assert not at.exception
    assert at.session_state["profile"] == "balanced"

    options = at.sidebar.radio[0].options
    assert "Chat" in options
    assert "Knowledge Base" in options
    assert "Research Agent" in options
    assert "Vision Tool" in options



def test_engine_caching() -> None:
    """Verify engine singleton factory functions return valid objects."""
    cfg = get_config()
    assert cfg is not None
    rag = get_rag_engine()
    assert rag is not None
    agent = get_research_agent()
    assert agent is not None
    cap = get_caption_engine()
    assert cap is not None


def test_chat_view_streaming() -> None:
    """Verify direct chat view accepts input and renders streamed tokens."""
    mock_tokens = ["Quantum ", "mechanics ", "describes ", "matter."]

    with patch(
        "aether_scientist.core.inference.InferenceEngine.stream",
        return_value=iter(mock_tokens),
    ):
        at = AppTest.from_file(APP_PATH, default_timeout=10)
        at.run()
        assert len(at.chat_input) > 0

        at.chat_input[0].set_value("Explain quantum mechanics").run()
        assert not at.exception
        assert len(at.chat_message) >= 2
        user_msg = at.chat_message[0]
        asst_msg = at.chat_message[1]
        assert "quantum mechanics" in user_msg.markdown[0].value.lower()
        assert "Quantum mechanics describes matter." in asst_msg.markdown[0].value


def test_knowledge_view_management() -> None:
    """Verify knowledge base view renders, supports clear button, and query."""
    mock_ans = GroundedAnswer(
        answer="Attention allows focusing on key tokens [1].",
        sources=[
            Source(
                doc_id="p1",
                title="Attention Paper",
                location="c0",
                snippet="Self-attention",
                score=0.92,
            )
        ],
        confidence=0.88,
    )

    at = AppTest.from_file(APP_PATH, default_timeout=10)
    at.run()
    at.sidebar.radio[0].set_value("Knowledge Base").run()
    assert not at.exception
    assert len(at.file_uploader) > 0

    clear_btn = [b for b in at.button if "Clear Index" in b.label]
    assert len(clear_btn) > 0
    clear_btn[0].click().run()
    assert not at.exception

    with patch(
        "aether_scientist.retrieval.engine.RAGEngine.grounded_generate",
        return_value=mock_ans,
    ) as mock_gen:
        at.chat_input[0].set_value("What is attention?").run()
        assert not at.exception
        assert any("Attention allows focusing" in m.value for m in at.markdown)
        mock_gen.assert_called_once()
        assert mock_gen.call_args[1].get("profile") == "fast"



def test_knowledge_view_file_upload() -> None:
    """Verify file upload and indexing triggers in knowledge base view."""
    at = AppTest.from_file(APP_PATH, default_timeout=10)
    at.run()
    at.sidebar.radio[0].set_value("Knowledge Base").run()
    at.file_uploader[0].upload("sample.txt", b"Scientific text content.").run()
    assert not at.exception

    idx_btn = [b for b in at.button if "Index Uploaded Files" in b.label]
    assert len(idx_btn) > 0

    mock_stats = IndexStats(
        docs=1, chunks=2, elapsed_seconds=0.04, images_extracted=0, captions_generated=0
    )
    with patch(
        "aether_scientist.retrieval.engine.RAGEngine.index",
        return_value=mock_stats,
    ):
        idx_btn[0].click().run()
        assert not at.exception
        assert len(at.success) > 0
        assert "Indexed 1 docs" in at.success[0].value


def test_research_agent_dashboard() -> None:
    """Verify research agent executes streamed pipeline and renders report."""
    mock_res = ResearchResult(
        question="How does photosynthesis work?",
        report_md="# Photosynthesis Report\n\nConverts light energy into chemical energy.",
        state=ResearchState(question="How does photosynthesis work?"),
        sources=[{"title": "Plant Biology", "source": "bio.pdf", "score": 0.95}],
        confidence=0.90,
        coverage={},
    )

    def _stream_gen(*args, **kwargs):
        yield {"event": "step", "kind": "plan", "description": "Analyzing question"}
        yield {"event": "step", "kind": "retrieve", "description": "Retrieving literature"}
        yield {"event": "result", "result_object": mock_res}

    with patch(
        "aether_scientist.agent.pipeline.ResearchAgent.stream",
        side_effect=_stream_gen,
    ):
        at = AppTest.from_file(APP_PATH, default_timeout=10)
        at.run()
        at.sidebar.radio[0].set_value("Research Agent").run()
        assert not at.exception
        assert len(at.text_input) > 0

        at.text_input[0].set_value("How does photosynthesis work?").run()
        gen_btn = [b for b in at.button if "Generate Report" in b.label]
        assert len(gen_btn) > 0
        gen_btn[0].click().run()
        assert not at.exception
        assert any("Photosynthesis Report" in m.value for m in at.markdown)


def test_vision_tool_view() -> None:
    """Verify vision view renders file uploader for figures."""
    at = AppTest.from_file(APP_PATH, default_timeout=10)
    at.run()
    at.sidebar.radio[0].set_value("Vision Tool").run()
    assert not at.exception
    assert len(at.file_uploader) > 0


def test_vision_tool_analyze_action() -> None:
    """Verify diagram upload and caption generation trigger."""
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(buf, format="PNG")
    valid_png = buf.getvalue()

    at = AppTest.from_file(APP_PATH, default_timeout=10)
    at.run()
    at.sidebar.radio[0].set_value("Vision Tool").run()
    at.file_uploader[0].upload("figure1.png", valid_png).run()
    assert not at.exception

    analyze_btn = [b for b in at.button if "Analyze" in b.label]
    assert len(analyze_btn) > 0

    with patch(
        "aether_scientist.multimodal.captioner.CaptionEngine.caption",
        return_value="A diagram illustrating convolutional neural network architecture.",
    ):
        analyze_btn[0].click().run()
        assert not at.exception
        assert len(at.success) > 0
        assert "convolutional neural network" in at.success[0].value
