"""Gradio interface for Hugging Face Spaces deployment of AetherScientist."""

from collections.abc import Generator
from pathlib import Path
from typing import Any

try:
    import spaces  # type: ignore

    _HAS_SPACES = True
except ImportError:
    _HAS_SPACES = False

try:
    import gradio as gr  # type: ignore
except ImportError:
    gr = None

from aether_scientist.agent.pipeline import ResearchAgent
from aether_scientist.core.config import AetherConfig
from aether_scientist.core.inference import InferenceEngine, _detect_device
from aether_scientist.core.profiles import list_profiles
from aether_scientist.multimodal.captioner import CaptionEngine
from aether_scientist.retrieval.engine import RAGEngine

_CONFIG = AetherConfig()
_RAG = RAGEngine(config=_CONFIG.rag)
_AGENT = ResearchAgent(config=_CONFIG)
_CAPTIONER = CaptionEngine()


def _gpu(duration: int = 60) -> Any:
    return spaces.GPU(duration=duration) if _HAS_SPACES else (lambda f: f)


def chat_response(
    message: str, history: list[dict[str, str]], profile: str, use_rag: bool
) -> str:
    """Handle chat with optional RAG grounding."""
    if use_rag:
        ans = _RAG.grounded_generate(message, profile=profile)
        sources_text = ""
        if ans.sources:
            sources_text = "\n\n**Sources:**\n" + "\n".join(
                f"- [{s.score:.2f}] {s.title or s.doc_id}: {s.snippet[:120]}..."
                for s in ans.sources
            )
        return ans.answer + sources_text
    engine = InferenceEngine.get(profile_name=profile)
    res = engine.generate(message)
    return res.get("text", "")


def ingest_files(file_paths: list[str] | None) -> str:
    """Index uploaded documents into RAG vector store."""
    if not file_paths:
        return "No files uploaded."
    stats = _RAG.index(file_paths)
    return (
        f"Indexed {stats.docs} docs ({stats.chunks} chunks) in "
        f"{stats.elapsed_seconds}s! Total chunks: {len(_RAG.store)}"
    )


def load_demo_corpus() -> str:
    """Load open-access demo abstracts into RAG vector store."""
    demo_p = Path("data/demo_abstracts.txt")
    if not demo_p.exists():
        return "Demo corpus file not found."
    stats = _RAG.index([demo_p.as_posix()])
    return (
        f"Indexed demo corpus: {stats.docs} doc ({stats.chunks} chunks) in "
        f"{stats.elapsed_seconds}s! Total chunks: {len(_RAG.store)}"
    )


def run_research(topic: str, profile: str) -> Generator[str, None, None]:
    """Run multi-step autonomous research agent."""
    if not topic.strip():
        yield "Please enter a research question."
        return
    report_accum = f"# Research Survey: {topic}\n\n*Running agent...*\n"
    yield report_accum
    res = _AGENT.research(topic, profile=profile)
    yield res.report_md


@_gpu(duration=60)
def analyze_diagram(image_path: str | None) -> str:
    """Generate grounded scientific caption for a figure or diagram."""
    if not image_path:
        return "Please upload an image."
    return _CAPTIONER.caption(image_path)


def create_gradio_app() -> Any:
    """Construct and return the Gradio Blocks UI."""
    if gr is None:
        raise RuntimeError("Gradio is not installed.")

    profile_choices = [p["name"] for p in list_profiles()]
    default_p = "fast" if "fast" in profile_choices else profile_choices[0]

    theme = gr.themes.Soft(primary_hue="blue")
    with gr.Blocks(title="AetherScientist 🔬", theme=theme) as demo:
        gr.Markdown(
            "# 🔬 AetherScientist\nDomain-specialized scientific research LLM with "
            f"RAG, citations & vision. Active compute: `{_detect_device()}`"
        )

        with gr.Tabs():
            with gr.TabItem("💬 Chat & Grounded QA"):
                with gr.Row():
                    p_drop = gr.Dropdown(
                        label="Model Profile", choices=profile_choices, value=default_p
                    )
                    rag_toggle = gr.Checkbox(label="Enable RAG Grounding", value=True)
                gr.ChatInterface(
                    fn=chat_response,
                    additional_inputs=[p_drop, rag_toggle],
                    type="messages",
                )

            with gr.TabItem("📚 Knowledge Base"):
                gr.Markdown("### Ingest Scientific Literature")
                with gr.Row():
                    f_upload = gr.File(
                        label="Upload Papers (PDF, TXT, MD)", file_count="multiple"
                    )
                    f_stat = gr.Textbox(
                        label="Store Status",
                        value=f"Chunks in store: {len(_RAG.store)}",
                        interactive=False,
                    )
                with gr.Row():
                    btn_ingest = gr.Button("Index Uploaded Files", variant="primary")
                    btn_demo = gr.Button("Load Demo Corpus", variant="secondary")
                btn_ingest.click(
                    fn=ingest_files, inputs=[f_upload], outputs=[f_stat]
                ).then(fn=lambda: f"Chunks in store: {len(_RAG.store)}", outputs=[f_stat])
                btn_demo.click(fn=load_demo_corpus, outputs=[f_stat]).then(
                    fn=lambda: f"Chunks in store: {len(_RAG.store)}", outputs=[f_stat]
                )

            with gr.TabItem("🔬 Research Agent"):
                gr.Markdown("### Autonomous Scientific Literature Survey")
                with gr.Row():
                    r_topic = gr.Textbox(
                        label="Research Topic",
                        placeholder=(
                            "e.g. How does the attention mechanism scale with sequence length?"
                        ),
                        scale=4,
                    )
                    r_prof = gr.Dropdown(
                        label="Profile",
                        choices=profile_choices,
                        value=default_p,
                        scale=1,
                    )
                r_btn = gr.Button("Start Autonomous Research", variant="primary")
                r_out = gr.Markdown(label="Generated Research Report")
                r_btn.click(fn=run_research, inputs=[r_topic, r_prof], outputs=[r_out])

            with gr.TabItem("🖼️ Vision & Diagrams"):
                gr.Markdown("### Figure & Architecture Diagram Understanding")
                with gr.Row():
                    v_img = gr.Image(type="filepath", label="Upload Scientific Diagram")
                    v_cap = gr.Textbox(label="Generated Caption", interactive=False)
                v_btn = gr.Button("Analyze Diagram", variant="primary")
                v_btn.click(fn=analyze_diagram, inputs=[v_img], outputs=[v_cap])

    return demo
