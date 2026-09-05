"""AetherScientist Streamlit application entry point."""

import streamlit as st

from aether_scientist.agent.pipeline import ResearchAgent
from aether_scientist.core.config import AetherConfig
from aether_scientist.core.inference import _detect_device
from aether_scientist.core.profiles import list_profiles, resolve
from aether_scientist.multimodal.captioner import CaptionEngine
from aether_scientist.retrieval.engine import RAGEngine
from ui.views.chat import render_chat_view
from ui.views.knowledge import render_knowledge_view
from ui.views.research import render_research_view
from ui.views.vision import render_vision_view

st.set_page_config(page_title="AetherScientist", page_icon="🔬", layout="wide")


def get_model_badge(profile_name: str | None = None) -> str:
    """Format model profile and device badge for UI outputs."""
    p_name = profile_name or st.session_state.get("profile", "fast")
    res = resolve(p_name)
    return f"{res.name} · {res.model} · {_detect_device()}"


@st.cache_resource
def get_config() -> AetherConfig:
    return AetherConfig()


@st.cache_resource
def get_rag_engine() -> RAGEngine:
    return RAGEngine(config=get_config().rag)


@st.cache_resource
def get_research_agent() -> ResearchAgent:
    return ResearchAgent(config=get_config())


@st.cache_resource
def get_caption_engine() -> CaptionEngine:
    return CaptionEngine()


def main() -> None:
    if "profile" not in st.session_state:
        st.session_state.profile = "fast"

    st.sidebar.title("🔬 AetherScientist")

    profiles = list_profiles()
    p_names = [p["name"] for p in profiles]
    def_idx = p_names.index("fast") if "fast" in p_names else 0
    cur_idx = (
        p_names.index(st.session_state.profile)
        if st.session_state.profile in p_names
        else def_idx
    )

    selected_prof = st.sidebar.selectbox("Active Profile", p_names, index=cur_idx)
    st.session_state.profile = selected_prof
    st.sidebar.caption(f"⚙️ {get_model_badge(selected_prof)}")

    choice = st.sidebar.radio(
        "Navigation", ["Chat", "Knowledge Base", "Research Agent", "Vision Tool"]
    )
    if choice == "Chat":
        render_chat_view()
    elif choice == "Knowledge Base":
        render_knowledge_view(get_rag_engine())
    elif choice == "Research Agent":
        render_research_view(get_research_agent())
    elif choice == "Vision Tool":
        render_vision_view(get_caption_engine())


if __name__ == "__main__":
    main()
