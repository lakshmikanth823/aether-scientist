"""AetherScientist Streamlit application entry point."""

import streamlit as st

from aether_scientist.agent.pipeline import ResearchAgent
from aether_scientist.core.config import AetherConfig
from aether_scientist.multimodal.captioner import CaptionEngine
from aether_scientist.retrieval.engine import RAGEngine
from ui.views.chat import render_chat_view
from ui.views.knowledge import render_knowledge_view
from ui.views.research import render_research_view
from ui.views.vision import render_vision_view

st.set_page_config(page_title="AetherScientist", page_icon="🔬", layout="wide")


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
    st.sidebar.title("🔬 AetherScientist")
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

