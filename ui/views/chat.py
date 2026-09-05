"""Interactive chat view with profile selection."""

import streamlit as st

from aether_scientist.core.inference import InferenceEngine
from aether_scientist.core.profiles import list_profiles


def render_chat_view() -> None:
    """Render direct chat interface with model profile selection."""
    st.header("💬 Scientific Chat")
    profiles = list_profiles()
    profile_names = [p["name"] for p in profiles]

    col1, col2 = st.columns([1, 2])
    with col1:
        default_idx = profile_names.index("fast") if "fast" in profile_names else 0
        selected = st.selectbox("Model Profile", profile_names, index=default_idx)
    with col2:
        info = next((p for p in profiles if p["name"] == selected), {})
        st.info(f"**Model**: `{info.get('model', '')}` — {info.get('description', '')}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a scientific question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream_gen = InferenceEngine.stream(prompt=prompt, profile=selected)
            response = st.write_stream(stream_gen)
        st.session_state.messages.append({"role": "assistant", "content": response})
