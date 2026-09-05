"""Interactive chat view with global profile routing and model badges."""

import streamlit as st

from aether_scientist.core.inference import InferenceEngine
from aether_scientist.core.profiles import resolve


def render_chat_view() -> None:
    """Render direct chat interface using global active profile."""
    profile = st.session_state.get("profile", "fast")
    res = resolve(profile)
    st.header("💬 Scientific Chat")
    st.caption(f"Active Profile: `{res.name}` · `{res.model}`")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("badge"):
                st.caption(msg["badge"])

    if prompt := st.chat_input("Ask a scientific question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        from ui.app import get_model_badge

        badge = get_model_badge(profile)
        with st.chat_message("assistant"):
            stream_gen = InferenceEngine.stream(prompt=prompt, profile=profile)
            response = st.write_stream(stream_gen)
            st.caption(badge)

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "badge": badge}
        )
