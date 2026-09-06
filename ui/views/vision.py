"""Vision and diagram understanding view using lightweight VLM captioning."""

from pathlib import Path

import streamlit as st

from aether_scientist.multimodal.captioner import CaptionEngine


def render_vision_view(caption_engine: CaptionEngine) -> None:
    """Render vision diagram inspection tool."""
    st.header("🖼️ Figure & Diagram Understanding")
    st.caption("Extract visual meaning and generate grounded scientific captions from figures.")

    uploaded = st.file_uploader(
        "Upload a figure, chart, or architecture diagram",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded:
        col_img, col_cap = st.columns([1, 1])
        with col_img:
            st.image(uploaded, caption=uploaded.name, width="stretch")

        with col_cap:
            if st.button("Analyze Diagram", type="primary"):
                cache_dir = Path(".aether_cache/images/ui")
                cache_dir.mkdir(parents=True, exist_ok=True)
                temp_path = cache_dir / uploaded.name
                temp_path.write_bytes(uploaded.getbuffer())

                with st.spinner("Generating descriptive caption..."):
                    caption = caption_engine.caption(temp_path)

                st.subheader("Generated Caption")
                st.success(caption)
