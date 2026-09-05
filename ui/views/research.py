"""Autonomous research agent view with real-time step execution tracking."""

import streamlit as st

from aether_scientist.agent.pipeline import ResearchAgent, ResearchResult


def render_research_view(agent: ResearchAgent) -> None:
    """Render autonomous research agent dashboard using active global profile."""
    profile = st.session_state.get("profile", "fast")
    st.header("🔬 Autonomous Research Agent")
    st.caption(
        f"Multi-step inquiry with active profile `{profile}`: "
        "Plans, retrieves, synthesizes, and reports."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Research Inquiry",
            placeholder="e.g. How does the attention mechanism improve sequence transduction?",
        )
    with col2:
        use_web = st.checkbox("Enable Web Search", value=False)

    start_research = st.button("Generate Report", type="primary", use_container_width=True)

    if start_research and question:
        final_result: ResearchResult | None = None
        with st.status("🔬 Agent executing research plan...", expanded=True) as status_box:
            try:
                for event in agent.stream(
                    question=question,
                    profile=profile,
                    use_web=use_web,
                ):
                    evt = event.get("event")
                    if evt == "step":
                        kind = event.get("kind", "").upper()
                        desc = event.get("description", "")
                        st.write(f"**[{kind}]** {desc}")
                    elif evt == "result":
                        final_result = event.get("result_object")
                status_box.update(
                    label="✅ Research report ready!", state="complete", expanded=False
                )
            except Exception as e:
                status_box.update(label=f"⚠️ Execution failed: {e}", state="error")
                st.error(f"Error during research: {e}")
                return

        if final_result:
            st.divider()
            from ui.app import get_model_badge

            st.caption(f"Generated with {get_model_badge(profile)}")
            st.markdown(final_result.report_md)

            if final_result.sources:
                with st.expander(f"📚 References & Evidence ({len(final_result.sources)})"):
                    for i, src in enumerate(final_result.sources, 1):
                        title = src.get("title", f"Source {i}")
                        loc = src.get("source", "")
                        score = src.get("score", 0.0)
                        st.markdown(f"**[{i}] {title}** (`score: {score:.2f}`)  \n*{loc}*")
