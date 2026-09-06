"""Knowledge base and RAG view with document ingestion and grounded search."""

from pathlib import Path

import streamlit as st

from aether_scientist.retrieval.engine import GroundedAnswer, RAGEngine


def render_knowledge_view(rag_engine: RAGEngine) -> None:
    """Render knowledge base manager and grounded RAG query view."""
    profile = st.session_state.get("profile", "fast")
    st.header("📚 Knowledge Base & RAG")
    st.caption(f"Grounded Retrieval active with profile `{profile}`")

    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []

    tab_upload, tab_query = st.tabs(["📁 Ingest Documents", "🔍 Grounded Query"])

    with tab_upload:
        col_up, col_stat = st.columns([2, 1])
        with col_up:
            uploaded = st.file_uploader(
                "Upload scientific papers (PDF, TXT, MD)",
                type=["pdf", "txt", "md"],
                accept_multiple_files=True,
            )
            if uploaded and st.button("Index Uploaded Files"):
                saved_paths = []
                cache_dir = Path(".aether_cache/uploads")
                cache_dir.mkdir(parents=True, exist_ok=True)
                with st.spinner("Processing and embedding documents..."):
                    for up in uploaded:
                        dest = cache_dir / up.name
                        dest.write_bytes(up.getbuffer())
                        saved_paths.append(str(dest))
                        if up.name not in st.session_state.indexed_files:
                            st.session_state.indexed_files.append(up.name)

                    stats = rag_engine.index(saved_paths)
                msg = (
                    f"Indexed {stats.docs} docs ({stats.chunks} chunks) "
                    f"in {stats.elapsed_seconds}s! "
                    f"[Images: {stats.images_extracted}, Captions: {stats.captions_generated}]"
                )
                st.success(msg)

            st.divider()
            if st.button("Load Demo Corpus"):
                demo_path = Path("data/demo_abstracts.txt")
                if demo_path.exists():
                    with st.spinner("Indexing demo scientific corpus..."):
                        stats = rag_engine.index([demo_path.as_posix()])
                    if "demo_abstracts.txt" not in st.session_state.indexed_files:
                        st.session_state.indexed_files.append("demo_abstracts.txt")
                    st.success(
                        f"Indexed demo corpus: {stats.docs} doc ({stats.chunks} chunks) "
                        f"in {stats.elapsed_seconds}s!"
                    )
                else:
                    st.error("Demo corpus data/demo_abstracts.txt not found.")

        with col_stat:
            st.metric("Total Chunks in Store", len(rag_engine.store))
            st.write("**Indexed Files:**")
            if st.session_state.indexed_files:
                for f in st.session_state.indexed_files:
                    st.text(f"• {f}")
            else:
                st.caption("No files uploaded yet.")

            if st.button("Clear Index", type="secondary"):
                rag_engine.store.clear()
                st.session_state.indexed_files = []
                st.session_state.rag_messages = []
                st.warning("Index cleared.")

    with tab_query:
        for q_msg in st.session_state.rag_messages:
            with st.chat_message(q_msg["role"]):
                st.markdown(q_msg["content"])
                if q_msg.get("badge"):
                    st.caption(q_msg["badge"])
                if q_msg.get("sources"):
                    with st.expander(f"Sources ({len(q_msg['sources'])})"):
                        for s in q_msg["sources"]:
                            ttl = s.title or s.doc_id
                            st.markdown(f"**[{s.score:.2f}] {ttl}**\n> {s.snippet}")

        if prompt := st.chat_input("Ask a grounded question..."):
            st.session_state.rag_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            from ui.app import get_model_badge

            badge = get_model_badge(profile)
            with st.chat_message("assistant"), st.spinner("Retrieving evidence..."):
                res: GroundedAnswer = rag_engine.grounded_generate(prompt, profile=profile)
                st.markdown(res.answer)
                st.caption(badge)
                if res.sources:
                    lbl = f"Sources ({len(res.sources)}) - Conf: {res.confidence:.2f}"
                    with st.expander(lbl):
                        for s in res.sources:
                            ttl = s.title or s.doc_id
                            st.markdown(f"**[{s.score:.2f}] {ttl}**\n> {s.snippet}")
            st.session_state.rag_messages.append(
                {
                    "role": "assistant",
                    "content": res.answer,
                    "sources": res.sources,
                    "badge": badge,
                }
            )
