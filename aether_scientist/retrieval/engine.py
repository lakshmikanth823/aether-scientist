import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aether_scientist.retrieval.chunker import chunk
from aether_scientist.retrieval.embeddings import EmbeddingEngine
from aether_scientist.retrieval.ingest import ingest_file
from aether_scientist.retrieval.store import Hit, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class Source:
    doc_id: str
    title: str
    location: str
    snippet: str
    score: float


@dataclass
class GroundedAnswer:
    answer: str
    sources: list[Source]
    confidence: float
    citations_valid: bool = True


@dataclass
class IndexStats:
    docs: int
    chunks: int
    elapsed_seconds: float
    images_extracted: int = 0
    captions_generated: int = 0


# Alias for backwards compatibility
IngestStats = IndexStats


class RAGEngine:
    """Retrieval-Augmented Generation engine with citation grounding and hallucination guard."""

    def __init__(
        self,
        config: Any = None,
        embedder: EmbeddingEngine | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.config = config
        cache_dir = getattr(config, "cache_dir", ".aether_cache") if config else ".aether_cache"
        backend = getattr(config, "backend", "auto") if config else "auto"
        self.embedder = embedder or EmbeddingEngine(backend=backend, cache_dir=cache_dir)
        self.store = store or VectorStore()
        self.chunk_size = getattr(config, "chunk_size", 512) if config else 512
        self.overlap = getattr(config, "overlap", 64) if config else 64
        self.top_k = getattr(config, "top_k", 4) if config else 4

    def index(self, paths: list[str]) -> IndexStats:
        """Ingest, chunk, embed, and store documents from given file paths."""
        t0 = time.perf_counter()
        total_docs = 0
        total_images = 0
        total_captions = 0
        all_chunks = []
        all_sources = []
        all_titles = []

        cfg = self.config
        cache_dir = getattr(cfg, "cache_dir", ".aether_cache") if cfg else ".aether_cache"
        for p in paths:
            docs = ingest_file(p, cache_dir=cache_dir)
            total_docs += len(docs)
            for d in docs:
                total_images += len(getattr(d, "images", []))
                total_captions += len(getattr(d, "captions", []))
                c_list = chunk(d.text, size=self.chunk_size, overlap=self.overlap, doc_id=d.doc_id)
                for c in c_list:
                    all_chunks.append(c)
                    all_sources.append(d.source)
                    all_titles.append(d.title)

        if all_chunks:
            vectors = self.embedder.embed([c.text for c in all_chunks])
            self.store.add(all_chunks, vectors, sources=all_sources, titles=all_titles)

        elapsed = round(time.perf_counter() - t0, 3)
        return IndexStats(
            docs=total_docs,
            chunks=len(all_chunks),
            elapsed_seconds=elapsed,
            images_extracted=total_images,
            captions_generated=total_captions,
        )

    def retrieve(self, query: str, k: int | None = None) -> list[Hit]:
        """Retrieve top-k relevant chunks for a query."""
        target_k = k if k is not None else self.top_k
        query_vec = self.embedder.embed([query])
        if query_vec.shape[0] == 0:
            return []
        return self.store.search(query_vec[0], k=target_k)

    def _guard_citations(self, text: str, n_sources: int) -> tuple[str, bool]:
        valid = True

        def _replace_cite(match: re.Match[str]) -> str:
            nonlocal valid
            idx = int(match.group(1))
            if 1 <= idx <= n_sources:
                return match.group(0)
            valid = False
            return ""

        cleaned = re.sub(r"\[(\d+)\]", _replace_cite, text)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned, valid

    def grounded_generate(
        self,
        query: str,
        domain: str = "physics",
        system_prompt: str = "",
        model_name: str = "distilgpt2",
        profile: Any = None,
    ) -> GroundedAnswer:
        """Generate an answer strictly grounded in retrieved evidence."""
        hits = self.retrieve(query, k=self.top_k)
        sources: list[Source] = [
            Source(
                doc_id=h.doc_id,
                title=h.title,
                location=f"chunk_{h.chunk.index}",
                snippet=h.chunk.text[:200],
                score=round(h.score, 4),
            )
            for h in hits
        ]

        if not hits:
            return GroundedAnswer(
                answer="Insufficient evidence found in indexed literature.",
                sources=[],
                confidence=0.0,
                citations_valid=True,
            )

        mean_score = float(np.mean([h.score for h in hits])) if hits else 0.0
        confidence = round(min(0.95, max(0.0, mean_score)), 4)

        context_blocks = "\n\n".join(
            f"[{i + 1}] Source: {h.title or Path(h.source).name}\n{h.chunk.text}"
            for i, h in enumerate(hits)
        )

        sys_msg = system_prompt or f"You are an expert {domain} scientist."
        system_instructions = (
            f"{sys_msg}\n\n"
            f"Context:\n{context_blocks}\n\n"
            f"Instructions:\n"
            f"- Answer the question ONLY using the facts from the numbered context above.\n"
            f"- Cite each statement with its source index in brackets, e.g. [1].\n"
            f"- If the context lacks information, state 'insufficient evidence'."
        )

        from aether_scientist.core.inference import InferenceEngine

        gen = InferenceEngine.generate(
            system_prompt=system_instructions,
            query=query,
            model_name=model_name,
            profile=profile,
            max_new_tokens=256,
        )
        raw_answer = gen.get("text", "")
        cleaned_answer, citations_valid = self._guard_citations(raw_answer, len(sources))

        return GroundedAnswer(
            answer=cleaned_answer,
            sources=sources,
            confidence=confidence,
            citations_valid=citations_valid,
        )

