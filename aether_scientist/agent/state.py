from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class AgentStepLimit(RuntimeError):  # noqa: N818
    """Raised when the agent exceeds the allowed execution step budget."""


@dataclass
class Step:
    """Individual workflow step recorded by the agent."""

    kind: str  # "plan" | "retrieve" | "synthesize" | "report"
    description: str
    status: str = "done"  # "done" | "failed"


@dataclass
class ResearchState:
    """State tracking for an autonomous multi-step research inquiry."""

    question: str
    profile: str = "offline"
    max_steps: int = 12
    sub_queries: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)

    def record(self, kind: str, description: str, status: str = "done") -> Step:
        """Record an execution step and enforce step limit."""
        if len(self.steps) >= self.max_steps:
            raise AgentStepLimit(
                f"Agent step limit of {self.max_steps} exceeded at step '{kind}'."
            )
        step = Step(kind=kind, description=description, status=status)
        self.steps.append(step)
        return step

    def add_sources(self, hits: list[Any]) -> list[int]:
        """Deduplicate sources by (doc_id, chunk_id), keeping highest score."""
        assigned_indices: list[int] = []
        for h in hits:
            if isinstance(h, dict):
                doc_id = h.get("doc_id", "")
                chunk_id = h.get("chunk_id", h.get("location", 0))
                title = h.get("title", "")
                source = h.get("source", "")
                snippet = h.get("snippet", "")
                score = float(h.get("score", 0.0))
            else:
                doc_id = getattr(h, "doc_id", "")
                chunk = getattr(h, "chunk", None)
                chunk_id = getattr(chunk, "index", 0) if chunk else 0
                title = getattr(h, "title", "")
                source = getattr(h, "source", "")
                snippet = chunk.text[:200] if chunk else ""
                score = float(getattr(h, "score", 0.0))

            key = (doc_id, chunk_id)
            existing_idx = None
            for idx, s in enumerate(self.sources):
                if (s.get("doc_id"), s.get("chunk_id")) == key:
                    existing_idx = idx + 1
                    if score > s.get("score", 0.0):
                        s["score"] = round(score, 4)
                        s["snippet"] = snippet
                    break

            if existing_idx is not None:
                assigned_indices.append(existing_idx)
            else:
                new_idx = len(self.sources) + 1
                fallback_title = Path(source).name if source else "Document"
                self.sources.append(
                    {
                        "index": new_idx,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "title": title or fallback_title,
                        "source": str(source),
                        "snippet": snippet,
                        "score": round(score, 4),
                    }
                )
                assigned_indices.append(new_idx)
        return assigned_indices

    def coverage(self) -> dict[str, Any]:
        """Compute coverage statistics across retrieved sources."""
        n_sources = len(self.sources)
        doc_ids = {
            s["doc_id"] or s["source"]
            for s in self.sources
            if s.get("doc_id") or s.get("source")
        }
        scores = [s["score"] for s in self.sources if "score" in s]
        mean_score = float(sum(scores) / len(scores)) if scores else 0.0
        return {
            "n_sources": n_sources,
            "n_docs": len(doc_ids),
            "mean_score": round(mean_score, 4),
        }
