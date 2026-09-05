import re
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aether_scientist.agent.planner import plan, plan_llm
from aether_scientist.agent.report import render_report
from aether_scientist.agent.state import ResearchState
from aether_scientist.core.config import AetherConfig
from aether_scientist.core.inference import InferenceEngine
from aether_scientist.core.model import AetherScientist


@dataclass
class ResearchResult:
    """Structured outcome of an autonomous research inquiry."""

    question: str
    report_md: str
    state: ResearchState
    sources: list[dict[str, Any]]
    confidence: float
    coverage: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "question": self.question,
            "report_md": self.report_md,
            "sources": self.sources,
            "confidence": self.confidence,
            "coverage": self.coverage,
            "steps": [
                {"kind": s.kind, "description": s.description, "status": s.status}
                for s in self.state.steps
            ],
            "findings": self.state.findings,
        }


class ResearchAgent:
    """Autonomous research assistant that plans, retrieves, synthesizes, and reports."""

    def __init__(self, config: AetherConfig | None = None) -> None:
        self.config: AetherConfig = config or AetherConfig()
        self.scientist: AetherScientist = AetherScientist(self.config)
        self.rag_engine = self.scientist.rag_engine

    def run(
        self,
        question: str,
        k: int = 3,
        use_llm_planner: bool = False,
        profile: str | None = None,
    ) -> ResearchResult:
        """Execute complete multi-step autonomous research pipeline."""
        for event in self.stream(
            question=question, k=k, use_llm_planner=use_llm_planner, profile=profile
        ):
            if event.get("event") == "result":
                return event["result_object"]
        raise RuntimeError("Pipeline failed to produce a final result.")

    def stream(
        self,
        question: str,
        k: int = 3,
        use_llm_planner: bool = False,
        profile: str | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream step progression events and yield final research report."""
        if len(self.rag_engine.store) == 0:
            raise IndexError("No documents indexed. Run: aether ingest <paths>")

        target_prof = profile or self.config.profile
        state = ResearchState(question=question, profile=target_prof)

        # 1. PLAN
        yield {"event": "step", "kind": "plan", "description": f"Decomposing: {question}"}
        state.record("plan", f"Decomposing question: {question}")
        if use_llm_planner:
            sub_queries = plan_llm(question, profile=target_prof, k=4)
        else:
            sub_queries = plan(question, k=4)
        state.sub_queries = sub_queries

        # 2. RETRIEVE & 3. SYNTHESIZE per sub-query
        for sq in sub_queries:
            yield {
                "event": "step",
                "kind": "retrieve",
                "sub_query": sq,
                "description": f"Retrieving: {sq}",
            }
            state.record("retrieve", f"Retrieving literature for: {sq}")
            hits = self.rag_engine.retrieve(sq, k=k)
            assigned_indices = state.add_sources(hits)

            yield {
                "event": "step",
                "kind": "synthesize",
                "sub_query": sq,
                "description": f"Synthesizing: {sq}",
            }
            state.record("synthesize", f"Synthesizing findings for: {sq}")

            if not hits:
                state.findings.append(
                    {
                        "sub_query": sq,
                        "text": "Insufficient evidence in indexed corpus.",
                        "citations": [],
                    }
                )
            else:
                ctx_items = []
                for i, h in enumerate(hits):
                    title = getattr(h, "title", "") or Path(getattr(h, "source", "")).name
                    chunk = getattr(h, "chunk", None)
                    chunk_text = getattr(chunk, "text", str(h))
                    ctx_items.append(f"[{assigned_indices[i]}] Source: {title}\n{chunk_text}")

                context_blocks = "\n\n".join(ctx_items)
                instructions = (
                    "You are an expert scientist synthesizing research evidence.\n\n"
                    f"Context:\n{context_blocks}\n\n"
                    "Instructions:\n"
                    "- Answer the sub-question strictly using facts from context above.\n"
                    f"- Cite statements using exact brackets, e.g. [{assigned_indices[0]}].\n"
                    "- If evidence is missing, state 'insufficient evidence'."
                )
                gen = InferenceEngine.generate(
                    system_prompt=instructions,
                    query=sq,
                    profile=target_prof,
                    max_new_tokens=256,
                )
                text = gen.get("text", "").strip()
                cites = [
                    int(c)
                    for c in re.findall(r"\[(\d+)\]", text)
                    if int(c) in assigned_indices
                ]
                state.findings.append(
                    {"sub_query": sq, "text": text, "citations": sorted(set(cites))}
                )

        # 4. REPORT
        yield {"event": "step", "kind": "report", "description": "Rendering final research report"}
        state.record("report", "Rendering final report")

        scores = [s["score"] for s in state.sources if "score" in s]
        mean_score = float(sum(scores) / len(scores)) if scores else 0.0
        confidence = round(min(0.95, max(0.0, mean_score)), 4)
        cov = state.coverage()
        report_md = render_report(
            question=question,
            state=state,
            confidence=confidence,
            profile=target_prof,
            device=InferenceEngine.device,
        )

        res = ResearchResult(
            question=question,
            report_md=report_md,
            state=state,
            sources=state.sources,
            confidence=confidence,
            coverage=cov,
        )
        yield {"event": "result", "data": res.to_dict(), "result_object": res}
