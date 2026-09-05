from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DomainResult:
    domain: str
    analysis: str
    confidence: float
    reasoning_chain: list[str]
    citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DomainAdapter(ABC):
    @abstractmethod
    def terminology(self) -> dict[str, str]:
        """Return terminology mapping term to subfield."""
        pass

    @abstractmethod
    def reasoning_patterns(self) -> list[str]:
        """Return reasoning patterns."""
        pass

    @abstractmethod
    def experimental_methods(self) -> list[str]:
        """Return experimental methods."""
        pass

    @property
    def domain_name(self) -> str:
        return self.__class__.__name__.replace("Adapter", "").lower()

    def system_prompt(self) -> str:
        """Return domain-specific system prompt for LLM."""
        terms = ", ".join(list(self.terminology().keys())[:5])
        patterns = ", ".join(self.reasoning_patterns()[:3])
        return (
            f"You are a {self.domain_name} research scientist. "
            f"Key concepts: {terms}. "
            f"Apply these reasoning approaches: {patterns}. "
            f"Provide detailed, accurate scientific analysis."
        )

    def process(self, query: list[int], context: list[str] | None = None) -> DomainResult:
        terms = self.terminology()
        patterns = self.reasoning_patterns()
        return self._apply_patterns(query, terms, patterns, context)

    def synthesize(self, papers: list[str]) -> DomainResult:
        patterns = self.reasoning_patterns()
        analysis = f"Synthesizing {len(papers)} papers in {self.domain_name}."
        chain = [f"Applied {pattern} to literature" for pattern in patterns[:2]]
        return DomainResult(
            domain=self.domain_name,
            analysis=analysis,
            confidence=0.85,
            reasoning_chain=chain,
            citations=papers,
        )

    def hypothesize(self, observations: list[str]) -> DomainResult:
        patterns = self.reasoning_patterns()
        analysis = f"Generated hypothesis from {len(observations)} observations."
        chain = [f"Correlated observation with {pattern}" for pattern in patterns[:2]]
        return DomainResult(
            domain=self.domain_name, analysis=analysis, confidence=0.75, reasoning_chain=chain
        )

    def _apply_patterns(
        self,
        query: list[int],
        terms: dict[str, str],
        patterns: list[str],
        context: list[str] | None = None,
    ) -> DomainResult:
        relevant_terms = self._extract_relevant_terms(query, terms)
        context_str = " ".join(context) if context else ""
        analysis = f"Processed query of length {len(query)} using {len(relevant_terms)} terms."
        chain = [f"Identified term: {term} in subfield {terms[term]}" for term in relevant_terms]
        chain.extend(
            [
                f"Applied reasoning: {pattern} with context length {len(context_str)}"
                for pattern in patterns[:3]
            ]
        )

        confidence = min(1.0, (len(relevant_terms) / max(1, len(query))) + 0.5)

        return DomainResult(
            domain=self.domain_name,
            analysis=analysis,
            confidence=confidence,
            reasoning_chain=chain,
            metadata={"query_length": len(query), "terms_matched": len(relevant_terms)},
        )

    def _extract_relevant_terms(self, query: list[int], terms: dict[str, str]) -> list[str]:
        q_sum = sum(query) if query else 0
        term_keys = list(terms.keys())
        if not term_keys:
            return []

        return [term_keys[(q_sum + i) % len(term_keys)] for i in range(min(3, len(term_keys)))]
