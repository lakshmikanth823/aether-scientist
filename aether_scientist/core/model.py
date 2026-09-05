import logging
from typing import Any

from aether_scientist.core.config import AetherConfig
from aether_scientist.core.inference import InferenceEngine
from aether_scientist.core.tokenizer import ScientificTokenizer
from aether_scientist.domains import load_adapter
from aether_scientist.domains.base import DomainAdapter
from aether_scientist.utils.efficiency import EfficiencyOptimizer

logger = logging.getLogger(__name__)


class AetherScientist:
    """Core class for AetherScientist."""

    def __init__(self, config: AetherConfig) -> None:
        self.config: AetherConfig = config
        self.tokenizer: ScientificTokenizer = ScientificTokenizer()
        self.adapters: dict[str, DomainAdapter] = self._load_adapters()
        self.efficiency_optimizer: EfficiencyOptimizer = EfficiencyOptimizer(
            bits=self.config.quantization_bits,
            cache_enabled=self.config.cache_enabled,
            cache_size=self.config.cache_size,
        )
        self._is_active: bool = False

    def _load_adapters(self) -> dict[str, DomainAdapter]:
        """Dynamically loads domain adapters."""
        adapters: dict[str, DomainAdapter] = {}
        for domain in self.config.domains:
            try:
                adapter = load_adapter(domain)
                adapters[domain] = adapter
            except Exception as e:
                logger.error(f"Failed to load adapter for domain {domain}: {e}")
        return adapters

    @property
    def active_domains(self) -> list[str]:
        """Returns the list of currently active and loaded domains."""
        return list(self.adapters.keys())

    @property
    def device_status(self) -> str:
        """Returns the configured device."""
        return self.config.device

    def __enter__(self) -> "AetherScientist":
        """Context manager entry point."""
        self._is_active = True
        logger.info(f"AetherScientist activated with domains: {self.active_domains}")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit point."""
        self._is_active = False
        logger.info("AetherScientist deactivated and resources released.")

    def _detect_domain(self, query: str) -> str:
        """Keyword-based domain detection using adapter terminology."""
        query_lower = query.lower()
        domain_scores: dict[str, int] = {}

        for domain, adapter in self.adapters.items():
            score = sum(1 for keyword in adapter.terminology() if keyword.lower() in query_lower)
            domain_scores[domain] = score

        if not domain_scores:
            return self.config.domains[0] if self.config.domains else "unknown"

        best_domain = max(domain_scores.items(), key=lambda item: item[1])[0]
        return best_domain if domain_scores[best_domain] > 0 else self.active_domains[0]

    def analyze(self, query: str, papers: list[str] | None = None) -> dict[str, Any]:
        """Single entry point for analyzing a scientific query."""
        papers = papers or []
        domain: str = self._detect_domain(query)
        adapter: DomainAdapter = self.adapters[domain]

        logger.info(f"Analyzing query '{query}' in domain '{domain}'")

        query_tokens = self.tokenizer.encode(query)
        domain_result = adapter.process(query_tokens, papers)

        # Build domain-specific prompt and generate
        prompt = f"{adapter.system_prompt()}\n\nQuestion: {query}\n\nAnswer:"
        generation = InferenceEngine.generate(
            prompt=prompt,
            model_name=self.config.model_name,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )

        result = {
            "analysis": domain_result.analysis,
            "confidence": domain_result.confidence,
            "reasoning_chain": domain_result.reasoning_chain,
            "generated_text": generation["text"],
            "generation_metadata": generation,
        }

        if papers:
            synthesis = self.synthesize_papers(papers, domain)
            result["literature_synthesis"] = synthesis

        return {
            "domain": domain,
            "query_tokens": query_tokens,
            "analysis_result": result,
            "config_used": self.config.model_name,
        }

    def synthesize_papers(self, papers: list[str], domain: str | None = None) -> dict[str, Any]:
        """Synthesizes literature for a given list of papers."""
        if not papers:
            return {"summary": "No papers provided.", "key_findings": []}

        target_domain = domain or (self.active_domains[0] if self.active_domains else "unknown")
        logger.info(f"Synthesizing {len(papers)} papers for domain '{target_domain}'")

        return {
            "summary": f"Synthesized {len(papers)} papers in {target_domain}.",
            "key_findings": [f"Finding from {paper}" for paper in papers[: self.config.max_papers]],
        }

    def generate_hypothesis(self, observations: list[str], domain: str) -> dict[str, Any]:
        """Generates a hypothesis based on provided observations."""
        logger.info(
            f"Generating hypothesis for domain '{domain}' based on {len(observations)} observations"
        )

        if domain not in self.adapters:
            raise ValueError(f"Domain '{domain}' is not loaded.")

        return {
            "hypothesis": f"Hypothesis derived from {len(observations)} observations in {domain}.",
            "confidence_score": 0.85,
        }

    def design_experiment(self, hypothesis: str, domain: str) -> dict[str, Any]:
        """Designs an experiment to test a given hypothesis."""
        logger.info(f"Designing experiment for hypothesis in domain '{domain}'")

        if domain not in self.adapters:
            raise ValueError(f"Domain '{domain}' is not loaded.")

        return {
            "experiment_design": f"Experiment to test: {hypothesis}",
            "required_equipment": ["Equipment A", "Equipment B"],
            "estimated_duration_days": 14,
        }
