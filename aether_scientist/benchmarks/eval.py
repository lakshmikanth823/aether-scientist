import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aether_scientist.core.inference import InferenceEngine
from aether_scientist.core.profiles import resolve

logger = logging.getLogger(__name__)


def compute_citation_validity(answers: list[str], n_contexts: int = 4) -> float:
    """Compute fraction of answers containing at least one valid [n] bracket."""
    if not answers:
        return 0.0
    valid_count = 0
    for ans in answers:
        cites = [int(c) for c in re.findall(r"\[(\d+)\]", ans)]
        if any(1 <= c <= n_contexts for c in cites):
            valid_count += 1
    return round(valid_count / len(answers), 4)


@dataclass
class EvalReport:
    """Benchmark evaluation report with accuracy and latency metrics."""

    total: int
    correct: int
    accuracy: float
    avg_latency_ms: float
    details: list[dict[str, Any]]
    citation_validity_rate: float | None = None
    profile: str = "offline"
    model: str = "distilgpt2"
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_quiz(use_sciq: bool = False) -> list[dict[str, Any]]:
    if use_sciq:
        try:
            from datasets import load_dataset

            ds = load_dataset("allenai/sciq", split="validation")
            return [
                {
                    "id": idx,
                    "domain": "sciq",
                    "question": item["question"],
                    "options": [
                        item["distractor1"],
                        item["distractor2"],
                        item["distractor3"],
                        item["correct_answer"],
                    ],
                    "answer": item["correct_answer"],
                }
                for idx, item in enumerate(ds)
            ]
        except Exception as e:
            logger.warning(f"Failed to load SciQ dataset, falling back to bundled quiz: {e}")

    quiz_path = Path(__file__).parent / "data" / "quiz.json"
    if not quiz_path.exists():
        raise FileNotFoundError(f"Quiz file not found at {quiz_path}")
    return json.loads(quiz_path.read_text(encoding="utf-8"))


def run_eval(
    n: int = 20,
    model: str = "distilgpt2",
    use_sciq: bool = False,
    profiles: list[str] | str | None = None,
    profile: str | None = None,
    generator: Any = None,
    real: bool = False,
    rag_answers: list[str] | None = None,
    n_contexts: int = 4,
) -> EvalReport | dict[str, Any]:
    """Run benchmark evaluation on a single profile or compare multiple profiles."""
    if isinstance(profiles, list) and len(profiles) > 1:
        return compare_profiles(
            profiles=profiles, n=n, generator=generator, real=real, use_sciq=use_sciq
        )

    chosen = (
        profiles[0]
        if (isinstance(profiles, list) and profiles)
        else (profiles or profile or model)
    )
    prof = resolve(chosen if isinstance(chosen, str) else None)

    if real and os.environ.get("AETHER_ALLOW_DOWNLOADS") != "1":
        logger.warning(
            "Real downloads disabled. Set AETHER_ALLOW_DOWNLOADS=1 to permit downloads."
        )

    questions = _load_quiz(use_sciq=use_sciq)[:n]
    if not questions:
        return EvalReport(
            0, 0, 0.0, 0.0, [], None, prof.name, prof.model, InferenceEngine.device
        )

    correct_count = 0
    total_latency = 0.0
    details: list[dict[str, Any]] = []

    for q in questions:
        opts_str = ", ".join(f"({chr(65 + i)}) {opt}" for i, opt in enumerate(q["options"]))
        prompt = (
            f"Question: {q['question']}\n"
            f"Options: {opts_str}\n"
            f"Answer with the exact correct option text.\nAnswer:"
        )

        t0 = time.perf_counter()
        if generator is not None:
            gen = generator(prompt=prompt, profile=prof, model_name=prof.model, max_new_tokens=32)
        else:
            gen = InferenceEngine.generate(prompt=prompt, profile=prof, max_new_tokens=32)
        latency_ms = (time.perf_counter() - t0) * 1000
        total_latency += latency_ms

        generated = gen.get("text", "").strip()
        expected = q["answer"].strip()
        gen_lower = generated.lower()
        exp_lower = expected.lower()
        is_correct = (exp_lower in gen_lower) or (gen_lower in exp_lower)

        if is_correct:
            correct_count += 1

        details.append(
            {
                "id": q["id"],
                "question": q["question"],
                "expected": expected,
                "generated": generated,
                "correct": is_correct,
                "latency_ms": round(latency_ms, 2),
            }
        )

    cit_rate = (
        compute_citation_validity(rag_answers, n_contexts=n_contexts)
        if rag_answers is not None
        else None
    )
    return EvalReport(
        total=len(questions),
        correct=correct_count,
        accuracy=round(correct_count / len(questions), 4) if questions else 0.0,
        avg_latency_ms=round(total_latency / len(questions), 2) if questions else 0.0,
        details=details,
        citation_validity_rate=cit_rate,
        profile=prof.name,
        model=prof.model,
        device=InferenceEngine.device,
    )


def compare_profiles(
    profiles: list[str],
    n: int = 20,
    generator: Any = None,
    real: bool = False,
    use_sciq: bool = False,
) -> dict[str, Any]:
    """Run comparative evaluation across multiple model profiles."""
    results: dict[str, Any] = {}
    summary: list[dict[str, Any]] = []

    for name in profiles:
        rep = run_eval(
            n=n, profile=name, generator=generator, real=real, use_sciq=use_sciq
        )
        rep_dict = rep.to_dict() if isinstance(rep, EvalReport) else rep
        results[name] = rep_dict
        summary.append(
            {
                "profile": name,
                "model": rep_dict["model"],
                "accuracy": rep_dict["accuracy"],
                "avg_latency_ms": rep_dict["avg_latency_ms"],
                "citation_validity": rep_dict["citation_validity_rate"],
                "device": rep_dict["device"],
            }
        )

    return {"profiles": results, "summary": summary}
