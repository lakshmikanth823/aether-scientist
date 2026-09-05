import re
from typing import Any


def extract_topic(question: str) -> str:
    """Extract the core research subject by stripping question words and punctuation."""
    q = question.strip()
    # 1. Cut at second clause: and what|and how|and why|and which|or what
    q = re.split(r"(?i)\s+(?:and\s+(?:what|how|why|which)|or\s+what)\b", q)[0]
    # 2. Strip leading question words and auxiliaries
    prefix_pattern = r"^(?:(?:what|how|why|which|when|where|is|are|do|does|did|can|could)\b\s*)+"
    topic = re.sub(prefix_pattern, "", q, flags=re.IGNORECASE).strip()
    # 3. Strip trailing punctuation
    topic = re.sub(r"[\?\.\!]+$", "", topic).strip()
    # 4. Strip trailing lone verbs
    topic = re.sub(r"\b(?:work|works|mean|means)\b\s*$", "", topic, flags=re.IGNORECASE).strip()
    # 5. Collapse whitespace
    topic = re.sub(r"\s+", " ", topic).strip()
    return topic


def plan(question: str, k: int = 4) -> list[str]:
    """Decompose research question into template-based sub-queries."""
    topic = extract_topic(question)
    if not topic:
        return [question]

    templates = [
        f"What is {topic}? (definition and background)",
        f"How does {topic} work? (mechanisms and principles)",
        f"What are applications of {topic}?",
        f"What are limitations and challenges of {topic}?",
    ]
    seen: set[str] = set()
    sub_queries: list[str] = []
    for sq in templates:
        if sq not in seen:
            seen.add(sq)
            sub_queries.append(sq)

    return sub_queries[: max(2, min(k, len(sub_queries)))]


def plan_llm(
    question: str,
    engine: Any = None,
    profile: str = "offline",
    k: int = 4,
) -> list[str]:
    """Decompose research question using LLM with fallback to heuristic planner."""
    from aether_scientist.core.inference import InferenceEngine

    prompt = (
        f"Decompose the following scientific question into {k} specific sub-questions:\n"
        f"Question: {question}\n\n"
        f"Numbered sub-questions:\n1."
    )
    try:
        if engine is not None and hasattr(engine, "generate"):
            res = engine.generate(prompt=prompt, profile=profile, max_new_tokens=128)
        else:
            res = InferenceEngine.generate(prompt=prompt, profile=profile, max_new_tokens=128)

        text = res.get("text", "").strip()
        lines = re.findall(r"(?:^|\n)\s*\d+[\.\)]\s*([^\n]+)", text)
        if not lines:
            lines = re.findall(r"(?:^|\n)\s*\d+[\.\)]\s*([^\n]+)", "1. " + text)
        parsed = [re.sub(r"^\d+[\.\)]\s*", "", line).strip() for line in lines if line.strip()]
        if len(parsed) >= 2:
            return parsed[:k]
    except Exception:
        pass

    return plan(question, k=k)
