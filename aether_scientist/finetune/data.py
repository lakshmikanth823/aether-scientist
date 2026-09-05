import json
from pathlib import Path
from typing import Any

from aether_scientist.core.profiles import ModelProfile

SYSTEM_PROMPT = (
    "You are AetherScientist, an expert scientific research assistant. "
    "Analyze, synthesize, and explain scientific concepts accurately."
)


def format_chatml_entry(
    user_content: str,
    assistant_content: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, list[dict[str, str]]]:
    """Build a single ChatML conversation dictionary with system, user, and assistant."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": assistant_content.strip()},
        ]
    }


def format_dataset(docs: list[str], profile: ModelProfile) -> list[dict[str, Any]]:
    """Convert raw scientific text or JSON records into ChatML instruction-tuning format."""
    dataset: list[dict[str, Any]] = []
    for doc in docs:
        text = doc.strip()
        if not text:
            continue
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                inst = data.get("instruction") or data.get("prompt") or data.get("question") or ""
                inp = data.get("input") or data.get("context") or ""
                out = data.get("output") or data.get("response") or data.get("answer") or ""
                user = f"{inst}\n\n{inp}".strip() if inp else inst
                if user and out:
                    dataset.append(format_chatml_entry(user, out))
                    continue
        except (json.JSONDecodeError, TypeError):
            pass

        if "\n\nAnswer:" in text:
            parts = text.split("\n\nAnswer:", 1)
            user = parts[0].replace("Question:", "").strip()
            dataset.append(format_chatml_entry(user, parts[1].strip()))
        else:
            user = f"Analyze the following scientific text and summarize key findings:\n\n{text}"
            dataset.append(format_chatml_entry(user, text))
    return dataset


def load_scientific_data(path: str | Path) -> list[str]:
    """Load scientific texts or records from .json, .jsonl, or .txt file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    content = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        return [line.strip() for line in content.splitlines() if line.strip()]
    if p.suffix == ".json":
        data = json.loads(content)
        if isinstance(data, list):
            return [json.dumps(item) if isinstance(item, dict) else str(item) for item in data]
        return [content]
    return [block.strip() for block in content.split("\n\n") if block.strip()]
