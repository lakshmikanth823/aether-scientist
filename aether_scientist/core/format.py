from typing import Any

from aether_scientist.core.profiles import ModelProfile

STATIC_MARKERS: list[str] = [
    "\n\nQuestion:",
    "\n\n###",
    "\n\n### Assistant:",
    "\n\nSystem:",
]


def get_stop_markers(tokenizer: Any = None) -> list[str]:
    """Dynamically resolve stop markers from static list and tokenizer tokens."""
    markers = list(STATIC_MARKERS)
    if tokenizer is None:
        return markers

    eos = getattr(tokenizer, "eos_token", None)
    if eos and isinstance(eos, str) and eos not in markers:
        markers.append(eos)

    additional = getattr(tokenizer, "additional_special_tokens", None) or []
    for token in additional:
        if isinstance(token, str) and token and token not in markers:
            markers.append(token)

    return markers


def format_prompt(
    tokenizer: Any,
    profile: ModelProfile,
    system: str = "",
    user: str = "",
) -> str:
    """Format prompt with chat template if profile supports chat, otherwise raw."""
    if profile.chat and tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    if system and user:
        return f"{system}\n\nQuestion: {user}\n\nAnswer:"
    return user or system


def clean_output(text: str, markers: list[str] | None = None) -> str:
    """Strip whitespace, cut at the earliest stop marker, and strip again."""
    cleaned = text.strip()
    if not markers:
        return cleaned

    earliest = -1
    for marker in markers:
        if not marker:
            continue
        idx = cleaned.find(marker)
        if idx != -1 and (earliest == -1 or idx < earliest):
            earliest = idx

    if earliest != -1:
        cleaned = cleaned[:earliest]

    return cleaned.strip()
