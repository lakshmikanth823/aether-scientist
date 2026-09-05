"""Prompt formatting, stop marker resolution, and degenerate output guard."""

from typing import Any

from aether_scientist.core.profiles import ModelProfile

STATIC_MARKERS: list[str] = [
    "\n\nQuestion:",
    "\n\n###",
    "\n\n### Assistant:",
    "\n\nSystem:",
]

COMMON_SHORT_WORDS: set[str] = {
    "a", "i", "am", "an", "as", "at", "be", "by", "do", "go",
    "he", "if", "in", "is", "it", "me", "my", "no", "of", "on",
    "or", "so", "to", "up", "us", "we",
}


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


def detect_degenerate(text: str) -> bool:
    """Detect degenerate model output: n-gram loops or charset collapse."""
    words = text.split()
    if not words:
        return False

    # 1. Charset collapse: > 25% of tokens are single/double-char fragments
    if len(words) >= 4:
        fragments = [w for w in words if len(w) <= 2 and w.lower() not in COMMON_SHORT_WORDS]
        if len(fragments) / len(words) > 0.25:
            return True

    # 2. n-gram loop: any 3-6 word n-gram repeating >= 3 times
    for n in range(3, min(7, len(words) // 2 + 1)):
        counts: dict[tuple[str, ...], int] = {}
        for i in range(len(words) - n + 1):
            ng = tuple(words[i : i + n])
            counts[ng] = counts.get(ng, 0) + 1
            if counts[ng] >= 3:
                return True

    return False


def find_loop_cut_index(text: str) -> int | None:
    """Find character index where the second occurrence of a repeating n-gram starts."""
    words = text.split()
    for n in range(6, 2, -1):
        occurrences: dict[tuple[str, ...], list[int]] = {}
        for i in range(len(words) - n + 1):
            ng = tuple(words[i : i + n])
            occurrences.setdefault(ng, []).append(i)

        for _ng, indices in occurrences.items():
            if len(indices) >= 3:
                loop_word_idx = indices[1]
                prefix = " ".join(words[:loop_word_idx])
                pos = text.find(prefix)
                if pos != -1:
                    return pos + len(prefix)
                return len(prefix)

    return None


def clean_output(
    text: str,
    markers: list[str] | None = None,
    profile_name: str = "offline",
) -> str:
    """Strip whitespace, cut at stop markers, and guard against degeneration."""
    cleaned = text.strip()
    if markers:
        earliest = -1
        for marker in markers:
            if not marker:
                continue
            idx = cleaned.find(marker)
            if idx != -1 and (earliest == -1 or idx < earliest):
                earliest = idx

        if earliest != -1:
            cleaned = cleaned[:earliest].strip()

    if detect_degenerate(cleaned):
        cut_pos = find_loop_cut_index(cleaned)
        cleaned = cleaned[:cut_pos].strip() if cut_pos is not None else ""
        if len(cleaned) < 40:

            return (
                f"[Model produced degenerate output on profile '{profile_name}'. "
                "Use a stronger profile (--profile balanced|quality).]"
            )

    return cleaned
