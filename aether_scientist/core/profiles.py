import logging
import os
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelProfile:
    """Configuration profile for a model inference preset."""

    name: str
    model: str
    chat: bool
    max_new_tokens: int
    temperature: float
    description: str


PROFILES: dict[str, ModelProfile] = {
    "offline": ModelProfile(
        name="offline",
        model="distilgpt2",
        chat=False,
        max_new_tokens=128,
        temperature=0.7,
        description="Tiny base model, tests only",
    ),
    "fast": ModelProfile(
        name="fast",
        model="HuggingFaceTB/SmolLM2-360M-Instruct",
        chat=True,
        max_new_tokens=256,
        temperature=0.3,
        description="360M instruct, CPU-friendly",
    ),
    "balanced": ModelProfile(
        name="balanced",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        chat=True,
        max_new_tokens=384,
        temperature=0.3,
        description="Best speed/quality mix",
    ),
    "quality": ModelProfile(
        name="quality",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        chat=True,
        max_new_tokens=512,
        temperature=0.2,
        description="Highest accuracy, GPU recommended",
    ),
}


def get_profile(name: str) -> ModelProfile:
    """Retrieve a model profile by name or raise ValueError."""
    if name not in PROFILES:
        valid = list(PROFILES.keys())
        raise ValueError(f"Unknown profile '{name}'. Valid profiles: {valid}")
    return PROFILES[name]


def resolve(name: str | None = None) -> ModelProfile:
    """Resolve profile using priority: AETHER_MODEL env > argument > 'offline'."""
    env_name = os.environ.get("AETHER_MODEL")
    target = (
        env_name.strip()
        if env_name and env_name.strip()
        else (name.strip() if name and name.strip() else "offline")
    )

    if target in PROFILES:
        return PROFILES[target]

    logger.warning(
        f"Unknown profile '{target}'. Falling back to 'offline'. Valid: {list(PROFILES.keys())}"
    )
    return PROFILES["offline"]


def list_profiles() -> list[dict]:
    """Return list of profile dictionaries for CLI and API display."""
    return [asdict(profile) for profile in PROFILES.values()]
