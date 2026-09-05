from typing import Any

from aether_scientist.core.profiles import get_profile

PROFILE_MODULES: dict[str, list[str]] = {
    "offline": ["c_attn"],
    "fast": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "balanced": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "quality": ["q_proj", "v_proj", "k_proj", "o_proj"],
}


def get_lora_config(
    profile_name: str = "fast",
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Generate PEFT LoraConfig dynamically adapted to the specified profile."""
    try:
        from peft import LoraConfig, TaskType
    except ImportError as err:
        msg = (
            "peft is required for fine-tuning. "
            "Install with 'pip install aether-scientist[finetune]'"
        )
        raise ImportError(msg) from err

    prof = get_profile(profile_name)
    modules = target_modules or PROFILE_MODULES.get(prof.name, ["q_proj", "v_proj"])
    bias = kwargs.pop("bias", "none")

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=modules,
        bias=bias,
        **kwargs,
    )
