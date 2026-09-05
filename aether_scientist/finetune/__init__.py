"""LoRA fine-tuning and scientific dataset processing for AetherScientist."""

from aether_scientist.finetune.config import get_lora_config
from aether_scientist.finetune.data import format_chatml_entry, format_dataset, load_scientific_data
from aether_scientist.finetune.trainer import run_finetune

__all__ = [
    "format_chatml_entry",
    "format_dataset",
    "get_lora_config",
    "load_scientific_data",
    "run_finetune",
]
