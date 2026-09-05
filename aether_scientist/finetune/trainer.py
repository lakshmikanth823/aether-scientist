import inspect
import logging
from pathlib import Path
from typing import Any

from aether_scientist.core.profiles import get_profile
from aether_scientist.finetune.config import get_lora_config
from aether_scientist.finetune.data import format_dataset, load_scientific_data

logger = logging.getLogger(__name__)


def run_finetune(
    data_path: str,
    profile_name: str = "fast",
    output_dir: str = "./lora_output",
    epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
) -> dict[str, Any]:
    """Execute lightweight LoRA instruction fine-tuning on custom scientific data."""
    try:
        import datasets
        import peft
        import torch
        import transformers
        import trl
    except ImportError as err:
        msg = (
            "Fine-tuning dependencies missing. "
            "Install with 'pip install aether-scientist[finetune]'"
        )
        raise ImportError(msg) from err

    raw = load_scientific_data(data_path)
    prof = get_profile(profile_name)
    dataset = format_dataset(raw, prof)
    if not dataset:
        raise ValueError(f"No valid records found in {data_path}")

    has_cuda = torch.cuda.is_available()
    if not has_cuda and len(dataset) > 100:
        logger.warning("No GPU detected and dataset > 100 rows. Fine-tuning aborted.")
        return {
            "status": "aborted",
            "reason": "No GPU detected and dataset exceeds 100 rows",
            "profile": profile_name,
            "samples": len(dataset),
        }

    tokenizer = transformers.AutoTokenizer.from_pretrained(prof.model)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None)

    model_kwargs: dict[str, Any] = {}
    if has_cuda:
        model_kwargs["device_map"] = "auto"
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
        except Exception:
            model_kwargs["torch_dtype"] = torch.float16

    base = transformers.AutoModelForCausalLM.from_pretrained(prof.model, **model_kwargs)
    lora_cfg = get_lora_config(profile_name)
    model = peft.get_peft_model(base, lora_cfg)

    train_data = datasets.Dataset.from_list(dataset)
    sft_cls = getattr(trl, "SFTConfig", transformers.TrainingArguments)
    args_kwargs: dict[str, Any] = {
        "output_dir": output_dir, "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size, "learning_rate": learning_rate,
        "logging_steps": 10, "save_strategy": "no", "use_cpu": not has_cuda,
    }
    if getattr(sft_cls, "__name__", "") == "SFTConfig":
        args_kwargs["max_seq_length"] = max_seq_length
    args = sft_cls(**args_kwargs)

    trainer_kwargs: dict[str, Any] = {"model": model, "train_dataset": train_data, "args": args}
    try:
        params = inspect.signature(trl.SFTTrainer.__init__).parameters
        if "processing_class" in params:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in params:
            trainer_kwargs["tokenizer"] = tokenizer
    except (ValueError, TypeError):
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = trl.SFTTrainer(**trainer_kwargs)
    trainer.train()

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    if hasattr(trainer, "model") and hasattr(trainer.model, "save_pretrained"):
        trainer.model.save_pretrained(str(out_p))
    elif hasattr(model, "save_pretrained"):
        model.save_pretrained(str(out_p))

    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(out_p))

    return {
        "status": "completed",
        "profile": profile_name,
        "output_dir": str(out_p),
        "epochs": epochs,
        "samples": len(dataset),
    }
