from pydantic import BaseModel, Field


class TokenizerConfig(BaseModel):
    vocab_size: int = 50000
    special_tokens: list[str] = Field(
        default_factory=lambda: ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[EQ]"]
    )


class EfficiencyConfig(BaseModel):
    use_flash_attention: bool = True
    kv_cache_quantization: bool = False
    compile_model: bool = False


class RagConfig(BaseModel):
    chunk_size: int = 512
    overlap: int = 64
    top_k: int = 4
    backend: str = "auto"
    cache_dir: str = ".aether_cache"


class AetherConfig(BaseModel):
    model_name: str = "distilgpt2"
    model_size: str = "82M"
    quantization_bits: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.7
    domains: list[str] = Field(default_factory=lambda: ["physics", "chemistry", "biology"])
    reasoning_depth: int = 8
    max_papers: int = 100
    max_sequence_length: int = 4096
    batch_size: int = 8
    device: str = "auto"  # auto-detect GPU/CPU
    cache_enabled: bool = True
    cache_size: int = 1000
    api_rate_limit: int = 60  # requests per minute
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    efficiency: EfficiencyConfig = Field(default_factory=EfficiencyConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
