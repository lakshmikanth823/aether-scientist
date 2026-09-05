# AetherScientist

Domain-specialized scientific research LLM framework with profile presets, chat formatting, RAG, citation grounding, and live streaming.

[![PyPI version](https://img.shields.io/pypi/v/aether-scientist.svg)](https://pypi.org/project/aether-scientist/)
[![CI](https://github.com/aetherscientist/aether-scientist/actions/workflows/ci.yml/badge.svg)](https://github.com/aetherscientist/aether-scientist/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/aether-scientist/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](https://github.com/aetherscientist/aether-scientist)

## 📦 Installation

```bash
# Core installation (zero-dependency TF-IDF retrieval + CLI + API)
pip install aether-scientist

# With optional SBERT embeddings and PDF ingestion support
pip install "aether-scientist[rag]"

# Development setup with test suite and linters
pip install -e ".[dev,rag]"
```

## 🤖 Model Profiles

AetherScientist features 4 built-in inference presets configured for scientific workflows:

| Profile | Hugging Face Model | Chat Template | Tokens | Temp | Best For |
|---|---|---|---|---|---|
| `offline` | `distilgpt2` | No | 128 | 0.7 | Fast offline development & CI testing |
| `fast` | `HuggingFaceTB/SmolLM2-360M-Instruct` | Yes | 256 | 0.3 | 360M instruct, CPU-friendly & lightweight |
| `balanced` | `Qwen/Qwen2.5-0.5B-Instruct` | Yes | 384 | 0.3 | Best speed/accuracy balance for desktop |
| `quality` | `Qwen/Qwen2.5-1.5B-Instruct` | Yes | 512 | 0.2 | Highest precision reasoning & literature synthesis |

List profiles from CLI:
```bash
aether profiles
```

> [!NOTE]
> Gated models (e.g. Llama-3, Mistral) are not bundled presets. To run gated checkpoints, authenticate with Hugging Face using `export HF_TOKEN="your_hf_token"`.

## ⚡ Device Support Matrix

InferenceEngine automatically detects and allocates compute accelerators:

| Platform | Accelerator | Detection Target | Fallback |
|---|---|---|---|
| NVIDIA GPU | CUDA | `torch.cuda.is_available()` | CPU |
| Apple Silicon | MPS | `torch.backends.mps.is_available()` | CPU |
| General / x86 / ARM | CPU | Standard PyTorch tensor execution | Safe zero-crash mode |

Verify active device: `aether info`

## 🔑 Environment Variables

- `AETHER_MODEL`: Overrides the default model profile (e.g. `export AETHER_MODEL=fast`).
- `AETHER_API_KEY`: API authentication key for FastAPI endpoints (defaults to `dev-aether-key`).
- `AETHER_ALLOW_DOWNLOADS`: Gatekeeper flag (`1` to permit downloading remote weights during benchmarks).
- `HF_TOKEN`: Personal Hugging Face access token for gated models.
- `HF_HOME`: Directory for caching model weights (default: `~/.cache/huggingface`).

## 💻 CLI Usage

```bash
# Ask with an instruct profile preset
aether ask "What is quantum decoherence?" --profile fast

# Ingest and index scientific literature into knowledge store
aether ingest paper1.pdf notes.md

# Search indexed knowledge base
aether search "quantum entanglement" -k 3

# Ask with grounded RAG and live token streaming
aether ask "What does the paper conclude?" --profile balanced --rag --stream

# Run multi-profile comparative benchmark
aether bench --profile offline --profile fast --n 20

# System diagnostics and configuration
aether info
```

## 🌐 API & Live Streaming

Start the API server:
```bash
python -m aether_scientist.api
# Or via uvicorn:
uvicorn aether_scientist.api.endpoints:app --host 0.0.0.0 --port 8000
```

Fetch available profiles:
```bash
curl http://localhost:8000/profiles
```

Live Server-Sent Events (SSE) token stream with profile selection:
```bash
curl -N -X POST http://localhost:8000/analyze/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${AETHER_API_KEY:-dev-aether-key}" \
  -d '{"query": "Explain entropy in thermodynamics", "profile": "fast"}'
```

## 🧠 Python API with Grounded RAG

```python
from aether_scientist.core import AetherConfig, AetherScientist

config = AetherConfig(profile="balanced")
scientist = AetherScientist(config)

# Index literature and ask grounded questions
scientist.rag_engine.index(["literature/quantum_physics.pdf"])
result = scientist.analyze("Explain quantum decoherence", use_rag=True)

print("Answer:", result["analysis_result"]["generated_text"])
print("Confidence:", result["analysis_result"]["confidence"])
print("Profile:", result["profile"])
print("Device:", result["device"])
```

## 📜 Contributing & License

- Conventional Commits: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`
- Full details in [CHANGELOG.md](CHANGELOG.md).
- Licensed under the [MIT License](LICENSE).
