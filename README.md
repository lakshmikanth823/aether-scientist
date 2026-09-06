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

# Autonomous multi-step research agent
aether research "CRISPR gene editing mechanisms" --out report.md

# System diagnostics and configuration
aether info
```

## 🔬 Autonomous Research Agent

AetherScientist features a multi-step research pipeline (`plan -> retrieve -> read -> synthesize -> cite`) that surveys your indexed literature and compiles a cited Markdown report:

```bash
# 1. Ingest literature or notes
aether ingest notes.md paper1.pdf

# 2. Run autonomous research agent
aether research "CRISPR gene editing mechanisms" --profile balanced --out report.md
```

### Agent CLI Options:
- `--profile <name>`: Model preset (`offline`, `fast`, `balanced`, `quality`). Default: `balanced`.
- `-k <int>`: Top passages retrieved per sub-question (default: 3).
- `--web`: Enable global web search & scraping retrieval when local index is empty or confidence is low.
- `--out <path>`: Write rendered report to a Markdown file.
- `--json`: Output structured JSON (state, sub-queries, references, coverage).
- `--llm-planner`: Use model-driven query planning instead of heuristic template planner.

## 🌐 Web-Enabled Research

When local literature is unavailable or insufficient, `--web` activates global research mode:
```bash
# Query live scientific literature from the web
aether research "Perovskite solar cell efficiency" --web --profile balanced --out solar.md

# Ask single grounded question with web fallback
aether ask "What is the latest tokamak plasma pressure record?" --web --profile fast
```
- **Lightweight Architecture**: Driven by `httpx` and `BeautifulSoup4` with zero heavy browser automation.
- **Automated Fallback**: If local vector store is empty, `--web` automatically queries DuckDuckGo/Wikipedia, scrapes relevant pages, cleans HTML, and chunks passages.
- **Citation Provenance**: Web sources are explicitly tagged with `[web]` in findings and resolved to original page URLs in references.


### Sample Report Excerpt:
```markdown
# Research Report: CRISPR gene editing mechanisms

## Executive Summary
Recent findings demonstrate precise double-strand breaks guided by sgRNA [1].

## 1. Underlying Principles & Mechanisms
Cas9 endonuclease induces targeted cleavage at protospacer adjacent motifs [1][2].

## References
- [1] notes.md (score: 0.88)
- [2] paper1.pdf (score: 0.82)
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

## 🎯 LoRA Fine-Tuning

Fine-tune instruction models (`fast` 360M, `balanced` 0.5B) on custom scientific corpora:

```bash
# 1. Install optional fine-tuning extras
pip install "aether-scientist[finetune]"

# 2. Run LoRA SFT on custom scientific JSON/JSONL dataset
aether finetune --data ./custom_papers.json --profile fast --epochs 1 --out ./lora_adapter
```

```python
from aether_scientist.finetune import run_finetune

res = run_finetune("./custom_papers.json", profile_name="fast", output_dir="./lora_adapter")
print("Adapter saved:", res["output_dir"])
```

## 🖼️ Multimodal & Diagram Understanding

Extract embedded PDF figures, generate VLM captions, and index them into RAG:

```bash
# 1. Install optional vision dependencies
pip install "aether-scientist[vision]"

# 2. Test captioner on an image or diagram
aether vision figure1.png

# 3. Ingesting PDFs automatically extracts images and captions diagrams
aether ingest paper.pdf
```

```python
from aether_scientist.multimodal.captioner import CaptionEngine

caption = CaptionEngine().caption("diagram.png")
print("Caption:", caption)
```

## 🌐 Interactive Web Interface

Launch the interactive Streamlit web dashboard for chat, RAG document indexing, live agent execution tracking, and diagram understanding:

```bash
pip install "aether-scientist[ui]"
streamlit run ui/app.py
```
> ![AetherScientist Web UI](docs/assets/ui-screenshot.png)

## 🗺️ Roadmap

The following advanced discovery capabilities are currently under active design for future releases:
- `/synthesize`: Cross-domain literature synthesis and consensus mapping (`501 Not Implemented — on roadmap`).
- `/hypothesize`: Automated scientific hypothesis generation from observation graphs (`501 Not Implemented — on roadmap`).
- `/experiment`: Laboratory experiment design and protocol generation (`501 Not Implemented — on roadmap`).

## 📜 Contributing & License

- Conventional Commits: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`
- Full details in [CHANGELOG.md](CHANGELOG.md).
- Licensed under the [MIT License](LICENSE).
