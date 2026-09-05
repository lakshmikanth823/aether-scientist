# AetherScientist

Domain-specialized scientific research LLM framework with RAG, citation grounding, and live streaming.

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

Docker one-liner:
```bash
docker run --rm -it -v $(pwd)/models:/models aether ask "What is entropy?" --domain physics
```

## 🔑 Environment Variables

- `AETHER_API_KEY`: API authentication key for FastAPI endpoints (defaults to `dev-aether-key` for local development).
- `HF_HOME`: Directory for caching model weights (default: `~/.cache/huggingface`).

## 💻 CLI Usage

```bash
# Ingest and index scientific literature into knowledge store
aether ingest paper1.pdf notes.md

# Search indexed knowledge base
aether search "quantum entanglement" -k 3

# Ask with grounded RAG and live token streaming
aether ask "What does the paper conclude?" --rag --stream

# Run benchmark evaluation across scientific domains
aether bench --n 20 --json

# Domain inspection and runtime info
aether domains && aether info
```

## 🌐 API & Live Streaming

Start server:
```bash
python -m aether_scientist.api
# Or via uvicorn:
uvicorn aether_scientist.api.endpoints:app --host 0.0.0.0 --port 8000
```

Live Server-Sent Events (SSE) token stream:
```bash
curl -N -X POST http://localhost:8000/analyze/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${AETHER_API_KEY:-dev-aether-key}" \
  -d '{"query": "Explain entropy in thermodynamics"}'
```

## 🧠 Python API with Grounded RAG

```python
from aether_scientist.core import AetherConfig, AetherScientist

config = AetherConfig(model_name="distilgpt2")
scientist = AetherScientist(config)

# Index literature and ask grounded questions
scientist.rag_engine.index(["literature/quantum_physics.pdf"])
result = scientist.analyze("Explain quantum decoherence", use_rag=True)

print("Answer:", result["analysis_result"]["generated_text"])
print("Confidence:", result["analysis_result"]["confidence"])
for src in result["analysis_result"]["sources"]:
    print(f"[{src['title']}]: {src['snippet']}")
```

## 🐳 Docker Deployment

```bash
docker compose up -d
curl http://localhost:8000/health
```

## 📜 Contributing & License

- Conventional Commits: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`
- Full details in [CHANGELOG.md](CHANGELOG.md).
- Licensed under the [MIT License](LICENSE).
