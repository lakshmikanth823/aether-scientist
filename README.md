# AetherScientist

Domain-specialized multimodal LLM framework for scientific research.

[![CI](https://github.com/aetherscientist/aether-scientist/actions/workflows/ci.yml/badge.svg)](https://github.com/aetherscientist/aether-scientist/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](https://github.com/aetherscientist/aether-scientist)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://pypi.org/project/aether-scientist/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ⚡ Quick Start

```bash
pip install -e ".[dev]"
```

## 💻 CLI Usage

```bash
# Ask a scientific question routed to domain specialist
aether ask "What is quantum entanglement?" --domain physics

# Analyze papers with specific focus
aether analyze -p paper1.pdf -p paper2.pdf --focus "reaction mechanisms"

# List loaded domain adapters
aether domains --json

# View configuration and model status
aether info
```

## 🌐 API Service

Start the FastAPI server:
```bash
python -m aether_scientist.api
# Or via uvicorn:
uvicorn aether_scientist.api.endpoints:app --host 0.0.0.0 --port 8000
```

Query the API via curl:
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: secret-aether-key" \
  -d '{"query": "Explain entropy in thermodynamics"}'
```

## 🐳 Docker Deployment

Run CLI in container:
```bash
docker build -t aether .
docker run --rm -v $(pwd)/models:/models aether ask "What is CRISPR-Cas9?" --domain biology
```

Or spin up the full API service with docker-compose:
```bash
docker compose up -d
curl http://localhost:8000/health
```

## 🧠 Python API

```python
from aether_scientist.core import AetherConfig, AetherScientist

config = AetherConfig(model_name="distilgpt2", domains=["physics", "chemistry"])
scientist = AetherScientist(config)

result = scientist.analyze("How does enthalpy change in exothermic reactions?")
print(result["analysis_result"]["generated_text"])
```

## 📜 Contributing & License

- Conventional Commits: `feat:`, `fix:`, `test:`, `docs:`, `chore:`
- Branch naming: `feature/[issue#]-[desc]`, `fix/[issue#]-[bug]`
- Licensed under the MIT License.
