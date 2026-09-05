# Changelog

All notable changes to this project are documented here.
Format follows Keep a Changelog; versioning follows SemVer.

## [1.0.0] - 2026-09-05

### Added
- Scientific tokenizer with LaTeX, chemistry, citation, and notation support
- Domain adapter architecture (physics, chemistry, biology)
- Lazy Hugging Face inference engine with streaming
- RAG pipeline: ingestion, chunking, embeddings (sbert/TF-IDF), vector store
- Citation grounding with hallucination guard and confidence scoring
- FastAPI with rate limiting, auth, SSE streaming endpoint
- Typer CLI: ask, analyze, domains, info, ingest, search, bench
- Benchmark harness with bundled 20-question scientific quiz
- Docker multi-stage build and docker-compose
- 56 tests, ruff-clean, <0.3s import time
