# Changelog

All notable changes to this project are documented here.
Format follows Keep a Changelog; versioning follows SemVer.

## [1.2.1] - 2026-09-05

### Fixed
- Handled vector dimension mismatch in `VectorStore.add` and `VectorStore.search` when switching embedding backends
- Silenced Hugging Face Hub progress bars and authentication warnings in CLI JSON output
- Ensured offline test isolation in `tests/test_streaming.py`

### Added
- Real-run empirical benchmarks documented in `BENCHMARKS.md`
- Verbatim real-run evidence in `docs/examples/` (generation, RAG on Attention paper, research report)

## [1.2.0] - 2026-09-05

### Added
- Autonomous research agent pipeline (`plan -> retrieve -> synthesize -> report`)
- Heuristic query planner generating targeted sub-questions with optional LLM planner mode
- Research state tracking deduplicated passages, 1-based global citation indexing, and topic coverage
- Markdown report renderer with validated inline citations and References section
- CLI command `aether research` with live step progress, `--out`, `--k`, and `--json` support
- REST API endpoint `POST /research` and SSE streaming endpoint `POST /research/stream`
- 100% offline agent test suite (planner, state, report, pipeline, CLI, API)

## [1.1.0] - 2026-09-05

### Added
- Model profile registry with four presets (`offline`, `fast`, `balanced`, `quality`)
- Chat-template prompt formatting and dynamic stop-marker output cleaning
- Compute device auto-detection across CUDA, Apple Silicon MPS, and CPU
- Multi-profile comparative benchmark mode in `aether bench --profile ...`
- Profiles management CLI command `aether profiles`
- API profile validation, `GET /profiles` endpoint, and response metadata

### Changed
- Inference engine is now profile-based with backward-compatible `model_name` mapping
- Test suite expanded to 80 tests with 100% offline coverage

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
