# Changelog

All notable changes to this project are documented here.
Format follows Keep a Changelog; versioning follows SemVer.

## [1.6.0] - 2026-09-05

### Added
- Streamlit interactive web interface with sidebar navigation router in `ui/app.py`
- Direct scientific chat view with real-time token streaming and model profile selection in `ui/views/chat.py`
- Knowledge base manager with document ingestion (PDF, TXT, MD) and grounded RAG query in `ui/views/knowledge.py`
- Autonomous research agent dashboard with live step execution tracking in `ui/views/research.py`
- Vision diagram tool with figure upload and VLM caption generation in `ui/views/vision.py`
- Optional `[ui]` extras group in `pyproject.toml` (`streamlit>=1.30`)
- 100% offline Streamlit `AppTest` suite covering all UI views and engine caching

## [1.5.0] - 2026-09-05


### Added
- PDF embedded image extraction pipeline using PyMuPDF (`fitz`) in `retrieval/images.py`
- Lightweight VLM diagram captioning engine with BLIP in `multimodal/captioner.py`
- Multimodal RAG ingestion injecting `[Figure {n}: {caption}]` annotations into document chunks
- CLI command `aether vision <image_path>` and REST endpoint `POST /vision`
- Optional `[vision]` extras group in `pyproject.toml` (`Pillow`, `transformers`)
- 100% offline unit and integration tests for figure extraction and vision RAG

## [1.4.0] - 2026-09-05

### Added
- Lightweight LoRA fine-tuning harness using Hugging Face `peft` and `trl` in `finetune/trainer.py`
- Scientific data formatter converting raw text and QA records to ChatML instruction format in `finetune/data.py`
- Dynamic `LoraConfig` generation tailored to model profiles (`fast`, `balanced`, `offline`) in `finetune/config.py`
- CLI command `aether finetune --data <path> --profile <profile> --epochs <n> --out <dir>`
- Optional `[finetune]` extras group in `pyproject.toml` (`peft`, `trl`, `datasets`, `bitsandbytes`)
- 100% offline test suite mocking `SFTTrainer`, model loading, and CPU guardrails

## [1.3.0] - 2026-09-05

### Added
- Web-search retrieval integration via `WebSearchEngine` with DuckDuckGo and Wikipedia fallback
- HTML page fetching and cleaning pipeline in `web_ingest.py` with `bs4` and regex tag stripping
- Automatic web fallback in `ResearchAgent` when local index is empty or low confidence with `--web`
- `--web` flag added to `aether research` and `aether ask` CLI commands
- `use_web` parameter added to `/research` and `/analyze` FastAPI endpoints
- Source provenance tracking with `[web]` tagging and direct URL references in research reports
- 100% offline test suite for web search, page cleaning, and agent web workflows

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
