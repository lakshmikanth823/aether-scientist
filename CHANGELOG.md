# Changelog

All notable changes to this project are documented here.
Format follows Keep a Changelog; versioning follows SemVer.

## [1.6.4] - 2026-09-06

### Fixed
- Stripped imperative query openings (`provide|summarize|explain... of|about|on`) in research planner topic extraction
- Added grammatical is/are and does/do agreement in planner sub-question templates for plural topics
- Trimmed dangling numbered list stubs (`: 1.`, ` 1. **`, etc.) and converted preceding trailing colons to periods
- Pre-stripped number stubs before degenerate output detection to prevent false charset collapse triggers

## [1.6.3] - 2026-09-06

### Fixed
- Out-of-vocabulary query handling: deleted artificial `mat[i, 0] = 1.0` fallback in `EmbeddingEngine`, keeping all-zero vector
- Zero-vector / low-score retrieval: `RAGEngine.retrieve()` returns empty list `[]` when query vector is all-zero or top score < 0.05
- Gated confidence sources: emptied `GroundedAnswer.sources` when confidence gate triggers, preserving raw hits in `retrieval_diagnostic`
- Context-window validation: dynamically resolved model context limit in `InferenceEngine.generate()` with prompt truncation and `IndexError` guard
- API error contracts: returned HTTP 409 Conflict when empty store is queried with RAG, and clean HTTP 500 without stack traces on unhandled exceptions
- Benchmark citation metrics: computed honest citation validity rate based on `[n]` bracket presence; set to `None` in quiz mode
- Streaming sources: emitted populated `sources` event payload in `/analyze/stream` when `use_rag=True`
- Markdown section tracking: recorded preceding `#` and `##` headings in `Chunk.section` across ingestion and chunking
- Multimodal error handling: provided distinguished error messages for missing vision dependencies vs model load failure
- Roadmap routes: returned HTTP 501 Not Implemented on `/synthesize`, `/hypothesize`, and `/experiment` routes
- Streamlit UI: updated `st.image` from deprecated `use_container_width=True` to `width="stretch"`

## [1.6.2] - 2026-09-05

### Fixed
- Sanitized extracted text: stripped literal special tokens (`<EOS>`, `<pad>`, `<s>`, `</s>`, `<|...|>`) before chunking
- Improved document title extraction: skipped license, arXiv, attribution, copyright, and permission boilerplate lines
- Chunker content deduplication: dropped identical whitespace-collapsed text chunks within documents
- Retrieval content-hash deduplication: merged identical text hits across store keeping highest similarity score
- Confidence gate: skipped LLM generation when mean top-k score < 0.35, returning honest insufficient-evidence response
- URL hallucination guard: stripped uncited web URLs from generated text and fell back to honest warning when fabricated
- Guaranteed single global profile selectbox across all Streamlit UI views
- Added `--rebuild` flag to CLI `ingest` command to wipe store before re-indexing

## [1.6.1] - 2026-09-05

### Fixed
- Routed all Streamlit views through a global sidebar profile selector defaulting to `fast`
- Displayed active profile, model, and compute device badges on generated outputs
- Deduplicated retrieved RAG hits and sources by `(doc_id, chunk_id)` maintaining score order
- Guarded against degenerate model output loops and charset collapse with fallback warnings
- Protected CLI against Windows console Unicode encoding errors on scientific symbols
- Added advisory stderr notification when `aether ask --rag` resolves to `offline` profile

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
