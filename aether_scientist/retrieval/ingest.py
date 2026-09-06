import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether_scientist.multimodal.captioner import CaptionEngine
from aether_scientist.retrieval.chunker import chunk
from aether_scientist.retrieval.images import ImageAsset, extract_images

# Regex to strip literal special-token strings before chunking
_SPECIAL_TOKEN_RE = re.compile(r"<(?:EOS|pad|s|/s)>|<\|[^|]*\|>")

# Boilerplate patterns to skip when extracting title
_BOILERPLATE_RE = re.compile(
    r"(?i)(?:arxiv|license|copyright|attribution|creative\s+commons|preprint|"
    r"under\s+review|published\s+in|proceedings\s+of|permission|reproduce|"
    r"scholarly\s+works|all\s+rights\s+reserved|author\s+manuscript)",
)


def _sanitize_text(text: str) -> str:
    """Replace literal special-token strings with a space."""
    return _SPECIAL_TOKEN_RE.sub(" ", text)


def _extract_title(text: str, pdf_meta_title: str, stem: str) -> str:
    """Extract best-effort document title."""
    if pdf_meta_title and pdf_meta_title.strip():
        return pdf_meta_title.strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if not _BOILERPLATE_RE.search(ln) and len(ln) > 3:
            return ln.lstrip("#").strip()
    return stem or "Untitled"


@dataclass
class Document:
    """Represents an ingested scientific document."""

    doc_id: str
    source: str
    title: str
    text: str
    chunks: list[Any] = field(default_factory=list)
    images: list[ImageAsset] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)


def ingest_file(
    path: str | Path,
    cache_dir: str | Path = ".aether_cache",
    extract_figs: bool = True,
) -> list[Document]:
    """Ingest a PDF or plain text/markdown file into Document representations."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise RuntimeError(
                "PyMuPDF is required for PDF ingestion. Install with: pip install pymupdf"
            ) from e

        doc = fitz.open(p)
        meta_title = doc.metadata.get("title", "") if doc.metadata else ""

        images: list[ImageAsset] = []
        if extract_figs:
            images = extract_images(p, cache_dir=cache_dir)

        page_images: dict[int, list[ImageAsset]] = {}
        for img in images:
            page_images.setdefault(img.page_num, []).append(img)

        pages_text: list[str] = []
        captions: list[str] = []
        captioner = CaptionEngine()
        fig_idx = 1

        for page_num in range(1, len(doc) + 1):
            p_text = doc[page_num - 1].get_text()
            if page_num in page_images:
                for img_asset in page_images[page_num]:
                    cap = captioner.caption(img_asset.image_path)
                    captions.append(cap)
                    p_text += f"\n\n[Figure {fig_idx}: {cap}]"
                    fig_idx += 1
            pages_text.append(p_text)

        raw_text = "\n".join(pages_text)
        text = _sanitize_text(raw_text)
        doc.close()

        title = _extract_title(text, meta_title, p.stem)
        doc_id = hashlib.sha256(f"{p.name}:{p.stat().st_size}".encode()).hexdigest()[:12]
        chunks = chunk(text, doc_id=doc_id)
        return [
            Document(
                doc_id=doc_id,
                source=str(p),
                title=title,
                text=text,
                chunks=chunks,
                images=images,
                captions=captions,
            )
        ]

    raw_text = p.read_text(encoding="utf-8", errors="replace")
    text = _sanitize_text(raw_text)
    title = _extract_title(text, "", p.stem)
    doc_id = hashlib.sha256(f"{p.name}:{p.stat().st_size}".encode()).hexdigest()[:12]
    chunks = chunk(text, doc_id=doc_id)
    return [Document(doc_id=doc_id, source=str(p), title=title, text=text, chunks=chunks)]
