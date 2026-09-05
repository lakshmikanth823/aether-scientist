import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether_scientist.multimodal.captioner import CaptionEngine
from aether_scientist.retrieval.images import ImageAsset, extract_images


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
        title = doc.metadata.get("title") if doc.metadata else ""

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

        text = "\n".join(pages_text)
        doc.close()

        if not title:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            title = lines[0] if lines else p.stem

        doc_id = hashlib.sha256(f"{p.name}:{p.stat().st_size}".encode()).hexdigest()[:12]
        return [
            Document(
                doc_id=doc_id,
                source=str(p),
                title=title,
                text=text,
                images=images,
                captions=captions,
            )
        ]

    text = p.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    title = lines[0].lstrip("#").strip() if lines else p.stem
    doc_id = hashlib.sha256(f"{p.name}:{p.stat().st_size}".encode()).hexdigest()[:12]
    return [Document(doc_id=doc_id, source=str(p), title=title, text=text)]
