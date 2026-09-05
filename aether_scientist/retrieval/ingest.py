import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """Represents an ingested scientific document."""

    doc_id: str
    source: str
    title: str
    text: str
    chunks: list[Any] = field(default_factory=list)


def ingest_file(path: str | Path) -> list[Document]:
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
        text = "\n".join(page.get_text() for page in doc)
        doc.close()

        if not title:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            title = lines[0] if lines else p.stem

        doc_id = hashlib.sha256(f"{p.name}:{p.stat().st_size}".encode()).hexdigest()[:12]
        return [Document(doc_id=doc_id, source=str(p), title=title, text=text)]

    text = p.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    title = lines[0].lstrip("#").strip() if lines else p.stem
    doc_id = hashlib.sha256(f"{p.name}:{p.stat().st_size}".encode()).hexdigest()[:12]
    return [Document(doc_id=doc_id, source=str(p), title=title, text=text)]
