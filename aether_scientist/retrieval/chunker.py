import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a bounded text chunk from a document."""

    chunk_id: str
    doc_id: str
    text: str
    index: int


def _split_sentences(paragraph: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    return [s.strip() for s in sentences if s.strip()]


def chunk(text: str, size: int = 512, overlap: int = 64, doc_id: str = "") -> list[Chunk]:
    """Chunk text by paragraph then sentence boundaries with specified overlap."""
    if not text.strip():
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for p in paragraphs:
        if len(p) <= size:
            units.append(p)
        else:
            for s in _split_sentences(p):
                if len(s) <= size:
                    units.append(s)
                else:
                    step = max(1, size - overlap)
                    units.extend(s[i : i + size] for i in range(0, len(s), step))

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    idx = 0

    for unit in units:
        add_len = len(unit) + (1 if current else 0)
        if current and (current_len + add_len > size):
            chunk_text = " ".join(current)
            cid = f"{doc_id}_{idx}" if doc_id else str(idx)
            chunks.append(Chunk(chunk_id=cid, doc_id=doc_id, text=chunk_text, index=idx))
            idx += 1

            overlap_units: list[str] = []
            accum_len = 0
            for u in reversed(current):
                needed = len(u) + (1 if overlap_units else 0)
                if accum_len + needed <= overlap:
                    overlap_units.insert(0, u)
                    accum_len += needed
                else:
                    break
            current = overlap_units
            current_len = sum(len(u) for u in current) + max(0, len(current) - 1)

        current.append(unit)
        current_len += len(unit) + (1 if len(current) > 1 else 0)

    if current:
        chunk_text = " ".join(current)
        cid = f"{doc_id}_{idx}" if doc_id else str(idx)
        chunks.append(Chunk(chunk_id=cid, doc_id=doc_id, text=chunk_text, index=idx))

    return chunks
