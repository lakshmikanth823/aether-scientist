import hashlib
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a bounded text chunk from a document."""

    chunk_id: str
    doc_id: str
    text: str
    index: int
    section: str = ""


def _split_sentences(paragraph: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    return [s.strip() for s in sentences if s.strip()]


def chunk(text: str, size: int = 512, overlap: int = 64, doc_id: str = "") -> list[Chunk]:
    """Chunk text by paragraph then sentence boundaries with specified overlap."""
    if not text.strip():
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    heading_re = re.compile(r"^(#{1,6}\s+[^\n]+)")
    units: list[tuple[str, str, bool]] = []
    current_sec = ""

    for p in paragraphs:
        p_has_new_heading = False
        for line in p.splitlines():
            m = heading_re.match(line.strip())
            if m:
                current_sec = m.group(1).strip()
                p_has_new_heading = True

        if len(p) <= size:
            units.append((p, current_sec, p_has_new_heading))
        else:
            first_s = True
            for s in _split_sentences(p):
                m = heading_re.match(s.strip())
                if m:
                    current_sec = m.group(1).strip()
                    p_has_new_heading = True
                if len(s) <= size:
                    units.append((s, current_sec, p_has_new_heading if first_s else False))
                else:
                    step = max(1, size - overlap)
                    for i in range(0, len(s), step):
                        units.append(
                            (
                                s[i : i + size],
                                current_sec,
                                p_has_new_heading if first_s and i == 0 else False,
                            )
                        )
                first_s = False

    chunks: list[Chunk] = []
    current: list[tuple[str, str]] = []
    current_len = 0
    idx = 0

    for unit_text, unit_sec, is_new_heading in units:
        add_len = len(unit_text) + (1 if current else 0)
        if current and (current_len + add_len > size or is_new_heading):
            chunk_text = " ".join(u[0] for u in current)
            chunk_sec = next((u[1] for u in reversed(current) if u[1]), "")
            cid = f"{doc_id}_{idx}" if doc_id else str(idx)
            chunks.append(
                Chunk(chunk_id=cid, doc_id=doc_id, text=chunk_text, index=idx, section=chunk_sec)
            )
            idx += 1

            if is_new_heading:
                current = []
                current_len = 0
            else:
                overlap_units: list[tuple[str, str]] = []
                accum_len = 0
                for u in reversed(current):
                    needed = len(u[0]) + (1 if overlap_units else 0)
                    if accum_len + needed <= overlap:
                        overlap_units.insert(0, u)
                        accum_len += needed
                    else:
                        break
                current = overlap_units
                current_len = sum(len(u[0]) for u in current) + max(0, len(current) - 1)

        current.append((unit_text, unit_sec))
        current_len += len(unit_text) + (1 if len(current) > 1 else 0)

    if current:
        chunk_text = " ".join(u[0] for u in current)
        chunk_sec = next((u[1] for u in reversed(current) if u[1]), "")
        cid = f"{doc_id}_{idx}" if doc_id else str(idx)
        chunks.append(
            Chunk(chunk_id=cid, doc_id=doc_id, text=chunk_text, index=idx, section=chunk_sec)
        )

    # Content-level dedup: drop chunks with identical whitespace-collapsed text
    seen_hashes: set[str] = set()
    unique: list[Chunk] = []
    for c in chunks:
        h = hashlib.md5(" ".join(c.text.split()).encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(c)

    return unique
