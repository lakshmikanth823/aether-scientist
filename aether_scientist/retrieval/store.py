import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from aether_scientist.retrieval.chunker import Chunk


@dataclass
class Hit:
    """Represents a vector search hit with similarity score."""

    chunk: Chunk
    doc_id: str
    source: str
    score: float
    title: str = ""


class VectorStore:
    """Numpy-backed in-memory vector store with cosine search and disk persistence."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.metadata: list[dict[str, str]] = []
        self.vectors: np.ndarray | None = None

    def add(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        sources: list[str] | None = None,
        titles: list[str] | None = None,
    ) -> int:
        """Add chunks and corresponding vectors to the store."""
        if len(chunks) == 0:
            return len(self.chunks)
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Vectors count {vectors.shape[0]} does not match chunks count {len(chunks)}"
            )

        srcs = sources or ["" for _ in chunks]
        ttls = titles or ["" for _ in chunks]
        for c, s, t in zip(chunks, srcs, ttls, strict=False):
            self.chunks.append(c)
            self.metadata.append({"source": s, "title": t})

        if self.vectors is None or self.vectors.size == 0:
            self.vectors = vectors.astype(np.float32)
        else:
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])

        return len(self.chunks)

    def search(self, query_vec: np.ndarray, k: int = 4) -> list[Hit]:
        """Perform linear cosine similarity search."""
        if self.vectors is None or len(self.chunks) == 0:
            return []

        q = query_vec.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        scores = np.dot(self.vectors, q.T).flatten()
        top_k = min(k, len(scores))
        if top_k <= 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        hits: list[Hit] = []
        for idx in top_indices:
            meta = self.metadata[idx] if idx < len(self.metadata) else {"source": "", "title": ""}
            hits.append(
                Hit(
                    chunk=self.chunks[idx],
                    doc_id=self.chunks[idx].doc_id,
                    source=meta.get("source", ""),
                    score=float(scores[idx]),
                    title=meta.get("title", ""),
                )
            )
        return hits

    def save(self, path: str | Path) -> None:
        """Save vectors and metadata to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        vec_file = p.with_suffix(".npz")
        meta_file = p.with_suffix(".json")

        vecs = self.vectors if self.vectors is not None else np.empty((0, 0), dtype=np.float32)
        np.savez_compressed(vec_file, vectors=vecs)

        data = {
            "chunks": [asdict(c) for c in self.chunks],
            "metadata": self.metadata,
        }
        meta_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        """Load vectors and metadata from disk."""
        p = Path(path)
        vec_file = p.with_suffix(".npz")
        meta_file = p.with_suffix(".json")

        if not vec_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"Vector store files not found at {p}")

        data = json.loads(meta_file.read_text(encoding="utf-8"))
        self.chunks = [Chunk(**item) for item in data.get("chunks", [])]
        self.metadata = data.get("metadata", [])

        with np.load(vec_file) as loader:
            self.vectors = loader["vectors"].astype(np.float32)

    def clear(self) -> None:
        """Reset the store."""
        self.chunks.clear()
        self.metadata.clear()
        self.vectors = None

    def __len__(self) -> int:
        return len(self.chunks)
