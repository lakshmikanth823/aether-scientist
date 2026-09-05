import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Dual-backend text embedding engine supporting sentence-transformers and TF-IDF."""

    def __init__(self, backend: str = "auto", cache_dir: str = ".aether_cache") -> None:
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._sbert_model: Any = None
        self._active_backend: str | None = None
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray = np.array([])
        self._load_vocab()

    def _load_vocab(self) -> None:
        vocab_file = self.cache_dir / "tfidf_vocab.json"
        idf_file = self.cache_dir / "tfidf_idf.npy"
        if vocab_file.exists() and idf_file.exists():
            try:
                self._vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
                self._idf = np.load(idf_file)
            except Exception as e:
                logger.warning(f"Could not load cached TF-IDF vocab: {e}")

    def _save_vocab(self) -> None:
        try:
            (self.cache_dir / "tfidf_vocab.json").write_text(
                json.dumps(self._vocab), encoding="utf-8"
            )
            np.save(self.cache_dir / "tfidf_idf.npy", self._idf)
        except Exception as e:
            logger.warning(f"Could not save TF-IDF vocab: {e}")

    def _init_sbert(self) -> bool:
        if self._sbert_model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer

            self._sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
            return True
        except Exception as e:
            logger.info(f"sentence-transformers unavailable, falling back to tfidf: {e}")
            return False

    def _get_backend(self) -> str:
        if self._active_backend:
            return self._active_backend
        if self.backend == "sbert":
            if not self._init_sbert():
                raise RuntimeError("sentence-transformers is required for sbert backend.")
            self._active_backend = "sbert"
        elif self.backend == "tfidf":
            self._active_backend = "tfidf"
        else:  # auto
            self._active_backend = "sbert" if self._init_sbert() else "tfidf"
        return self._active_backend

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())

    def _embed_tfidf(self, texts: list[str]) -> np.ndarray:
        tokenized = [self._tokenize(t) for t in texts]
        if not self._vocab:
            all_tokens = sorted({t for doc in tokenized for t in doc})
            self._vocab = {t: idx for idx, t in enumerate(all_tokens)}
            n_docs = len(texts)
            df = np.zeros(len(self._vocab), dtype=np.float32)
            for doc in tokenized:
                for t in set(doc):
                    if t in self._vocab:
                        df[self._vocab[t]] += 1
            self._idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
            self._save_vocab()

        vocab_size = max(1, len(self._vocab))
        mat = np.zeros((len(texts), vocab_size), dtype=np.float32)
        for i, doc in enumerate(tokenized):
            for t in doc:
                if t in self._vocab:
                    mat[i, self._vocab[t]] += 1.0
            mask = mat[i] > 0
            mat[i, mask] = (1.0 + np.log(mat[i, mask])) * self._idf[mask]
            norm = np.linalg.norm(mat[i])
            if norm > 0:
                mat[i] /= norm
            else:
                mat[i, 0] = 1.0
        return mat

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings into normalized 2D numpy array."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        backend = self._get_backend()
        if backend == "sbert":
            vecs = self._sbert_model.encode(texts, normalize_embeddings=True)
            return np.asarray(vecs, dtype=np.float32)
        return self._embed_tfidf(texts)
