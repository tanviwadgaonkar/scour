"""Dense retrieval: embed every document once, embed the query at search time,
rank by cosine similarity.

The embedder is an interface so the index can be tested with a toy embedder
and run for real with a sentence-transformers model. Vectors are unit
normalised at build time, so cosine similarity is a single matrix-vector dot.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Adapter for sentence-transformers. all-MiniLM-L6-v2 is 22M parameters,
    384 dimensions, and embeds 1,400 abstracts on a laptop CPU in well under a minute."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=64, show_progress_bar=False), dtype=np.float32)


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class DenseIndex:
    def __init__(self, doc_ids: list[str], vectors: np.ndarray, embedder: Embedder) -> None:
        self.doc_ids = doc_ids
        self.vectors = _normalise(vectors.astype(np.float32))
        self.embedder = embedder

    @classmethod
    def build(cls, docs: Iterable[tuple[str, str, str]], embedder: Embedder) -> "DenseIndex":
        doc_ids, texts = [], []
        for doc_id, title, text in docs:
            doc_ids.append(doc_id)
            texts.append(f"{title}. {text}" if title else text)
        return cls(doc_ids, embedder.encode(texts), embedder)

    def score_all(self, query: str) -> dict[str, float]:
        q = _normalise(self.embedder.encode([query]))[0]
        sims = self.vectors @ q
        return {doc_id: float(s) for doc_id, s in zip(self.doc_ids, sims)}

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return heapq.nlargest(k, self.score_all(query).items(), key=lambda kv: (kv[1], kv[0]))

    # Cache the embeddings so evaluation runs do not re-embed the corpus. The
    # cache records a fingerprint of the corpus and the model; a cache built
    # from different documents or a different model is never reused.
    def save(self, path: str | Path, fingerprint: str = "") -> None:
        path = Path(path)
        np.save(path.with_suffix(".npy"), self.vectors)
        path.with_suffix(".json").write_text(json.dumps({"fingerprint": fingerprint, "doc_ids": self.doc_ids}))

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder, fingerprint: str = "") -> "DenseIndex | None":
        """The cached index, or None if missing or built for another corpus/model."""
        path = Path(path)
        meta_path, vec_path = path.with_suffix(".json"), path.with_suffix(".npy")
        if not (meta_path.exists() and vec_path.exists()):
            return None
        meta = json.loads(meta_path.read_text())
        if not isinstance(meta, dict) or meta.get("fingerprint") != fingerprint:
            return None
        return cls(meta["doc_ids"], np.load(vec_path), embedder)

    @classmethod
    def cached(cls, path: str | Path, docs: list[tuple[str, str, str]], embedder: Embedder,
               model_name: str = "") -> "DenseIndex":
        """Load the cache if it matches ``docs`` and ``model_name``; otherwise build and save it."""
        fp = corpus_fingerprint(docs, model_name)
        index = cls.load(path, embedder, fp)
        if index is None:
            index = cls.build(docs, embedder)
            index.save(path, fp)
        return index


def corpus_fingerprint(docs: list[tuple[str, str, str]], model_name: str = "") -> str:
    h = hashlib.sha256(model_name.encode("utf-8"))
    for doc_id, title, text in docs:
        for part in (doc_id, title, text):
            h.update(part.encode("utf-8") + b"\x00")
    return h.hexdigest()
