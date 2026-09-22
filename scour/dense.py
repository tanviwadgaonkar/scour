"""Dense retrieval: embed every document once, embed the query at search time,
rank by cosine similarity.

The embedder is an interface so the index can be tested with a toy embedder
and run for real with a sentence-transformers model. Vectors are unit
normalised at build time, so cosine similarity is a single matrix-vector dot.
"""

from __future__ import annotations

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

    # Cache the embeddings so evaluation runs do not re-embed the corpus.
    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.save(path.with_suffix(".npy"), self.vectors)
        path.with_suffix(".json").write_text(json.dumps(self.doc_ids))

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder) -> "DenseIndex":
        path = Path(path)
        return cls(json.loads(path.with_suffix(".json").read_text()), np.load(path.with_suffix(".npy")), embedder)
