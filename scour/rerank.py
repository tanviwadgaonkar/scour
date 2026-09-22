"""Second-stage reranking with a cross-encoder.

A cross-encoder reads the query and a document together and outputs one
relevance score. It is far more accurate than comparing two separate
embeddings, and far more expensive: one forward pass per (query, doc) pair.
That is why it runs over the top-N of a cheaper first stage, never over the
whole corpus.
"""

from __future__ import annotations

import heapq
from typing import Mapping, Protocol

from scour.hybrid import Ranker


class PairScorer(Protocol):
    def score(self, query: str, docs: list[str]) -> list[float]: ...


class CrossEncoderScorer:
    """Adapter for sentence-transformers CrossEncoder. The MS MARCO MiniLM
    model is small enough to score 50 pairs on CPU in well under a second."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder  # optional dependency

        self.model = CrossEncoder(model_name)

    def score(self, query: str, docs: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, d) for d in docs], show_progress_bar=False)]


class RerankedRanker:
    def __init__(self, first_stage: Ranker, scorer: PairScorer, texts: Mapping[str, str], top_n: int = 50) -> None:
        self.first_stage = first_stage
        self.scorer = scorer
        self.texts = texts
        self.top_n = top_n

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        candidates = [doc_id for doc_id, _ in self.first_stage.search(query, k=self.top_n)]
        if not candidates:
            return []
        scores = self.scorer.score(query, [self.texts[d] for d in candidates])
        return heapq.nlargest(k, zip(candidates, scores), key=lambda kv: (kv[1], kv[0]))
