"""BM25 ranking over an InvertedIndex.

score(D, Q) = sum over query terms t of

    idf(t) * tf(t, D) * (k1 + 1)
    -------------------------------------------
    tf(t, D) + k1 * (1 - b + b * |D| / avgdl)

idf(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

k1 controls how quickly repeated occurrences of a term stop adding score
(term-frequency saturation). b controls how much long documents are
penalised (length normalisation). The defaults k1=1.5, b=0.75 are the
usual starting point; the evaluation harness lets you sweep them.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict

from scour.index import InvertedIndex
from scour.tokenizer import tokenize


class BM25:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75) -> None:
        self.index = index
        self.k1 = k1
        self.b = b

    def idf(self, term: str) -> float:
        n = self.index.document_frequency(term)
        N = self.index.num_docs
        return math.log(1 + (N - n + 0.5) / (n + 0.5))

    def score_all(self, query: str) -> dict[str, float]:
        """Return a score for every document that shares at least one term with the query."""
        scores: dict[str, float] = defaultdict(float)
        avgdl = self.index.avg_doc_length
        for term in set(tokenize(query)):
            postings = self.index.get_postings(term)
            if not postings:
                continue
            idf = self.idf(term)
            for p in postings:
                dl = self.index.doc_lengths[p.doc_id]
                denom = p.tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                scores[p.doc_id] += idf * p.tf * (self.k1 + 1) / denom
        return scores

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self.score_all(query)
        # Tie-break on doc_id so results are deterministic.
        return heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], kv[0]))
