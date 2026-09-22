"""A plain TF-IDF scorer, kept as the baseline BM25 is measured against.

score(D, Q) = sum over query terms t of tf(t, D) * ln(N / n(t))

No term-frequency saturation and no length normalisation, which is exactly
what BM25 adds. The gap between the two on the benchmark is the point.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict

from scour.index import InvertedIndex
from scour.tokenizer import tokenize


class TFIDF:
    def __init__(self, index: InvertedIndex) -> None:
        self.index = index

    def score_all(self, query: str) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        N = self.index.num_docs
        for term in set(tokenize(query)):
            postings = self.index.get_postings(term)
            if not postings:
                continue
            idf = math.log(N / len(postings))
            for p in postings:
                scores[p.doc_id] += p.tf * idf
        return scores

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return heapq.nlargest(k, self.score_all(query).items(), key=lambda kv: (kv[1], kv[0]))
