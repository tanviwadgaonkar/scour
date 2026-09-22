"""Hybrid retrieval: fuse a lexical ranking (BM25) and a dense ranking with
reciprocal rank fusion.

    rrf(d) = sum over rankers r of 1 / (k + rank_r(d))

RRF is used instead of adding raw scores because BM25 scores and cosine
similarities live on different scales; ranks are comparable, scores are not.
k=60 is the value from the original paper (Cormack et al., 2009) and works
well without tuning.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Protocol, Sequence


class Ranker(Protocol):
    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]: ...


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            fused[doc_id] += 1.0 / (k + rank)
    return dict(fused)


class HybridRanker:
    def __init__(self, rankers: Sequence[Ranker], depth: int = 100, rrf_k: int = 60) -> None:
        self.rankers = list(rankers)
        self.depth = depth
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        rankings = [[doc_id for doc_id, _ in r.search(query, k=self.depth)] for r in self.rankers]
        fused = reciprocal_rank_fusion(rankings, self.rrf_k)
        return heapq.nlargest(k, fused.items(), key=lambda kv: (kv[1], kv[0]))
