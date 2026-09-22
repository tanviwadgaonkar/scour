"""Standard IR metrics. ``ranked`` is the list of doc ids the engine returned,
best first. ``rels`` maps doc id -> relevance grade (0 means not relevant).
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def precision_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if rels.get(d, 0) > 0) / k


def recall_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    relevant = {d for d, g in rels.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def average_precision(ranked: Sequence[str], rels: Mapping[str, int]) -> float:
    relevant = {d for d, g in rels.items() if g > 0}
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def dcg_at_k(gains: Sequence[int], k: int) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains[:k], 1))


def ndcg_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    gains = [rels.get(d, 0) for d in ranked]
    ideal = sorted((g for g in rels.values() if g > 0), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(gains, k) / idcg
