"""The query front door: BM25 ranking plus quoted phrase filtering.

    engine.search('"boundary layer" transition')

Words inside quotes must appear next to each other, in order, in the
document. The whole query (phrase words included) is still scored by
BM25; the phrase only decides which documents are allowed in.
"""

from __future__ import annotations

import heapq
import re
from dataclasses import dataclass

from scour.bm25 import BM25
from scour.index import InvertedIndex
from scour.tokenizer import tokenize

PHRASE_RE = re.compile(r'"([^"]+)"')


@dataclass(slots=True)
class Hit:
    doc_id: str
    score: float
    title: str = ""


def phrase_matches(index: InvertedIndex, phrase: str) -> set[str]:
    """Doc ids in which the phrase's terms occur consecutively."""
    terms = tokenize(phrase)
    if not terms:
        return set()
    if len(terms) == 1:
        return {p.doc_id for p in index.get_postings(terms[0])}

    # Map doc_id -> positions for each term; start from the rarest term.
    per_term = [{p.doc_id: p.positions for p in index.get_postings(t)} for t in terms]
    candidates = set(min(per_term, key=len).keys())
    for positions in per_term:
        candidates &= positions.keys()

    matches: set[str] = set()
    for doc_id in candidates:
        first = per_term[0][doc_id]
        rest = [set(per_term[i][doc_id]) for i in range(1, len(terms))]
        for start in first:
            if all(start + offset in rest[offset - 1] for offset in range(1, len(terms))):
                matches.add(doc_id)
                break
    return matches


class SearchEngine:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75) -> None:
        self.index = index
        self.ranker = BM25(index, k1=k1, b=b)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        phrases = PHRASE_RE.findall(query)
        scores = self.ranker.score_all(query)

        if phrases:
            allowed: set[str] | None = None
            for phrase in phrases:
                docs = phrase_matches(self.index, phrase)
                allowed = docs if allowed is None else allowed & docs
            scores = {d: s for d, s in scores.items() if d in (allowed or set())}

        top = heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], kv[0]))
        return [Hit(doc_id, score, self.index.titles.get(doc_id, "")) for doc_id, score in top]
