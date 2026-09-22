"""An inverted index: for every term, the list of documents it appears in.

Layout in memory::

    postings["boundary"] -> [Posting(doc_id="12", tf=3, positions=[4, 19, 40]), ...]
    doc_lengths["12"]    -> 57

Positions are kept so phrase queries can check that terms are adjacent.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from scour.tokenizer import tokenize


@dataclass(slots=True)
class Posting:
    doc_id: str
    tf: int
    positions: list[int] = field(default_factory=list)


class InvertedIndex:
    def __init__(self) -> None:
        self.postings: dict[str, list[Posting]] = defaultdict(list)
        self.doc_lengths: dict[str, int] = {}
        self.titles: dict[str, str] = {}

    # ---- building -------------------------------------------------------

    def add_document(self, doc_id: str, text: str, title: str = "") -> None:
        if doc_id in self.doc_lengths:
            raise ValueError(f"duplicate doc_id: {doc_id}")
        tokens = tokenize(text)
        positions: dict[str, list[int]] = defaultdict(list)
        for pos, term in enumerate(tokens):
            positions[term].append(pos)
        for term, pos_list in positions.items():
            self.postings[term].append(Posting(doc_id, len(pos_list), pos_list))
        self.doc_lengths[doc_id] = len(tokens)
        if title:
            self.titles[doc_id] = title

    @classmethod
    def build(cls, docs: Iterable[tuple[str, str, str]]) -> "InvertedIndex":
        """Build from an iterable of (doc_id, title, text)."""
        index = cls()
        for doc_id, title, text in docs:
            index.add_document(doc_id, text, title)
        return index

    # ---- statistics -----------------------------------------------------

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)

    @property
    def num_terms(self) -> int:
        return len(self.postings)

    @property
    def avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def document_frequency(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def get_postings(self, term: str) -> list[Posting]:
        return self.postings.get(term, [])

    # ---- persistence ----------------------------------------------------

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(
                {"postings": dict(self.postings), "doc_lengths": self.doc_lengths, "titles": self.titles},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "InvertedIndex":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        index = cls()
        index.postings.update(data["postings"])
        index.doc_lengths = data["doc_lengths"]
        index.titles = data["titles"]
        return index
