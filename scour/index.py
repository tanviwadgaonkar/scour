"""An inverted index: for every term, the list of documents it appears in.

Layout in memory::

    postings["boundary"] -> [Posting(doc_id="12", tf=3, positions=[4, 19, 40]), ...]
    doc_lengths["12"]    -> 57

Positions are kept so phrase queries can check that terms are adjacent.

Titles are indexed together with the body (title tokens first, then a
one-position gap so a phrase can never span title and body). Both rankers
therefore see the same title + text, and title terms naturally carry extra
weight. On Cranfield, where every abstract repeats its title, this lifts BM25
nDCG@10 from 0.334 to 0.348; ``index_titles=False`` reproduces the text-only
baseline.

Persistence uses pickle: only load index files you built yourself.
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
    FORMAT_VERSION = 2

    def __init__(self, index_titles: bool = True) -> None:
        self.index_titles = index_titles
        self.postings: dict[str, list[Posting]] = defaultdict(list)
        self.doc_lengths: dict[str, int] = {}
        self.titles: dict[str, str] = {}

    # ---- building -------------------------------------------------------

    def add_document(self, doc_id: str, text: str, title: str = "") -> None:
        if doc_id in self.doc_lengths:
            raise ValueError(f"duplicate doc_id: {doc_id}")
        title_tokens = tokenize(title) if (title and self.index_titles) else []
        body_tokens = tokenize(text)
        positions: dict[str, list[int]] = defaultdict(list)
        for pos, term in enumerate(title_tokens):
            positions[term].append(pos)
        offset = len(title_tokens) + 1 if title_tokens else 0  # gap: no cross-boundary phrases
        for pos, term in enumerate(body_tokens):
            positions[term].append(offset + pos)
        for term, pos_list in positions.items():
            self.postings[term].append(Posting(doc_id, len(pos_list), pos_list))
        self.doc_lengths[doc_id] = len(title_tokens) + len(body_tokens)
        if title:
            self.titles[doc_id] = title

    @classmethod
    def build(cls, docs: Iterable[tuple[str, str, str]], index_titles: bool = True) -> "InvertedIndex":
        """Build from an iterable of (doc_id, title, text)."""
        index = cls(index_titles=index_titles)
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
                {"version": self.FORMAT_VERSION, "index_titles": self.index_titles,
                 "postings": dict(self.postings), "doc_lengths": self.doc_lengths, "titles": self.titles},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "InvertedIndex":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        if data.get("version") != cls.FORMAT_VERSION:
            raise ValueError(f"{path}: index format {data.get('version', 1)} is not "
                             f"{cls.FORMAT_VERSION}; rebuild it with `python -m scour index`")
        index = cls(index_titles=data["index_titles"])
        index.postings.update(data["postings"])
        index.doc_lengths = data["doc_lengths"]
        index.titles = data["titles"]
        return index
