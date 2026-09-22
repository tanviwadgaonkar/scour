"""Turn raw text into a list of index terms.

Design decisions:
- Lowercase and split on anything that is not a letter or digit. Simple,
  fast, and good enough for English technical text.
- Stopwords are removed so the index does not fill up with postings for
  "the" and "of". Positions are assigned after removal, so a phrase query
  is matched on the same token stream the index was built from.
- No stemming yet. Stage 2 adds a Porter stemmer as a measured change:
  the evaluation harness reports whether it actually helps.
"""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have he in is it its of on or
    that the to was were will with this these those
    """.split()
)


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Return the ordered list of terms for ``text``."""
    tokens = TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
