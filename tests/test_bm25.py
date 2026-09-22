import math

import pytest

from scour.bm25 import BM25
from scour.tfidf import TFIDF


def test_idf_matches_formula(index):
    bm25 = BM25(index)
    n, N = 3, 4  # "boundary" appears in 3 of 4 docs
    assert bm25.idf("boundary") == pytest.approx(math.log(1 + (N - n + 0.5) / (n + 0.5)))


def test_rare_term_outranks_common_term(index):
    # "wing" is rare (1 doc); "boundary" is common (3 docs).
    results = BM25(index).search("wing boundary", k=4)
    assert results[0][0] == "3"


def test_repeated_term_scores_higher_but_saturates(index):
    bm25 = BM25(index)
    scores = bm25.score_all("boundary")
    assert scores["4"] > scores["1"]  # tf=2 beats tf=1
    # Saturation: doubling tf does not double the score.
    assert scores["4"] < 2 * scores["1"]


def test_unknown_terms_are_ignored(index):
    assert BM25(index).search("zzzz") == []


def test_score_matches_hand_computation(index):
    bm25 = BM25(index, k1=1.5, b=0.75)
    tf, dl, avgdl = 1, index.doc_lengths["3"], index.avg_doc_length
    idf = math.log(1 + (4 - 1 + 0.5) / (1 + 0.5))
    expected = idf * tf * (1.5 + 1) / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))
    assert bm25.score_all("wing")["3"] == pytest.approx(expected)


def test_tfidf_baseline_runs(index):
    results = TFIDF(index).search("wing", k=2)
    assert results[0][0] == "3"
