"""Dense, hybrid and rerank logic tested with toy models, so the suite stays
offline and fast. The real sentence-transformer adapters are exercised by the
evaluation run, not by unit tests."""

import numpy as np
import pytest

from scour.bm25 import BM25
from scour.dense import DenseIndex
from scour.hybrid import HybridRanker, reciprocal_rank_fusion
from scour.rerank import RerankedRanker
from tests.conftest import DOCS

VOCAB = ["boundary", "layer", "heat", "wing", "shock", "plate"]


class BagOfWordsEmbedder:
    """Counts vocabulary words: a stand-in with the same interface as a real model."""

    def encode(self, texts):
        return np.array([[t.lower().split().count(w) for w in VOCAB] for t in texts], dtype=np.float32)


@pytest.fixture
def dense():
    return DenseIndex.build(DOCS, BagOfWordsEmbedder())


def test_dense_ranks_by_cosine(dense):
    top = dense.search("wing", k=1)
    assert top[0][0] == "3"
    assert 0.0 < top[0][1] <= 1.0


def test_dense_roundtrip(dense, tmp_path):
    dense.save(tmp_path / "idx")
    loaded = DenseIndex.load(tmp_path / "idx", BagOfWordsEmbedder())
    assert loaded.search("heat", k=1)[0][0] == dense.search("heat", k=1)[0][0]


def test_rrf_hand_computed():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["a"] == pytest.approx(fused["b"])


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]])
    assert max(fused, key=fused.get) == "a"


def test_hybrid_combines_rankers(index, dense):
    hybrid = HybridRanker([BM25(index), dense], depth=4)
    ids = [d for d, _ in hybrid.search("boundary layer heat", k=4)]
    assert set(ids) == {"1", "2", "4"} | ({"3"} if "3" in ids else set())
    assert ids[0] in {"2", "4"}


class ReverseScorer:
    def score(self, query, docs):
        return list(range(len(docs)))  # last candidate gets the highest score


def test_reranker_reorders_first_stage(index):
    bm25 = BM25(index)
    texts = {doc_id: text for doc_id, _, text in DOCS}
    first = [d for d, _ in bm25.search("boundary", k=3)]
    reranked = [d for d, _ in RerankedRanker(bm25, ReverseScorer(), texts, top_n=3).search("boundary", k=3)]
    assert reranked == list(reversed(first))


def test_reranker_empty_query(index):
    assert RerankedRanker(BM25(index), ReverseScorer(), {}, top_n=3).search("zzzz") == []
