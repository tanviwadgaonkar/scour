import math

import pytest

from scour.eval.metrics import average_precision, ndcg_at_k, precision_at_k, recall_at_k


REL = {"a": 3, "b": 1, "c": 2}  # d and e are not relevant


def test_precision_and_recall():
    ranked = ["a", "d", "b", "e", "c"]
    assert precision_at_k(ranked, REL, 2) == 0.5
    assert recall_at_k(ranked, REL, 3) == pytest.approx(2 / 3)


def test_average_precision():
    ranked = ["a", "d", "b"]
    # hits at rank 1 (1/1) and rank 3 (2/3), divided by 3 relevant docs
    assert average_precision(ranked, REL) == pytest.approx((1 + 2 / 3) / 3)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "c", "b"], REL, 3) == pytest.approx(1.0)


def test_ndcg_worse_ranking_is_lower():
    assert ndcg_at_k(["b", "c", "a"], REL, 3) < 1.0


def test_ndcg_hand_computed():
    ranked = ["b", "a"]
    dcg = 1 / math.log2(2) + 3 / math.log2(3)
    idcg = 3 / math.log2(2) + 2 / math.log2(3)
    assert ndcg_at_k(ranked, REL, 2) == pytest.approx(dcg / idcg)
