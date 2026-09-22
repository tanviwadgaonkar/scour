import pytest

from scour.index import InvertedIndex


def test_document_frequency(index):
    assert index.document_frequency("boundary") == 3
    assert index.document_frequency("wing") == 1
    assert index.document_frequency("missing") == 0


def test_term_frequency_and_positions(index):
    posting = {p.doc_id: p for p in index.get_postings("boundary")}["4"]
    assert posting.tf == 2
    assert posting.positions == [4, 6]


def test_doc_lengths_exclude_stopwords(index):
    # "the boundary layer on a flat plate in supersonic flow" -> the/on/a/in removed
    assert index.doc_lengths["1"] == len(["boundary", "layer", "flat", "plate", "supersonic", "flow"])


def test_stats(index):
    assert index.num_docs == 4
    assert index.avg_doc_length == pytest.approx(sum(index.doc_lengths.values()) / 4)


def test_duplicate_doc_id_rejected(index):
    with pytest.raises(ValueError):
        index.add_document("1", "again")


def test_save_and_load_roundtrip(index, tmp_path):
    path = tmp_path / "index.pkl"
    index.save(path)
    loaded = InvertedIndex.load(path)
    assert loaded.num_docs == index.num_docs
    assert loaded.document_frequency("boundary") == 3
    assert loaded.titles["1"] == "Boundary layer"
