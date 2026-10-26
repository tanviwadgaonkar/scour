import pytest

from scour.index import InvertedIndex


def test_document_frequency(index):
    assert index.document_frequency("boundary") == 3
    assert index.document_frequency("wing") == 1
    assert index.document_frequency("missing") == 0


def test_term_frequency_and_positions(body_index):
    posting = {p.doc_id: p for p in body_index.get_postings("boundary")}["4"]
    assert posting.tf == 2
    assert posting.positions == [4, 6]


def test_doc_lengths_exclude_stopwords(body_index):
    # "the boundary layer on a flat plate in supersonic flow" -> the/on/a/in removed
    assert body_index.doc_lengths["1"] == len(["boundary", "layer", "flat", "plate", "supersonic", "flow"])


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


def test_titles_are_indexed_before_the_body(index, body_index):
    # Doc 3: title "Wing theory", body "lift on a swept wing ...": "theory" occurs in both.
    posting = {p.doc_id: p for p in index.get_postings("theory")}["3"]
    assert posting.tf == 2
    assert posting.positions[0] == 1  # second title token
    assert index.doc_lengths["3"] == 2 + body_index.doc_lengths["3"]


def test_title_and_body_positions_leave_a_gap():
    idx = InvertedIndex.build([("1", "boundary", "layer flow")])
    [b] = idx.get_postings("boundary")
    [l] = idx.get_postings("layer")
    assert l.positions[0] - b.positions[-1] > 1  # not adjacent


def test_old_index_format_is_rejected(tmp_path):
    import pickle
    path = tmp_path / "v1.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"postings": {}, "doc_lengths": {}, "titles": {}}, fh)
    with pytest.raises(ValueError, match="rebuild"):
        InvertedIndex.load(path)


def test_save_and_load_keeps_title_setting(body_index, tmp_path):
    path = tmp_path / "index.pkl"
    body_index.save(path)
    assert InvertedIndex.load(path).index_titles is False
