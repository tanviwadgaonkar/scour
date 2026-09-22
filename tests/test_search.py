from scour.search import SearchEngine, phrase_matches


def test_phrase_requires_adjacency(index):
    # "flat plate" is adjacent in docs 1 and 3; "plate flat" nowhere.
    assert phrase_matches(index, "flat plate") == {"1", "3"}
    assert phrase_matches(index, "plate flat") == set()


def test_phrase_filters_results(index):
    engine = SearchEngine(index)
    hits = engine.search('"heat transfer" boundary')
    assert [h.doc_id for h in hits] == ["2"]


def test_plain_query_returns_hits_with_titles(index):
    hits = SearchEngine(index).search("boundary layer", k=3)
    assert len(hits) == 3
    assert hits[0].title


def test_phrase_with_no_matches_returns_nothing(index):
    assert SearchEngine(index).search('"supersonic wing"') == []
