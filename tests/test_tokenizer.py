from scour.tokenizer import tokenize


def test_lowercases_and_splits_on_punctuation():
    assert tokenize("Boundary-Layer, FLOW!") == ["boundary", "layer", "flow"]


def test_removes_stopwords_by_default():
    assert tokenize("the flow of the fluid") == ["flow", "fluid"]


def test_keeps_stopwords_when_asked():
    assert tokenize("the flow", remove_stopwords=False) == ["the", "flow"]


def test_keeps_digits():
    assert tokenize("mach 2.5 flow") == ["mach", "2", "5", "flow"]
