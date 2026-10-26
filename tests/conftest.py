import pytest

from scour.index import InvertedIndex

DOCS = [
    ("1", "Boundary layer", "the boundary layer on a flat plate in supersonic flow"),
    ("2", "Heat transfer", "heat transfer in the boundary layer of a blunt body"),
    ("3", "Wing theory", "lift on a swept wing at low speed with flat plate theory"),
    ("4", "Shock waves", "shock wave interaction with a turbulent boundary layer boundary layer growth"),
]


@pytest.fixture
def index() -> InvertedIndex:
    return InvertedIndex.build(DOCS)


@pytest.fixture
def body_index() -> InvertedIndex:
    """Bodies only: for tests that hand-check positions, lengths and scores."""
    return InvertedIndex.build(DOCS, index_titles=False)
