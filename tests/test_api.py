import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tests.conftest import DOCS  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    with open(tmp_path / "docs.jsonl", "w") as fh:
        for doc_id, title, text in DOCS:
            fh.write(json.dumps({"id": doc_id, "title": title, "text": text}) + "\n")
    monkeypatch.setenv("SCOUR_DATA", str(tmp_path))
    monkeypatch.setenv("SCOUR_NEURAL", "0")
    from scour import api

    api.STATE.clear()
    with TestClient(api.app) as c:
        yield c


def test_search_endpoint(client):
    r = client.get("/search", params={"q": "wing", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["hits"][0]["id"] == "3"
    assert body["hits"][0]["title"] == "Wing theory"
    assert "latency_ms" in body


def test_unknown_mode_rejected(client):
    assert client.get("/search", params={"q": "wing", "mode": "magic"}).status_code == 400


def test_home_page_renders(client):
    r = client.get("/")
    assert r.status_code == 200 and "scour" in r.text


def test_home_page_escapes_results(client):
    # Titles and snippets come from whatever corpus SCOUR_DATA points at.
    html = client.get("/").text
    assert "const esc=" in html
    assert "${h.snippet}" not in html and "${esc(h.snippet)}" in html


def test_health_reports_loaded_corpus(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "docs": 4, "modes": ["bm25"]}


def test_overlong_query_is_rejected(client):
    # Rerank cost grows with query length; cap it at the API boundary.
    assert client.get("/search", params={"q": "wing " * 200}).status_code == 422
