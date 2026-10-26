import json

from scour.eval.cranfield import code_to_gain, convert, parse_cran_blocks

ALL = """.I 1
.T
experimental investigation of the aerodynamics of a
wing in a slipstream .
.A
brenckman,m.
.B
j. ae. scs. 25, 1958, 324.
.W
experimental investigation of the aerodynamics of a
wing in a slipstream .
.I 2
.T
simple shear flow past a flat plate .
.A
ting-yili
.B
dept. of aero. eng.
.W
simple shear flow past a flat plate in an incompressible fluid .
"""

QRY = """.I 001
.W
what similarity laws must be obeyed .
.I 004
.W
what problems of heat conduction arise .
"""

QREL = """1 2 3
1 1 -1
2 2 1
"""


def test_parse_blocks():
    recs = parse_cran_blocks(ALL)
    assert [r["I"] for r in recs] == ["1", "2"]
    assert "flat plate" in recs[1]["W"]
    assert recs[0]["A"].strip() == "brenckman,m."


def test_gain_mapping():
    assert [code_to_gain(c) for c in (1, 2, 3, 4, -1)] == [4, 3, 2, 1, 0]


def test_convert_uses_positional_query_ids(tmp_path):
    convert(ALL, QRY, QREL, tmp_path)
    queries = [json.loads(l) for l in (tmp_path / "queries.jsonl").read_text().splitlines()]
    # second query is ".I 004" in the file but id "2" positionally
    assert [q["id"] for q in queries] == ["1", "2"]
    docs = [json.loads(l) for l in (tmp_path / "docs.jsonl").read_text().splitlines()]
    assert docs[1]["title"] == "simple shear flow past a flat plate ."
    qrels = (tmp_path / "qrels.tsv").read_text().splitlines()
    assert qrels[1] == "1\t2\t2"
    assert qrels[2] == "1\t1\t0"


def test_download_rejects_checksum_mismatch(tmp_path, monkeypatch):
    import io
    import pytest
    from scour.eval import cranfield

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(cranfield.urllib.request, "urlopen", lambda *a, **k: FakeResponse(b"tampered"))
    with pytest.raises(SystemExit, match="checksum mismatch"):
        cranfield.download(tmp_path)
    assert not (tmp_path / "docs.jsonl").exists()
