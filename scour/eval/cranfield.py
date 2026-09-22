"""Download and convert the Cranfield collection (1,400 abstracts, 225 queries,
graded relevance judgements) into the JSONL/TSV layout the harness reads.

    python -m scour.eval.cranfield --out data/cranfield

Cranfield is small enough to run in seconds and old enough that its quirks
are well documented. Two of them matter here:

1. Relevance codes run 1 (complete answer) to 4 (minimal interest), with -1
   for "not relevant". We convert to gains 4..1 and 0 so nDCG rewards the
   best documents most.
2. Query ids in cranqrel are the 1-based *position* of the query in cran.qry,
   not the .I number printed inside the file. Getting this wrong silently
   halves every score.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import urllib.request
from pathlib import Path

URL = "http://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz"


def parse_cran_blocks(text: str) -> list[dict[str, str]]:
    """Parse the ".I / .T / .A / .B / .W" record format into dicts."""
    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    field = ""
    for line in text.splitlines():
        if line.startswith(".I"):
            if current is not None:
                records.append(current)
            current = {"I": line[2:].strip()}
            field = ""
        elif line[:2] in (".T", ".A", ".B", ".W"):
            field = line[1]
            current[field] = ""
        elif current is not None and field:
            current[field] += line + "\n"
    if current is not None:
        records.append(current)
    return records


def code_to_gain(code: int) -> int:
    return 5 - code if 1 <= code <= 4 else 0


def convert(all_text: str, qry_text: str, qrel_text: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "docs.jsonl", "w", encoding="utf-8") as fh:
        for rec in parse_cran_blocks(all_text):
            fh.write(json.dumps({
                "id": rec["I"],
                "title": " ".join(rec.get("T", "").split()),
                "text": " ".join(rec.get("W", "").split()),
            }) + "\n")

    with open(out / "queries.jsonl", "w", encoding="utf-8") as fh:
        for position, rec in enumerate(parse_cran_blocks(qry_text), 1):
            fh.write(json.dumps({"id": str(position), "text": " ".join(rec.get("W", "").split())}) + "\n")

    with open(out / "qrels.tsv", "w", encoding="utf-8") as fh:
        fh.write("query_id\tdoc_id\tgain\n")
        for line in qrel_text.splitlines():
            parts = line.split()
            if len(parts) != 3:
                continue
            qid, doc_id, code = parts
            fh.write(f"{qid}\t{doc_id}\t{code_to_gain(int(code))}\n")


def download(out: Path) -> None:
    print(f"downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=60) as resp:
        payload = resp.read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        members = {m.name.split("/")[-1]: m for m in tar.getmembers() if m.isfile()}
        def read(name: str) -> str:
            return tar.extractfile(members[name]).read().decode("utf-8", errors="replace")
        convert(read("cran.all.1400"), read("cran.qry"), read("cranqrel"), out)
    print(f"wrote docs.jsonl, queries.jsonl, qrels.tsv to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/cranfield")
    download(Path(parser.parse_args().out))
