"""Command line entry point.

    python -m scour index  --docs data/cranfield/docs.jsonl --out index.pkl
    python -m scour search --index index.pkl "boundary layer"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from scour.index import InvertedIndex
from scour.search import SearchEngine


def read_docs(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                yield d["id"], d.get("title", ""), d["text"]


def cmd_index(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    index = InvertedIndex.build(read_docs(Path(args.docs)))
    index.save(args.out)
    elapsed = time.perf_counter() - start
    print(f"indexed {index.num_docs} docs, {index.num_terms} terms in {elapsed:.2f}s -> {args.out}")


def cmd_search(args: argparse.Namespace) -> None:
    engine = SearchEngine(InvertedIndex.load(args.index), k1=args.k1, b=args.b)
    start = time.perf_counter()
    hits = engine.search(args.query, k=args.k)
    elapsed_ms = (time.perf_counter() - start) * 1000
    for rank, hit in enumerate(hits, 1):
        title = f"  {hit.title}" if hit.title else ""
        print(f"{rank:2d}. {hit.doc_id:>8}  {hit.score:6.3f}{title}")
    print(f"({len(hits)} results, {elapsed_ms:.2f} ms)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="scour")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build an index from a JSONL file of {id, title, text}")
    p_index.add_argument("--docs", required=True)
    p_index.add_argument("--out", default="index.pkl")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="query an index")
    p_search.add_argument("query")
    p_search.add_argument("--index", default="index.pkl")
    p_search.add_argument("-k", type=int, default=10)
    p_search.add_argument("--k1", type=float, default=1.5)
    p_search.add_argument("--b", type=float, default=0.75)
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
